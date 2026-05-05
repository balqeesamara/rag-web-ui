import hashlib
from typing import List, Any, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from sqlalchemy import text
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy.orm import selectinload
import time
import asyncio

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import KnowledgeBase, Document, ProcessingTask, DocumentChunk, DocumentUpload
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DocumentResponse,
    PreviewRequest
)
from app.services.document_processor import process_document_background, upload_document, preview_document, PreviewResult, SUPPORTED_EXTENSIONS
from app.core.config import settings
from app.core.storage import save_file, delete_kb_files, delete_file


router = APIRouter()

logger = logging.getLogger(__name__)

class TestRetrievalRequest(BaseModel):
    query: str
    kb_id: int
    top_k: int

@router.post("", response_model=KnowledgeBaseResponse)
def create_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_in: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Create new knowledge base.
    """
    kb = KnowledgeBase(
        name=kb_in.name,
        description=kb_in.description,
        user_id=current_user.id
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"Knowledge base created: {kb.name} for user {current_user.id}")
    return kb

@router.get("", response_model=List[KnowledgeBaseResponse])
def get_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
) -> Any:
    """
    Retrieve knowledge bases.
    """
    knowledge_bases = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return knowledge_bases

@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get knowledge base by ID.
    """
    from sqlalchemy.orm import joinedload
    
    kb = (
        db.query(KnowledgeBase)
        .options(
            joinedload(KnowledgeBase.documents)
            .joinedload(Document.processing_tasks)
        )
        .filter(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return kb

@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    kb_in: KnowledgeBaseUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Update knowledge base.
    """
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    for field, value in kb_in.dict(exclude_unset=True).items():
        setattr(kb, field, value)

    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"Knowledge base updated: {kb.name} for user {current_user.id}")
    return kb

@router.delete("/{kb_id}")
async def delete_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Delete knowledge base and all associated resources.
    """
    logger = logging.getLogger(__name__)
    
    kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        # Get all document file paths before deletion
        document_paths = [doc.file_path for doc in kb.documents]
        
        # Clean up external resources first
        cleanup_errors = []

        # 1. Clean up stored files
        try:
            delete_kb_files(current_user.id, kb_id)
            logger.info(f"Cleaned up storage files for knowledge base {kb_id}")
        except Exception as e:
            cleanup_errors.append(f"Failed to clean up storage files: {str(e)}")
            logger.error(f"Storage cleanup error for kb {kb_id}: {str(e)}")
        
        # 2. Clean up Qdrant collection
        try:
            qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            qdrant.delete_collection(f"kb_{kb_id}")
            logger.info(f"Cleaned up Qdrant collection for knowledge base {kb_id}")
        except Exception as e:
            cleanup_errors.append(f"Failed to clean up Qdrant collection: {str(e)}")
            logger.error(f"Qdrant cleanup error for kb {kb_id}: {str(e)}")
        
        # Finally, delete database records in a single transaction
        db.delete(kb)
        db.commit()
        
        # Report any cleanup errors in the response
        if cleanup_errors:
            return {
                "message": "Knowledge base deleted with cleanup warnings",
                "warnings": cleanup_errors
            }
        
        return {"message": "Knowledge base and all associated resources deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge base: {str(e)}")

# Batch upload documents
@router.post("/{kb_id}/documents/upload")
async def upload_kb_documents(
    kb_id: int,
    files: List[UploadFile],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload multiple documents to local storage.
    """
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = []
    for file in files:
        # Reject unsupported extensions early with a clear error
        _, ext = os.path.splitext(file.filename or "")
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # 1. 计算文件 hash
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # 2. 检查是否存在完全相同的文件（名称和hash都相同）
        existing_document = db.query(Document).filter(
            Document.file_name == file.filename,
            Document.file_hash == file_hash,
            Document.knowledge_base_id == kb_id
        ).first()
        
        if existing_document:
            # 完全相同的文件，直接返回
            results.append({
                "document_id": existing_document.id,
                "file_name": existing_document.file_name,
                "status": "exists",
                "message": "文件已存在且已处理完成",
                "skip_processing": True
            })
            continue
        
        # 3. Save to temp directory
        temp_path = f"user_{current_user.id}/kb_{kb_id}/temp/{file.filename}"
        try:
            file_size = len(file_content)
            save_file(temp_path, file_content)
        except Exception as e:
            logger.error(f"Failed to save file to storage: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload file")
        
        # 4. 创建上传记录
        upload = DocumentUpload(
            knowledge_base_id=kb_id,
            file_name=file.filename,
            file_hash=file_hash,
            file_size=len(file_content),
            content_type=file.content_type,
            temp_path=temp_path
        )
        db.add(upload)
        db.commit()
        db.refresh(upload)
        
        results.append({
            "upload_id": upload.id,
            "file_name": file.filename,
            "temp_path": temp_path,
            "status": "pending",
            "skip_processing": False
        })
    
    return results

@router.post("/{kb_id}/documents/preview")
async def preview_kb_documents(
    kb_id: int,
    preview_request: PreviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[int, PreviewResult]:
    """
    Preview multiple documents' chunks.
    """
    results = {}
    for doc_id in preview_request.document_ids:
        document = db.query(Document).join(KnowledgeBase).filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == current_user.id
        ).first()
        
        if document:
            file_path = document.file_path
        else:
            upload = db.query(DocumentUpload).join(KnowledgeBase).filter(
                DocumentUpload.id == doc_id,
                DocumentUpload.knowledge_base_id == kb_id,
                KnowledgeBase.user_id == current_user.id
            ).first()
            
            if not upload:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
            file_path = upload.temp_path
        
        preview = await preview_document(
            file_path,
            chunk_size=preview_request.chunk_size,
            chunk_overlap=preview_request.chunk_overlap
        )
        results[doc_id] = preview
    
    return results

@router.post("/{kb_id}/documents/process")
async def process_kb_documents(
    kb_id: int,
    upload_results: List[dict],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process multiple documents asynchronously.
    """
    start_time = time.time()
    
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    task_info = []
    upload_ids = []
    
    for result in upload_results:
        if result.get("skip_processing"):
            continue
        upload_ids.append(result["upload_id"])
    
    if not upload_ids:
        return {"tasks": []}
    
    uploads = db.query(DocumentUpload).filter(
        DocumentUpload.id.in_(upload_ids),
        DocumentUpload.knowledge_base_id == kb_id
    ).all()
    uploads_dict = {upload.id: upload for upload in uploads}
    if len(uploads_dict) != len(set(upload_ids)):
        raise HTTPException(status_code=400, detail="One or more upload IDs are invalid")
    
    all_tasks = []
    for upload_id in upload_ids:
        upload = uploads_dict.get(upload_id)
        if not upload:
            continue
            
        task = ProcessingTask(
            document_upload_id=upload_id,
            knowledge_base_id=kb_id,
            status="pending"
        )
        all_tasks.append(task)
    
    db.add_all(all_tasks)
    db.commit()
    
    for task in all_tasks:
        db.refresh(task)
    
    task_data = []
    for i, upload_id in enumerate(upload_ids):
        if i < len(all_tasks):
            task = all_tasks[i]
            upload = uploads_dict.get(upload_id)
            
            task_info.append({
                "upload_id": upload_id,
                "task_id": task.id
            })
            
            if upload:
                task_data.append({
                    "task_id": task.id,
                    "upload_id": upload_id,
                    "temp_path": upload.temp_path,
                    "file_name": upload.file_name
                })
    
    background_tasks.add_task(
        add_processing_tasks_to_queue,
        task_data,
        kb_id,
        current_user.id
    )
    
    return {"tasks": task_info}

async def add_processing_tasks_to_queue(task_data, kb_id, user_id):
    """Helper function to add document processing tasks to the queue without blocking the main response."""
    for data in task_data:
        asyncio.create_task(
            process_document_background(
                data["temp_path"],
                data["file_name"],
                kb_id,
                data["task_id"],
                None,
                user_id
            )
        )
    logger.info(f"Added {len(task_data)} document processing tasks to queue")

@router.post("/cleanup")
async def cleanup_temp_files(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Clean up expired temporary files.
    """
    expired_time = datetime.utcnow() - timedelta(hours=24)
    expired_uploads = db.query(DocumentUpload).filter(
        DocumentUpload.created_at < expired_time
    ).all()
    
    for upload in expired_uploads:
        try:
            delete_file(upload.temp_path)
        except Exception as e:
            logger.error(f"Failed to delete temp file {upload.temp_path}: {str(e)}")

        db.delete(upload)
    
    db.commit()
    
    return {"message": f"Cleaned up {len(expired_uploads)} expired uploads"}

@router.get("/{kb_id}/documents/tasks")
async def get_processing_tasks(
    kb_id: int,
    task_ids: str = Query(..., description="Comma-separated list of task IDs to check status for"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get status of multiple processing tasks.
    """
    task_id_list = [int(id.strip()) for id in task_ids.split(",")]
    
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id
    ).first()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
        
    tasks = (
        db.query(ProcessingTask)
        .options(
            selectinload(ProcessingTask.document_upload)
        )
        .filter(
            ProcessingTask.id.in_(task_id_list),
            ProcessingTask.knowledge_base_id == kb_id
        )
        .all()
    )
    
    return {
        task.id: {
            "document_id": task.document_id,
            "status": task.status,
            "error_message": task.error_message,
            "upload_id": task.document_upload_id,
            "file_name": task.document_upload.file_name if task.document_upload else None
        }
        for task in tasks
    }

@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    doc_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Delete a single document and all its associated data:
    - Physical file from local storage
    - All chunk vectors from Qdrant
    - All chunk records from MySQL
    - Processing task records from MySQL
    - The document record itself from MySQL
    """
    from app.services.document_processor import _chunk_id_to_point_id
    from qdrant_client.models import PointIdsList

    # Verify the KB belongs to this user
    kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    document = (
        db.query(Document)
        .filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id
        )
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    cleanup_warnings = []

    try:
        # 1. Collect chunk IDs before deleting them (needed for Qdrant point IDs)
        chunk_ids = [
            c.id for c in
            db.query(DocumentChunk.id)
            .filter(
                DocumentChunk.document_id == doc_id,
                DocumentChunk.kb_id == kb_id
            )
            .all()
        ]

        # 2. Delete vectors from Qdrant
        if chunk_ids:
            try:
                qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
                point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
                qdrant.delete(
                    collection_name=f"kb_{kb_id}",
                    points_selector=PointIdsList(points=point_ids),
                )
                logger.info(f"Deleted {len(point_ids)} Qdrant points for document {doc_id}")
            except Exception as e:
                cleanup_warnings.append(f"Qdrant cleanup warning: {str(e)}")
                logger.error(f"Failed to delete Qdrant points for document {doc_id}: {e}")

        # 3. Delete chunk rows from MySQL (explicit, don't rely on cascade here
        #    since we already fetched the IDs and want the delete to be transactional)
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc_id,
            DocumentChunk.kb_id == kb_id
        ).delete(synchronize_session=False)

        # 4. Delete processing task records for this document
        db.query(ProcessingTask).filter(
            ProcessingTask.document_id == doc_id
        ).delete(synchronize_session=False)

        # 5. Delete the physical file from local storage
        try:
            delete_file(document.file_path)
            logger.info(f"Deleted file from storage: {document.file_path}")
        except Exception as e:
            cleanup_warnings.append(f"File storage cleanup warning: {str(e)}")
            logger.error(f"Failed to delete file {document.file_path}: {e}")

        # 6. Delete the document record itself
        db.delete(document)
        db.commit()

        logger.info(f"Document {doc_id} deleted from KB {kb_id}")

        response = {"message": f"Document '{document.file_name}' deleted successfully"}
        if cleanup_warnings:
            response["warnings"] = cleanup_warnings
        return response

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    doc_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get document details by ID.
    """
    document = (
        db.query(Document)
        .join(KnowledgeBase)
        .filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == current_user.id
        )
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document

@router.post("/test-retrieval")
async def test_retrieval(
    request: TestRetrievalRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Test retrieval quality for a given query against a knowledge base.
    """
    try:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.id == request.kb_id,
            KnowledgeBase.user_id == current_user.id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base {request.kb_id} not found",
            )
        
        from app.services.retrieval import hybrid_search
        docs = await hybrid_search(
            query=request.query,
            kb_ids=[request.kb_id],
            db=db,
        )
        response = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": 0.0}
            for doc in docs[: request.top_k]
        ]
        return {"results": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
