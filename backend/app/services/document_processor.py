import logging
import os
import uuid
import hashlib
import traceback
from app.db.session import SessionLocal
from typing import Optional, List, Dict, Set, Tuple
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from fastembed import SparseTextEmbedding
from app.core.config import settings
from app.core.storage import get_abs_path, save_file, move_file, delete_file
from app.models.knowledge import ProcessingTask, Document, DocumentChunk
from app.services.chunk_record import ChunkRecord

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_qdrant_client: Optional[QdrantClient] = None
_sparse_embedder: Optional[SparseTextEmbedding] = None
_EMBED_BATCH_SIZE = 32
_QDRANT_UPSERT_BATCH = 100


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _qdrant_client


def _get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(
            model_name=settings.SPLADE_MODEL,
            cache_dir=settings.FASTEMBED_CACHE_DIR,
        )
    return _sparse_embedder


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert a SHA-256 hex chunk ID to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def _ensure_qdrant_collection(client: QdrantClient, kb_id: int) -> None:
    """Create the Qdrant collection for a knowledge base if it does not exist."""
    collection_name = f"kb_{kb_id}"
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=settings.DENSE_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )


async def _embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Compute dense embeddings via the OpenAI-compatible API, in batches."""
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
    )
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        response = await client.embeddings.create(
            input=batch,
            model=settings.OPENAI_EMBEDDINGS_MODEL,
        )
        all_embeddings.extend(r.embedding for r in response.data)
    return all_embeddings


def _build_qdrant_points(
    chunk_payloads: List[Tuple[str, str, dict, int]],  # (chunk_id, text, metadata, index)
    dense_embeddings: List[List[float]],
    sparse_embeddings,
    kb_id: int,
    document_id: int,
    file_name: str,
) -> List[PointStruct]:
    """Build Qdrant PointStruct list from pre-computed embeddings."""
    points = []
    for (chunk_id, chunk_text, source_meta, chunk_index), dense_emb, sparse_emb in zip(
        chunk_payloads, dense_embeddings, sparse_embeddings
    ):
        points.append(
            PointStruct(
                id=_chunk_id_to_point_id(chunk_id),
                vector={
                    "dense": dense_emb,
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={
                    "chunk_text": chunk_text,
                    "kb_id": kb_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    **source_meta,
                },
            )
        )
    return points


async def _upsert_to_qdrant(
    chunk_payloads: List[Tuple[str, str, dict, int]],
    kb_id: int,
    document_id: int,
    file_name: str,
) -> None:
    """Compute both vector types and upsert all points to Qdrant."""
    if not chunk_payloads:
        return
    texts = [p[1] for p in chunk_payloads]
    dense_embs = await _embed_texts_batch(texts)
    sparse_embs = list(_get_sparse_embedder().embed(texts))

    client = _get_qdrant_client()
    _ensure_qdrant_collection(client, kb_id)
    points = _build_qdrant_points(
        chunk_payloads, dense_embs, sparse_embs, kb_id, document_id, file_name
    )
    for i in range(0, len(points), _QDRANT_UPSERT_BATCH):
        client.upsert(
            collection_name=f"kb_{kb_id}",
            points=points[i : i + _QDRANT_UPSERT_BATCH],
        )

class UploadResult(BaseModel):
    file_path: str
    file_name: str
    file_size: int
    content_type: str
    file_hash: str

class TextChunk(BaseModel):
    content: str
    metadata: Optional[Dict] = None

class PreviewResult(BaseModel):
    chunks: List[TextChunk]
    total_chunks: int

async def process_document(file_path: str, file_name: str, kb_id: int, document_id: int, chunk_size: int = None, chunk_overlap: int = None) -> None:
    """Process document and store in vector database with incremental updates"""
    logger = logging.getLogger(__name__)
    # Use env-configured defaults when callers do not supply explicit values.
    # WARNING: chunk_size and chunk_overlap must stay consistent across all
    # documents in a knowledge base. Do not change CHUNK_SIZE / OVERLAP_PERCENTAGE
    # in .env after documents have been ingested — re-upload existing documents
    # to re-index them with the new settings.
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap
    
    try:
        preview_result = await preview_document(file_path, chunk_size, chunk_overlap)
        
        # Initialize chunk record manager
        chunk_manager = ChunkRecord(kb_id)
        
        # Get existing chunk hashes for this file
        existing_hashes = chunk_manager.list_chunks(file_name)
        
        # Prepare new chunks
        new_chunks = []
        current_hashes = set()
        
        for chunk_index, chunk in enumerate(preview_result.chunks):
            # Calculate chunk hash
            chunk_hash = hashlib.sha256(
                (chunk.content + str(chunk.metadata)).encode()
            ).hexdigest()
            current_hashes.add(chunk_hash)
            
            # Skip if chunk hasn't changed
            if chunk_hash in existing_hashes:
                continue
            
            # Create unique ID for the chunk
            chunk_id = hashlib.sha256(
                f"{kb_id}:{file_name}:{chunk_hash}".encode()
            ).hexdigest()
            
            # chunk_metadata holds only variable source metadata (page number, source path)
            # chunk_text and chunk_index are stored as proper columns
            metadata = {k: v for k, v in chunk.metadata.items()
                        if k not in ("kb_id", "document_id", "chunk_id", "file_name")}
            
            new_chunks.append({
                "id": chunk_id,
                "kb_id": kb_id,
                "document_id": document_id,
                "file_name": file_name,
                "chunk_text": chunk.content,
                "chunk_index": chunk_index,
                "metadata": metadata,
                "hash": chunk_hash
            })
        
        # Add new chunks to MySQL + Qdrant
        if new_chunks:
            logger.info(f"Adding {len(new_chunks)} new/updated chunks")
            chunk_manager.add_chunks(new_chunks)
            chunk_payloads = [
                (c["id"], c["chunk_text"], c.get("metadata") or {}, c["chunk_index"])
                for c in new_chunks
            ]
            await _upsert_to_qdrant(chunk_payloads, kb_id, document_id, file_name)
        
        # Delete removed chunks from MySQL + Qdrant
        chunks_to_delete = chunk_manager.get_deleted_chunks(current_hashes, file_name)
        if chunks_to_delete:
            logger.info(f"Removing {len(chunks_to_delete)} deleted chunks")
            chunk_manager.delete_chunks(chunks_to_delete)
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunks_to_delete]
            _get_qdrant_client().delete(
                collection_name=f"kb_{kb_id}",
                points_selector=PointIdsList(points=point_ids),
            )
        
        logger.info("Document processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

async def upload_document(file: UploadFile, kb_id: int, user_id: int) -> UploadResult:
    """Step 1: Upload document to local storage"""
    content = await file.read()
    file_size = len(content)

    file_hash = hashlib.sha256(content).hexdigest()

    # Clean and normalize filename
    file_name = "".join(c for c in file.filename if c.isalnum() or c in ('-', '_', '.')).strip()
    object_path = f"user_{user_id}/kb_{kb_id}/{file_name}"

    content_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".txt": "text/plain"
    }

    _, ext = os.path.splitext(file_name)
    content_type = content_types.get(ext.lower(), "application/octet-stream")

    try:
        save_file(object_path, content)
    except Exception as e:
        logging.error(f"Failed to save file to storage: {str(e)}")
        raise

    return UploadResult(
        file_path=object_path,
        file_name=file_name,
        file_size=file_size,
        content_type=content_type,
        file_hash=file_hash
    )

async def preview_document(file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> PreviewResult:
    """Step 2: Generate preview chunks"""
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    abs_path = get_abs_path(file_path)

    try:
        # Select appropriate loader
        if ext == ".pdf":
            loader = PyPDFLoader(abs_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(abs_path)
        elif ext == ".md":
            loader = TextLoader(abs_path)
        else:  # Default to text loader
            loader = TextLoader(abs_path)

        # Load and split the document
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)

        # Convert to preview format
        preview_chunks = [
            TextChunk(
                content=chunk.page_content,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]

        return PreviewResult(
            chunks=preview_chunks,
            total_chunks=len(chunks)
        )
    except Exception as e:
        logging.error(f"Failed to preview document {file_path}: {str(e)}")
        raise

async def process_document_background(
    temp_path: str,
    file_name: str,
    kb_id: int,
    task_id: int,
    db: Session = None,
    user_id: int = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> None:
    """Process document in background"""
    logger = logging.getLogger(__name__)
    # Use env-configured defaults when callers do not supply explicit values.
    # WARNING: chunk_size and chunk_overlap must stay consistent across all
    # documents in a knowledge base. Do not change CHUNK_SIZE / OVERLAP_PERCENTAGE
    # in .env after documents have been ingested — re-upload existing documents
    # to re-index them with the new settings.
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap
    logger.info(f"Starting background processing for task {task_id}, file: {file_name}")

    # if we don't pass in db, create a new database session
    if db is None:
        db = SessionLocal()
        should_close_db = True
    else:
        should_close_db = False
    
    task = db.query(ProcessingTask).get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return
    
    try:
        logger.info(f"Task {task_id}: Setting status to processing")
        task.status = "processing"
        db.commit()

        # 1. Resolve file path — file already lives on disk under UPLOAD_DIR
        local_temp_path = get_abs_path(temp_path)
        logger.info(f"Task {task_id}: Using file at {local_temp_path}")

        try:
            # 2. Load and split the document
            _, ext = os.path.splitext(file_name)
            ext = ext.lower()

            logger.info(f"Task {task_id}: Loading document with extension {ext}")
            if ext == ".pdf":
                loader = PyPDFLoader(local_temp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(local_temp_path)
            elif ext == ".md":
                loader = TextLoader(local_temp_path)
            else:
                loader = TextLoader(local_temp_path)

            logger.info(f"Task {task_id}: Loading document content")
            documents = loader.load()
            logger.info(f"Task {task_id}: Document loaded successfully")

            logger.info(f"Task {task_id}: Splitting document into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Task {task_id}: Document split into {len(chunks)} chunks")

            # 3. Initialise Qdrant collection (creates if not exists)
            logger.info(f"Task {task_id}: Ensuring Qdrant collection kb_{kb_id}")
            _ensure_qdrant_collection(_get_qdrant_client(), kb_id)

            # 4. Move temp file to permanent location
            permanent_path = f"user_{user_id}/kb_{kb_id}/{file_name}"
            try:
                logger.info(f"Task {task_id}: Moving file to permanent storage")
                move_file(temp_path, permanent_path)
                logger.info(f"Task {task_id}: File moved to permanent storage")
                # Update local path after move
                local_temp_path = get_abs_path(permanent_path)
            except Exception as e:
                error_msg = f"Failed to move file to permanent storage: {str(e)}"
                logger.error(f"Task {task_id}: {error_msg}")
                raise Exception(error_msg)

            # 5. Create document record
            logger.info(f"Task {task_id}: Creating document record")
            document = Document(
                file_name=file_name,
                file_path=permanent_path,
                file_hash=task.document_upload.file_hash,
                file_size=task.document_upload.file_size,
                content_type=task.document_upload.content_type,
                knowledge_base_id=kb_id
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            logger.info(f"Task {task_id}: Document record created with ID {document.id}")

            # 6. Store document chunks in MySQL + collect Qdrant payload
            logger.info(f"Task {task_id}: Storing document chunks in MySQL")
            qdrant_payloads: List[Tuple[str, str, dict, int]] = []
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha256(
                    f"{kb_id}:{file_name}:{chunk.page_content}".encode()
                ).hexdigest()

                chunk.metadata["source"] = file_name

                # chunk_text and chunk_index are proper columns; store only
                # variable source metadata (page number, source path) in JSON
                source_metadata = {k: v for k, v in chunk.metadata.items()
                                   if k not in ("kb_id", "document_id", "chunk_id", "file_name")}

                doc_chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    kb_id=kb_id,
                    file_name=file_name,
                    chunk_text=chunk.page_content,
                    chunk_index=i,
                    chunk_metadata=source_metadata,
                    hash=hashlib.sha256(
                        (chunk.page_content + str(chunk.metadata)).encode()
                    ).hexdigest()
                )
                db.add(doc_chunk)
                qdrant_payloads.append((chunk_id, chunk.page_content, source_metadata, i))
                if i > 0 and i % 100 == 0:
                    logger.info(f"Task {task_id}: Stored {i} chunks")
                    db.commit()

            # 7. Upsert dense + sparse vectors to Qdrant (all ingestion paths always run)
            logger.info(f"Task {task_id}: Upserting {len(qdrant_payloads)} chunks to Qdrant")
            await _upsert_to_qdrant(qdrant_payloads, kb_id, document.id, file_name)
            logger.info(f"Task {task_id}: Chunks added to Qdrant")

            # 8. Update task status
            logger.info(f"Task {task_id}: Updating task status to completed")
            task.status = "completed"
            task.document_id = document.id

            # 9. Update upload record status
            upload = task.document_upload
            if upload:
                logger.info(f"Task {task_id}: Updating upload record status to completed")
                upload.status = "completed"

            db.commit()
            logger.info(f"Task {task_id}: Processing completed successfully")

        except Exception:
            raise

    except Exception as e:
        logger.error(f"Task {task_id}: Error processing document: {str(e)}")
        logger.error(f"Task {task_id}: Stack trace: {traceback.format_exc()}")
        task.status = "failed"
        task.error_message = str(e)
        db.commit()

        # Clean up temp file on failure
        try:
            logger.info(f"Task {task_id}: Cleaning up temporary file after error")
            delete_file(temp_path)
            logger.info(f"Task {task_id}: Temporary file cleaned up after error")
        except Exception:
            logger.warning(f"Task {task_id}: Failed to clean up temporary file after error")
    finally:
        if should_close_db and db:
            db.close()
