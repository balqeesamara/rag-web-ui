"""
POST /api/query          — stateless RAG query, returns JSON (no SSE, no chat session)
GET  /api/query/kb/{id}/ingest-status — KB processing readiness check

These two endpoints are the only additions needed for an external eval harness.
The harness never needs to touch the chat/session machinery.
"""
import time
import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.models.knowledge import KnowledgeBase, ProcessingTask
from app.services.retrieval import hybrid_search_with_legs
from app.services.confidence import score_retrieval

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    kb_ids: List[int]
    # Per-request leg flags — AND-ed with global .env settings.
    # Default all True so the endpoint behaves like the full hybrid pipeline
    # unless the caller explicitly disables a leg for benchmarking.
    use_dense:     bool = True
    use_sparse:    bool = True
    use_exact:     bool = True
    use_graph_rag: bool = False
    # Pass False to skip LLM answer generation — retrieval-only benchmark runs
    # are much faster and don't consume LLM tokens.
    generate_answer: bool = True


class ContextChunk(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    contexts: List[ContextChunk]
    confidence: str                  # "high" | "low" | "none"
    suggestion: Optional[str]
    retrieval_info: dict
    latency_ms: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", response_model=QueryResponse)
async def query(
    *,
    db: Session = Depends(get_db),
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Stateless RAG query — no chat session created, nothing persisted.

    The response contains:
    - answer           LLM answer (or null if generate_answer=False)
    - contexts         retrieved chunks with metadata
    - confidence       high / low / none
    - suggestion       human-readable hint when confidence != high
    - retrieval_info   per-leg status (ok / failed / disabled + count)
    - latency_ms       wall-clock time for the full call
    """
    t0 = time.monotonic()

    # Verify all KBs exist and belong to this user
    kbs = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id.in_(body.kb_ids),
            KnowledgeBase.user_id == current_user.id,
        )
        .all()
    )
    if len(kbs) != len(body.kb_ids):
        raise HTTPException(status_code=404, detail="One or more knowledge bases not found")

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieval_result = await hybrid_search_with_legs(
        query=body.question,
        kb_ids=body.kb_ids,
        db=db,
        use_dense=body.use_dense,
        use_sparse=body.use_sparse,
        use_exact=body.use_exact,
        use_graph_rag=body.use_graph_rag,
    )
    docs            = retrieval_result["docs"]
    retrieval_info  = retrieval_result["retrieval_info"]
    failed_legs     = retrieval_info["failed_legs"]

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence_result = score_retrieval(docs, retrieval_info)

    answer: Optional[str] = None
    if body.generate_answer and docs:
        context_text = "\n\n".join(
            f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
        )
        system_prompt = (
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY the provided context. "
            "Cite sources as [1], [2], etc. "
            "If the context does not contain enough information, say so briefly."
        )
        openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )
        response = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context_text}"},
                {"role": "user", "content": body.question},
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "[QUERY] question=%r | kb_ids=%s | docs=%d | confidence=%s | latency=%dms",
        body.question[:80], body.kb_ids, len(docs), confidence_result.level, latency_ms,
    )

    return QueryResponse(
        question=body.question,
        answer=answer,
        contexts=[ContextChunk(content=d.page_content, metadata=d.metadata) for d in docs],
        confidence=confidence_result.level,
        suggestion=confidence_result.suggestion,
        retrieval_info={**retrieval_info, "confidence_breakdown": confidence_result.breakdown},
        latency_ms=latency_ms,
    )


# ── KB ingest status ───────────────────────────────────────────────────────────

class IngestStatus(BaseModel):
    kb_id: int
    total: int
    completed: int
    failed: int
    pending: int
    ready: bool       # True when total > 0 and completed == total and failed == 0


@router.get("/kb/{kb_id}/ingest-status", response_model=IngestStatus)
def ingest_status(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Returns processing status for every document task in a knowledge base.
    Poll this until ready=True before running eval queries.
    """
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == current_user.id)
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    tasks = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.knowledge_base_id == kb_id)
        .all()
    )

    total     = len(tasks)
    completed = sum(1 for t in tasks if t.status == "completed")
    failed    = sum(1 for t in tasks if t.status == "failed")
    pending   = total - completed - failed

    return IngestStatus(
        kb_id=kb_id,
        total=total,
        completed=completed,
        failed=failed,
        pending=pending,
        ready=(total > 0 and completed == total and failed == 0),
    )
