"""
4-leg hybrid retrieval fused with Reciprocal Rank Fusion (RRF):

  Leg 1 — Dense   : Qdrant cosine-similarity search on Qwen3 embeddings
  Leg 2 — Sparse  : Qdrant learned sparse-vector search (SPLADE via FastEmbed)
  Leg 3 — Exact   : MySQL InnoDB FULLTEXT search (BM25/TF-IDF, server-side)
  Leg 4 — Graph   : Neo4j VectorCypherRetriever (opt-in per chat, requires use_graph_rag=True)

All three base legs are run independently; their rank lists are merged by weighted
RRF.  Individual legs can be disabled via .env (retrieval only — ingestion
always indexes all three, so re-enabling a leg needs no re-indexing).

Configuration (.env / settings):
  HYBRID_DENSE_WEIGHT          — RRF weight for the dense leg          (default 0.5)
  HYBRID_QDRANT_SPARSE_WEIGHT  — RRF weight for the Qdrant sparse leg  (default 0.3)
  HYBRID_EXACT_WEIGHT          — RRF weight for the MySQL exact leg     (default 0.2)
  HYBRID_GRAPH_WEIGHT          — RRF weight for the graph leg           (default 0.3)
  RETRIEVAL_TOP_K              — number of documents returned           (default 6)
  RETRIEVAL_DENSE_ENABLED      — enable/disable dense leg               (default true)
  RETRIEVAL_QDRANT_SPARSE_ENABLED — enable/disable sparse leg           (default true)
  RETRIEVAL_EXACT_ENABLED      — enable/disable exact leg               (default true)
  RETRIEVAL_GRAPH_ENABLED      — enable/disable graph leg               (default true)

Absent-leg design
-----------------
A document absent from a leg (no hit, score=0, or leg disabled) contributes
0 to its RRF score from that leg.  It can still surface via the other legs —
this is the correct behaviour: a paraphrase match with no exact keyword
overlap should be returned by the dense/sparse legs, not suppressed.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from langchain_core.documents import Document as LangchainDocument
from openai import OpenAI as SyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)

# RRF smoothing constant — standard value from the original paper (k=60).
_RRF_K = 60

# ── Module-level singletons (lazy-initialised) ────────────────────────────────
_qdrant_client: Optional[QdrantClient] = None
_openai_client: Optional[SyncOpenAI] = None
_sparse_embedder: Optional[SparseTextEmbedding] = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _qdrant_client


def _get_openai_client() -> SyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = SyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )
    return _openai_client


def _get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(
            model_name=settings.SPLADE_MODEL,
            cache_dir=settings.FASTEMBED_CACHE_DIR,
        )
    return _sparse_embedder


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class _Candidate:
    doc: LangchainDocument
    content_hash: str
    dense_rank: int = -1           # -1 = absent from this leg
    qdrant_sparse_rank: int = -1
    exact_rank: int = -1
    graph_rank: int = -1

    @property
    def rrf_score(self) -> float:
        score = 0.0
        if self.dense_rank >= 0:
            score += settings.HYBRID_DENSE_WEIGHT / (_RRF_K + self.dense_rank)
        if self.qdrant_sparse_rank >= 0:
            score += settings.HYBRID_QDRANT_SPARSE_WEIGHT / (_RRF_K + self.qdrant_sparse_rank)
        if self.exact_rank >= 0:
            score += settings.HYBRID_EXACT_WEIGHT / (_RRF_K + self.exact_rank)
        if self.graph_rank >= 0:
            score += settings.HYBRID_GRAPH_WEIGHT / (_RRF_K + self.graph_rank)
        return score


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _qdrant_payload_to_doc(payload: dict) -> LangchainDocument:
    chunk_text = payload.get("chunk_text", "")
    metadata = {k: v for k, v in payload.items() if k != "chunk_text"}
    return LangchainDocument(page_content=chunk_text, metadata=metadata)


# ── Search legs ───────────────────────────────────────────────────────────────

def _dense_search(query: str, kb_ids: List[int], candidates: int) -> Dict[str, _Candidate]:
    """Qdrant cosine-similarity search using the dense (OpenAI) embedding."""
    logger.info("[DENSE] embedding request | model=%s | query=%r", settings.OPENAI_EMBEDDINGS_MODEL, query[:120])
    response = _get_openai_client().embeddings.create(
        input=query,
        model=settings.OPENAI_EMBEDDINGS_MODEL,
    )
    query_vector = response.data[0].embedding
    logger.info("[DENSE] embedding response | dim=%d | first5=%s",
                len(query_vector), [round(v, 4) for v in query_vector[:5]])

    result: Dict[str, _Candidate] = {}
    rank = 0
    for kb_id in kb_ids:
        logger.info("[DENSE] qdrant query | collection=kb_%d | using=dense | limit=%d", kb_id, candidates)
        try:
            hits = _get_qdrant_client().query_points(
                collection_name=f"kb_{kb_id}",
                query=query_vector,
                using="dense",
                limit=candidates,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning("dense_search: Qdrant query failed for kb_%d: %s", kb_id, e)
            continue
        logger.info("[DENSE] qdrant response | kb_%d | hits=%d", kb_id, len(hits))
        for hit in hits:
            text = (hit.payload or {}).get("chunk_text", "")
            h = _content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    dense_rank=rank,
                )
                logger.debug("[DENSE]   rank=%d score=%.4f text=%r", rank, getattr(hit, 'score', -1), text[:80])
                rank += 1
    logger.info("[DENSE] unique candidates=%d", len(result))
    return result


def _qdrant_sparse_search(query: str, kb_ids: List[int], candidates: int) -> Dict[str, _Candidate]:
    """Qdrant learned-sparse search (SPLADE via FastEmbed)."""
    logger.info("[SPARSE] SPLADE embed | model=%s | query=%r", settings.SPLADE_MODEL, query[:120])
    sparse_emb = next(iter(_get_sparse_embedder().embed([query])))
    query_sparse = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )
    logger.info("[SPARSE] SPLADE response | nnz=%d | top_terms_indices=%s | top_values=%s",
                len(sparse_emb.indices),
                sparse_emb.indices[:5].tolist(),
                [round(v, 4) for v in sparse_emb.values[:5].tolist()])

    result: Dict[str, _Candidate] = {}
    rank = 0
    for kb_id in kb_ids:
        logger.info("[SPARSE] qdrant query | collection=kb_%d | using=sparse | limit=%d", kb_id, candidates)
        try:
            hits = _get_qdrant_client().query_points(
                collection_name=f"kb_{kb_id}",
                query=query_sparse,
                using="sparse",
                limit=candidates,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning("qdrant_sparse_search: Qdrant query failed for kb_%d: %s", kb_id, e)
            continue
        logger.info("[SPARSE] qdrant response | kb_%d | hits=%d", kb_id, len(hits))
        for hit in hits:
            text = (hit.payload or {}).get("chunk_text", "")
            h = _content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    qdrant_sparse_rank=rank,
                )
                logger.debug("[SPARSE]   rank=%d score=%.4f text=%r", rank, getattr(hit, 'score', -1), text[:80])
                rank += 1
    logger.info("[SPARSE] unique candidates=%d", len(result))
    return result


def _exact_search(query: str, kb_ids: List[int], db: Session, candidates: int) -> Dict[str, _Candidate]:
    """MySQL InnoDB FULLTEXT search — exact keyword / BM25 scoring, server-side."""
    if not query.strip():
        return {}

    sql = text(
        """
        SELECT chunk_text, chunk_metadata,
               MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS fts_score
        FROM   document_chunks
        WHERE  kb_id IN :kb_ids
          AND  MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
        ORDER  BY fts_score DESC
        LIMIT  :candidates
        """
    ).bindparams(bindparam("kb_ids", expanding=True))

    logger.info("[EXACT] MySQL FTS query | query=%r | kb_ids=%s | candidates=%d", query[:120], kb_ids, candidates)
    try:
        rows = db.execute(sql, {"query": query, "kb_ids": kb_ids, "candidates": candidates}).fetchall()
    except Exception as e:
        logger.warning("exact_search: MySQL FTS query failed: %s", e)
        return {}

    logger.info("[EXACT] MySQL FTS response | rows=%d", len(rows))
    if rows:
        for i, row in enumerate(rows[:5]):
            logger.info("  exact[%d] fts_score=%.4f text=%r", i, row.fts_score, (row.chunk_text or "")[:80])

    result: Dict[str, _Candidate] = {}
    for rank, row in enumerate(rows):
        chunk_text = row.chunk_text or ""
        h = _content_hash(chunk_text)
        if h not in result:
            raw_meta = row.chunk_metadata
            if isinstance(raw_meta, str):
                try:
                    meta = json.loads(raw_meta)
                except (ValueError, TypeError):
                    meta = {}
            elif isinstance(raw_meta, dict):
                meta = raw_meta
            else:
                meta = {}
            result[h] = _Candidate(
                doc=LangchainDocument(
                    page_content=chunk_text,
                    metadata=meta,
                ),
                content_hash=h,
                exact_rank=rank,
            )
    return result


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge_candidates(
    dense: Dict[str, "_Candidate"],
    qdrant_sparse: Dict[str, "_Candidate"],
    exact: Dict[str, "_Candidate"],
    graph: Dict[str, "_Candidate"],
    top_k: int,
) -> list["_Candidate"]:
    merged: Dict[str, _Candidate] = {**dense}

    for h, c in qdrant_sparse.items():
        if h in merged:
            merged[h].qdrant_sparse_rank = c.qdrant_sparse_rank
        else:
            merged[h] = c

    for h, c in exact.items():
        if h in merged:
            merged[h].exact_rank = c.exact_rank
        else:
            merged[h] = c

    for h, c in graph.items():
        if h in merged:
            merged[h].graph_rank = c.graph_rank
        else:
            merged[h] = c

    ranked = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)

    logger.info("[RRF] total unique candidates=%d | returning top_k=%d", len(ranked), top_k)
    for i, c in enumerate(ranked[:top_k]):
        logger.info(
            "  rrf[%d] score=%.5f dense_rank=%s sparse_rank=%s exact_rank=%s graph_rank=%s text=%r",
            i, c.rrf_score,
            c.dense_rank if c.dense_rank >= 0 else "-",
            c.qdrant_sparse_rank if c.qdrant_sparse_rank >= 0 else "-",
            c.exact_rank if c.exact_rank >= 0 else "-",
            c.graph_rank if c.graph_rank >= 0 else "-",
            c.doc.page_content[:80],
        )
    return ranked[:top_k]


# ── Public API ────────────────────────────────────────────────────────────────

def _run_leg(name: str, fn, *args) -> tuple[dict, str | None]:
    """Run a single retrieval leg, catching any exception.
    Returns (results_dict, error_message_or_None)."""
    try:
        return fn(*args), None
    except Exception as exc:
        logger.error("[LEG:%s] failed: %s", name, exc)
        return {}, str(exc)


async def hybrid_search_with_legs(
    query: str,
    kb_ids: List[int],
    db: Session,
    use_dense:     bool = True,
    use_sparse:    bool = True,
    use_exact:     bool = True,
    use_graph_rag: bool = False,
) -> dict:
    """
    Like hybrid_search but returns a richer dict:
      {
        "docs": [...],
        "retrieval_info": {
          "legs": {
            "dense":         {"status": "ok"|"failed"|"disabled", "count": N, "error": str|None},
            "qdrant_sparse": {...},
            "exact":         {...},
            "graph":         {...},
          },
          "failed_legs": ["dense", ...],
        }
      }
    Each leg runs independently — a failure in one never blocks the others.

    Per-call flags AND with global .env settings: a leg is only enabled when
    both the chat-level flag and the global env flag are True.
    """
    top_k = settings.RETRIEVAL_TOP_K
    pool  = top_k * 4

    enabled = {
        "dense":         use_dense    and settings.RETRIEVAL_DENSE_ENABLED,
        "qdrant_sparse": use_sparse   and settings.RETRIEVAL_QDRANT_SPARSE_ENABLED,
        "exact":         use_exact    and settings.RETRIEVAL_EXACT_ENABLED,
        "graph":         use_graph_rag and settings.RETRIEVAL_GRAPH_ENABLED,
    }

    legs: dict[str, dict] = {}

    if enabled["dense"]:
        results, err = _run_leg("dense", _dense_search, query, kb_ids, pool)
        legs["dense"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        dense = results
    else:
        dense = {}
        legs["dense"] = {"status": "disabled", "count": 0, "error": None}

    if enabled["qdrant_sparse"]:
        results, err = _run_leg("qdrant_sparse", _qdrant_sparse_search, query, kb_ids, pool)
        legs["qdrant_sparse"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        qdrant_sparse = results
    else:
        qdrant_sparse = {}
        legs["qdrant_sparse"] = {"status": "disabled", "count": 0, "error": None}

    if enabled["exact"]:
        results, err = _run_leg("exact", _exact_search, query, kb_ids, db, pool)
        legs["exact"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        exact = results
    else:
        exact = {}
        legs["exact"] = {"status": "disabled", "count": 0, "error": None}

    graph: Dict[str, _Candidate] = {}
    if enabled["graph"]:
        try:
            from app.services.graph_service import graph_search as _graph_search
            graph_docs = _graph_search(query_text=query, kb_ids=kb_ids, top_k=pool)
            for rank, doc in enumerate(graph_docs):
                h = _content_hash(doc.page_content)
                if h not in graph:
                    graph[h] = _Candidate(doc=doc, content_hash=h, graph_rank=rank)
            legs["graph"] = {"status": "ok", "count": len(graph), "error": None}
        except Exception as exc:
            logger.error("[LEG:graph] failed: %s", exc)
            legs["graph"] = {"status": "failed", "count": 0, "error": str(exc)}
    else:
        legs["graph"] = {"status": "disabled", "count": 0, "error": None}

    failed_legs = [k for k, v in legs.items() if v["status"] == "failed"]
    if failed_legs:
        logger.warning("hybrid_search_with_legs: failed legs=%s", failed_legs)

    candidates = _rrf_merge_candidates(dense, qdrant_sparse, exact, graph, top_k)

    # Annotate each doc with which legs found it — used by confidence scoring
    # (cross-leg agreement signal). Stored in metadata["_legs"] as a list of
    # leg names so the scoring module can inspect it without touching internals.
    docs: List[LangchainDocument] = []
    for c in candidates:
        contributing = []
        if c.dense_rank >= 0:         contributing.append("dense")
        if c.qdrant_sparse_rank >= 0: contributing.append("qdrant_sparse")
        if c.exact_rank >= 0:         contributing.append("exact")
        if c.graph_rank >= 0:         contributing.append("graph")
        c.doc.metadata["_legs"] = contributing
        docs.append(c.doc)

    logger.info("hybrid_search_with_legs returned %d docs | failed_legs=%s", len(docs), failed_legs)

    # ── Cross-encoder reranking (optional) ────────────────────────────────────
    # Applied after RRF merge so the reranker only sees the top-K candidates,
    # not the full pool. Non-fatal — a reranker failure falls back to RRF order.
    if settings.RERANKER_ENABLED and docs:
        try:
            from app.services.reranker import rerank
            docs = rerank(query=query, docs=docs)
            logger.info("hybrid_search_with_legs: reranker reduced to %d docs", len(docs))
        except Exception as exc:
            logger.warning("hybrid_search_with_legs: reranker failed (using RRF order): %s", exc)

    return {
        "docs": docs,
        "retrieval_info": {
            "legs": legs,
            "failed_legs": failed_legs,
        },
    }


async def hybrid_search(
    query: str,
    kb_ids: List[int],
    db: Session,
    use_graph_rag: bool = False,
) -> List[LangchainDocument]:
    """Run enabled retrieval legs in parallel (sync calls) and merge via RRF."""
    top_k = settings.RETRIEVAL_TOP_K
    pool = top_k * 4  # over-fetch so RRF has room to rerank

    enabled = {
        "dense": settings.RETRIEVAL_DENSE_ENABLED,
        "qdrant_sparse": settings.RETRIEVAL_QDRANT_SPARSE_ENABLED,
        "exact": settings.RETRIEVAL_EXACT_ENABLED,
        "graph": use_graph_rag and settings.RETRIEVAL_GRAPH_ENABLED,
    }
    logger.info(
        "hybrid_search | kb_ids=%s top_k=%d legs=%s weights=(%.2f, %.2f, %.2f, %.2f)",
        kb_ids, top_k,
        [k for k, v in enabled.items() if v],
        settings.HYBRID_DENSE_WEIGHT,
        settings.HYBRID_QDRANT_SPARSE_WEIGHT,
        settings.HYBRID_EXACT_WEIGHT,
        settings.HYBRID_GRAPH_WEIGHT,
    )

    dense        = _dense_search(query, kb_ids, pool)           if enabled["dense"]          else {}
    qdrant_sparse = _qdrant_sparse_search(query, kb_ids, pool)  if enabled["qdrant_sparse"]  else {}
    exact        = _exact_search(query, kb_ids, db, pool)       if enabled["exact"]          else {}

    graph: Dict[str, _Candidate] = {}
    if enabled["graph"]:
        try:
            from app.services.graph_service import graph_search as _graph_search
            graph_docs = _graph_search(query_text=query, kb_ids=kb_ids, top_k=pool)
            for rank, doc in enumerate(graph_docs):
                h = _content_hash(doc.page_content)
                if h not in graph:
                    graph[h] = _Candidate(
                        doc=doc,
                        content_hash=h,
                        graph_rank=rank,
                    )
            logger.info("[GRAPH] graph_search returned %d unique candidates", len(graph))
        except Exception as e:
            logger.warning("hybrid_search: graph leg failed (non-fatal): %s", e)

    docs = _rrf_merge(dense, qdrant_sparse, exact, graph, top_k)
    logger.info("hybrid_search returned %d documents", len(docs))
    return docs
