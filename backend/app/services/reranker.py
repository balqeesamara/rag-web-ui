"""
Cross-encoder reranker service.

Uses sentence-transformers CrossEncoder to re-score (query, chunk) pairs
after RRF merging. More accurate than RRF alone because it jointly encodes
the query and each chunk together instead of scoring them independently.

Model: cross-encoder/ms-marco-MiniLM-L-12-v2
  - Trained on MS MARCO passage ranking (127M query-passage pairs)
  - 12-layer MiniLM — fast enough for CPU, ~6ms per chunk on modern hardware
  - Outputs a raw logit (higher = more relevant); no fixed 0–1 scale

Integration point: called in hybrid_search_with_legs() after RRF merge,
before the docs are passed to the LLM.
"""

import logging
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once on first use, reused across all requests.
# CrossEncoder is stateless between calls so it is safe to share.
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        model_name = settings.RERANKER_MODEL
        cache_dir = settings.RERANKER_CACHE_DIR

        logger.info(
            "Reranker: loading cross-encoder model=%s cache=%s",
            model_name,
            cache_dir,
        )
        os.makedirs(cache_dir, exist_ok=True)
        _cross_encoder = CrossEncoder(
            model_name,
            cache_folder=cache_dir,
            # No max_length override — MiniLM handles up to 512 tokens natively.
            # Longer chunks are truncated by the tokeniser, which is fine for
            # passage-level relevance scoring.
        )
        logger.info("Reranker: model loaded")
    return _cross_encoder


def rerank(
    query: str,
    docs: List[LangchainDocument],
    top_n: Optional[int] = None,
) -> List[LangchainDocument]:
    """
    Re-score docs against query using the cross-encoder and return the
    top_n most relevant ones in descending order.

    Args:
        query:  The retrieval query (standalone question after rewriting).
        docs:   Candidates from RRF merge — already filtered by min_rrf_score.
        top_n:  How many to keep. Defaults to settings.RERANKER_TOP_N.

    Returns:
        List of LangchainDocuments, re-ordered by cross-encoder score,
        truncated to top_n. Each doc gets metadata["_reranker_score"] set.
    """
    if not docs:
        return docs

    if top_n is None:
        top_n = settings.RERANKER_TOP_N

    encoder = _get_cross_encoder()

    pairs: List[Tuple[str, str]] = [(query, doc.page_content) for doc in docs]
    scores: List[float] = encoder.predict(pairs).tolist()

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    logger.info(
        "Reranker: query=%r | input=%d chunks | keeping top_n=%d | "
        "score range=[%.3f, %.3f]",
        query[:80],
        len(docs),
        min(top_n, len(scored)),
        scored[-1][0] if scored else 0.0,
        scored[0][0] if scored else 0.0,
    )

    result = []
    for score, doc in scored[:top_n]:
        doc.metadata["_reranker_score"] = round(score, 4)
        result.append(doc)

    return result
