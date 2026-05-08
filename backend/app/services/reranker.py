"""
Cross-encoder reranker service.

Uses sentence-transformers CrossEncoder to re-score (query, chunk) pairs
after RRF merging. More accurate than RRF alone because it jointly encodes
the query and each chunk together instead of scoring them independently.

Model: cross-encoder/ms-marco-MiniLM-L-12-v2
  - Trained on MS MARCO passage ranking (127M query-passage pairs)
  - 12-layer MiniLM — fast enough for CPU, ~6ms per chunk on modern hardware
  - Outputs a raw logit (higher = more relevant); no fixed 0–1 scale

Score distribution (empirical, ms-marco-MiniLM-L-12-v2, Identity activation):
  Scores are bimodal — relevant chunks cluster 1–10, irrelevant cluster -5 to -11.
  There is almost no middle ground. 0.0 is a reliable cutoff for this model.

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

        os.makedirs(cache_dir, exist_ok=True)

        kwargs = dict(
            cache_folder=cache_dir,
            model_kwargs={},
            processor_kwargs={},
        )

        # Try loading from local cache first to avoid HF Hub version-check
        # roundtrips on every container start. Falls back to downloading if
        # the model isn't cached yet.
        try:
            _cross_encoder = CrossEncoder(model_name, local_files_only=True, **kwargs)
            logger.info("Reranker: model loaded from local cache")
        except Exception:
            logger.info("Reranker: model not in cache, downloading model=%s", model_name)
            _cross_encoder = CrossEncoder(model_name, **kwargs)
            logger.info("Reranker: model downloaded and cached")

    return _cross_encoder


def rerank(
    query: str,
    docs: List[LangchainDocument],
    score_threshold: Optional[float] = None,
) -> List[LangchainDocument]:
    """
    Re-score docs against query using the cross-encoder and filter by threshold.

    All chunks scoring above the threshold are returned, ordered by score.
    No top_n cap — if 8 out of 10 chunks are relevant, all 8 pass.

    Args:
        query:           The retrieval query.
        docs:            Candidates from RRF merge.
        score_threshold: Min logit to pass. Defaults to RERANKER_SCORE_THRESHOLD.

    Returns:
        Docs re-ordered by cross-encoder score, filtered by threshold only.
        Each doc gets metadata["_reranker_score"] set.
    """
    if not docs:
        return docs

    if score_threshold is None:
        score_threshold = settings.RERANKER_SCORE_THRESHOLD

    encoder = _get_cross_encoder()

    pairs: List[Tuple[str, str]] = [(query, doc.page_content) for doc in docs]
    scores: List[float] = encoder.predict(pairs).tolist()

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    logger.info(
        "Reranker: query=%r | input=%d | threshold=%.2f | score range=[%.3f, %.3f]",
        query[:80],
        len(docs),
        score_threshold,
        scored[-1][0] if scored else 0.0,
        scored[0][0] if scored else 0.0,
    )

    for rank, (score, doc) in enumerate(scored):
        snippet = doc.page_content[:80].replace("\n", " ")
        logger.info("  reranker[%d] score=%.4f text=%r", rank, score, snippet)

    result = []
    for score, doc in scored:
        if score < score_threshold:
            break  # sorted descending — nothing below this will pass
        doc.metadata["_reranker_score"] = round(score, 4)
        result.append(doc)

    logger.info(
        "Reranker: %d/%d chunks passed threshold=%.2f",
        len(result), len(scored), score_threshold,
    )
    return result
