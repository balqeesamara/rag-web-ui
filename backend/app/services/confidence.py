"""
Retrieval confidence scoring.

Two modes depending on whether the reranker is enabled:

── Reranker ON (default) ──────────────────────────────────────────────────────
All docs passed to this function have already cleared the reranker threshold,
so every doc is considered genuinely relevant. Signals:

  A. Top score      (50 pts) — best reranker logit, normalised over [0, 8]
                               score ≥ 8  → 50 pts (ceiling)
                               score = 0  → 0 pts
                               Linear in between: pts = clamp(score/8, 0, 1) * 50
  B. Doc count      (30 pts) — stepped: 1 doc → 10, 2 → 20, 3+ → 30
  C. Mean score     (20 pts) — mean logit of all passing docs, same normalisation
                               as A. Penalises cases where one good chunk is
                               surrounded by many marginal ones.

── Reranker OFF ───────────────────────────────────────────────────────────────
Falls back to the original four-signal model based on retrieval leg metadata:

  A. Source coverage    (30 pts) — fraction of enabled legs that returned results
  B. Cross-leg agreement (35 pts) — fraction of top-k chunks confirmed by ≥2 legs
  C. Volume fill rate   (25 pts) — docs returned / RETRIEVAL_TOP_K
  D. Source diversity   (10 pts) — unique source files, capped at 3

Score → level (both modes)
  ≥ 80  very_high
  ≥ 55  high
  ≥ 30  medium
  >  0  low
     0  none

The same function is used by chat_service.py (streaming chat) and query.py
(stateless eval endpoint) so the logic is never duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings


# ── Level ──────────────────────────────────────────────────────────────────────

LEVELS = ("none", "low", "medium", "high", "very_high")

# Reranker logit normalisation range.
# ms-marco-MiniLM-L-12-v2 observed range:
#   Specific Q&A queries:  relevant chunks  ~1 to 10
#   Broad/meta queries:    relevant chunks  ~-5 to -1, irrelevant ~-6 to -11
# Normalise over [-5, 8] so both query types produce meaningful confidence scores.
_RERANKER_SCORE_MIN = -5.0
_RERANKER_SCORE_MAX = 8.0


@dataclass
class ConfidenceResult:
    level: str                  # one of LEVELS
    score: int                  # 0-100
    suggestion: Optional[str]
    breakdown: dict             # per-signal scores for transparency


def _level_and_suggestion(score: int, failed_legs: list) -> tuple[str, Optional[str]]:
    if score == 0:
        level = "none"
    elif score < 30:
        level = "low"
    elif score < 55:
        level = "medium"
    elif score < 80:
        level = "high"
    else:
        level = "very_high"

    suggestion: Optional[str] = None
    if failed_legs:
        suggestion = (
            f"Some knowledge sources were unavailable "
            f"({', '.join(failed_legs)}). Results may be incomplete."
        )
    elif level == "none":
        suggestion = (
            "No relevant documents found. "
            "Try rephrasing your question or using different keywords."
        )
    elif level == "low":
        suggestion = (
            "Few relevant documents found. "
            "Try more specific keywords or check that the relevant documents have been ingested."
        )
    elif level == "medium":
        suggestion = (
            "Some relevant documents found. "
            "Results may be partial — consider rephrasing for better coverage."
        )
    # high / very_high → no suggestion needed

    return level, suggestion


def score_retrieval(
    docs: List[LangchainDocument],
    retrieval_info: dict,
) -> ConfidenceResult:
    """
    Compute retrieval confidence from docs + retrieval_info.

    retrieval_info shape (from hybrid_search_with_legs):
      {
        "legs": {
          "dense":         {"status": "ok"|"failed"|"disabled", "count": N},
          "qdrant_sparse": {...},
          "exact":         {...},
          "graph":         {...},
        },
        "failed_legs": ["dense", ...]
      }
    """
    failed_legs = retrieval_info.get("failed_legs", [])

    # Zero docs → none, regardless of mode or leg stats.
    if not docs:
        legs = retrieval_info.get("legs", {})
        enabled_legs = [k for k, v in legs.items() if v["status"] != "disabled"]
        return ConfidenceResult(
            level="none",
            score=0,
            suggestion="No relevant documents found. Try rephrasing your question or using different keywords.",
            breakdown={
                "mode": "reranker" if settings.RERANKER_ENABLED else "legacy",
                "total": 0,
                "enabled_legs": enabled_legs,
                "failed_legs": failed_legs,
                "docs_returned": 0,
            },
        )

    if settings.RERANKER_ENABLED:
        return _score_reranker(docs, failed_legs)
    else:
        return _score_legacy(docs, retrieval_info, failed_legs)


def _score_reranker(
    docs: List[LangchainDocument],
    failed_legs: list,
) -> ConfidenceResult:
    """Confidence based on reranker scores. All docs have passed the threshold."""
    reranker_scores = [
        doc.metadata["_reranker_score"]
        for doc in docs
        if "_reranker_score" in doc.metadata
    ]

    if not reranker_scores:
        # Reranker ran but didn't annotate — fall back to presence only.
        top_score = mean_score = 0.0
    else:
        top_score = max(reranker_scores)
        mean_score = sum(reranker_scores) / len(reranker_scores)

    def normalise(s: float) -> float:
        """Map logit [_RERANKER_SCORE_MIN, _RERANKER_SCORE_MAX] → [0, 1], clamped."""
        span = _RERANKER_SCORE_MAX - _RERANKER_SCORE_MIN
        return max(0.0, min((s - _RERANKER_SCORE_MIN) / span, 1.0))

    # A: top score (50 pts)
    a = normalise(top_score) * 50

    # B: doc count — stepped at 1/2/3+ (30 pts)
    n = len(docs)
    if n >= 3:
        b = 30.0
    elif n == 2:
        b = 20.0
    else:
        b = 10.0

    # C: mean score (20 pts)
    c = normalise(mean_score) * 20

    score = round(a + b + c)
    level, suggestion = _level_and_suggestion(score, failed_legs)

    breakdown = {
        "mode": "reranker",
        "top_reranker_score": round(top_score, 3),
        "mean_reranker_score": round(mean_score, 3),
        "top_score_pts": round(a),
        "doc_count_pts": round(b),
        "mean_score_pts": round(c),
        "total": score,
        "docs_returned": n,
        "failed_legs": failed_legs,
    }

    return ConfidenceResult(level=level, score=score, suggestion=suggestion, breakdown=breakdown)


def _score_legacy(
    docs: List[LangchainDocument],
    retrieval_info: dict,
    failed_legs: list,
) -> ConfidenceResult:
    """Original four-signal confidence model used when reranker is disabled."""
    legs  = retrieval_info.get("legs", {})
    top_k = settings.RETRIEVAL_TOP_K

    # A: source coverage
    enabled_legs   = [k for k, v in legs.items() if v["status"] != "disabled"]
    producing_legs = [k for k in enabled_legs if legs[k]["count"] > 0]
    a = len(producing_legs) / max(len(enabled_legs), 1)

    # B: cross-leg agreement
    multi_leg_count = sum(
        1 for doc in docs
        if len(doc.metadata.get("_legs", [])) >= 2
    )
    b = multi_leg_count / len(docs)

    # C: volume fill rate
    c = min(len(docs) / max(top_k, 1), 1.0)

    # D: source diversity
    sources = {doc.metadata.get("source") or doc.metadata.get("file_name") or "" for doc in docs}
    sources.discard("")
    d = min(len(sources), 3) / 3.0

    raw   = 30 * a + 35 * b + 25 * c + 10 * d
    score = round(raw)
    level, suggestion = _level_and_suggestion(score, failed_legs)

    breakdown = {
        "mode": "legacy",
        "source_coverage":     round(a * 30),
        "cross_leg_agreement": round(b * 35),
        "volume_fill":         round(c * 25),
        "source_diversity":    round(d * 10),
        "total":               score,
        "enabled_legs":        enabled_legs,
        "producing_legs":      producing_legs,
        "failed_legs":         failed_legs,
        "docs_returned":       len(docs),
        "top_k":               top_k,
    }

    return ConfidenceResult(level=level, score=score, suggestion=suggestion, breakdown=breakdown)
