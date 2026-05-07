"""
Retrieval confidence scoring.

Produces a score in [0, 100] from four independent signals derived from
the retrieval_info dict returned by hybrid_search_with_legs().

Signals
-------
A. Source coverage   (30 pts) — fraction of enabled legs that returned results
B. Cross-leg agreement (35 pts) — fraction of top-k chunks confirmed by ≥2 legs
C. Volume fill rate  (25 pts) — docs returned / RETRIEVAL_TOP_K
D. Source diversity  (10 pts) — unique source files in returned chunks (capped at 3)

Score → level
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


@dataclass
class ConfidenceResult:
    level: str                  # one of LEVELS
    score: int                  # 0-100
    suggestion: Optional[str]
    breakdown: dict             # per-signal scores for transparency


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
    legs        = retrieval_info.get("legs", {})
    failed_legs = retrieval_info.get("failed_legs", [])
    top_k       = settings.RETRIEVAL_TOP_K

    # ── A: Source coverage ─────────────────────────────────────────────────────
    # Fraction of *enabled* legs that returned at least one result.
    enabled_legs  = [k for k, v in legs.items() if v["status"] != "disabled"]
    producing_legs = [k for k in enabled_legs if legs[k]["count"] > 0]
    a = len(producing_legs) / max(len(enabled_legs), 1)

    # ── B: Cross-leg agreement ─────────────────────────────────────────────────
    # A chunk "agrees" across legs if it was found by ≥2 legs.
    # We approximate this from the per-chunk metadata that hybrid_search_with_legs
    # exposes: each doc's metadata["_legs"] list (added below).
    # If metadata is unavailable (legacy path), fall back to 0.
    if docs:
        multi_leg_count = sum(
            1 for doc in docs
            if len(doc.metadata.get("_legs", [])) >= 2
        )
        b = multi_leg_count / len(docs)
    else:
        b = 0.0

    # ── C: Volume fill rate ────────────────────────────────────────────────────
    c = min(len(docs) / max(top_k, 1), 1.0)

    # ── D: Source diversity ────────────────────────────────────────────────────
    # Unique source files, capped at 3 (beyond 3 there's diminishing confidence value)
    sources = {doc.metadata.get("source") or doc.metadata.get("file_name") or "" for doc in docs}
    sources.discard("")
    d = min(len(sources), 3) / 3.0

    # ── Weighted sum ───────────────────────────────────────────────────────────
    raw = 30 * a + 35 * b + 25 * c + 10 * d
    score = round(raw)

    # ── Map to level ───────────────────────────────────────────────────────────
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

    # ── Suggestion ─────────────────────────────────────────────────────────────
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

    breakdown = {
        "source_coverage":    round(a * 30),
        "cross_leg_agreement": round(b * 35),
        "volume_fill":        round(c * 25),
        "source_diversity":   round(d * 10),
        "total":              score,
        "enabled_legs":       enabled_legs,
        "producing_legs":     producing_legs,
        "failed_legs":        failed_legs,
        "docs_returned":      len(docs),
        "top_k":              top_k,
    }

    return ConfidenceResult(level=level, score=score, suggestion=suggestion, breakdown=breakdown)
