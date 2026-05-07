# Retrieval Confidence Scoring

Every query response includes a confidence score that reflects how well the
retrieval system found relevant content for the question. The score is computed
from four independent signals and mapped to one of five levels displayed in the
UI as a stepped progress bar.

---

## Levels

| Level     | Score range | Bar steps | Meaning |
|-----------|-------------|-----------|---------|
| Very High | 80 – 100    | ████      | All retrieval sources fired, strong cross-leg agreement, full result set |
| High      | 55 – 79     | ███░      | Most sources found results, good agreement |
| Medium    | 30 – 54     | ██░░      | Partial results, some sources missed or disagreed |
| Low       | 1 – 29      | █░░░      | Very few results, most sources found nothing |
| None      | 0           | ░░░░      | Zero documents retrieved |

---

## Scoring model

The score is a weighted sum of four signals, each normalised to [0, 1]:

```
score = 30·A  +  35·B  +  25·C  +  10·D
```

### A — Source coverage  (30 pts)

Fraction of enabled retrieval legs that returned at least one result.

```
A = producing_legs / enabled_legs
```

An enabled leg that returns zero results contributes 0. A disabled leg (turned
off via `.env`) is excluded from the denominator entirely — disabling a leg does
not penalise the score.

Example: dense=ok, sparse=ok, exact=0 results, graph=disabled → A = 2/3 = 0.67 → 20 pts

### B — Cross-leg agreement  (35 pts, highest weight)

Fraction of the top-k chunks that were independently found by two or more
retrieval legs before RRF merging.

```
B = chunks_found_by_≥2_legs / total_chunks_returned
```

This is the strongest signal. A chunk that dense vector search AND keyword search
both rank highly is very unlikely to be a false positive. A chunk surfaced by only
one leg may be a weak match.

The implementation tracks which legs contributed to each chunk via
`metadata["_legs"]`, written by `hybrid_search_with_legs()` before the docs are
returned.

### C — Volume fill rate  (25 pts)

How many of the requested top-k slots were actually filled.

```
C = min(docs_returned / RETRIEVAL_TOP_K, 1.0)
```

Getting a full result set (e.g. 6/6) is a positive signal. Getting 2/6 suggests
the KB has sparse coverage for this query.

### D — Source diversity  (10 pts)

Number of distinct source files among the returned chunks, capped at 3.

```
D = min(unique_sources, 3) / 3
```

Chunks from multiple documents are a mild positive signal — a single document
dominating the results can indicate an overly narrow match. The signal is weak
(10 pts max) because a single highly relevant document should not be penalised.

---

## Suggestions

When confidence is below Very High, a human-readable suggestion is included in
the response. Priority order:

1. Failed legs — if any leg errored (infrastructure issue), the suggestion names
   the unavailable sources regardless of score.
2. None — no documents found; suggests rephrasing.
3. Low — few documents; suggests more specific keywords.
4. Medium — partial results; suggests rephrasing for better coverage.
5. High / Very High — no suggestion shown.

---

## Implementation

| File | Role |
|------|------|
| `backend/app/services/confidence.py` | Core scoring logic (`score_retrieval()`) |
| `backend/app/services/retrieval.py`  | Annotates each doc with `metadata["_legs"]` inside `hybrid_search_with_legs()` |
| `backend/app/services/chat_service.py` | Calls `score_retrieval()`, emits result in `2:` stream event |
| `backend/app/api/api_v1/query.py` | Same for the stateless `/query` endpoint |
| `frontend/src/components/chat/answer.tsx` | `ConfidenceBar` component + `Answer` prop wiring |

### Stream event payload (`2:`)

```json
{
  "context":     [...],
  "confidence":  "high",
  "score":       67,
  "suggestion":  null,
  "failed_legs": [],
  "breakdown": {
    "source_coverage":     20,
    "cross_leg_agreement": 28,
    "volume_fill":         25,
    "source_diversity":    7,
    "total":               67,
    "enabled_legs":        ["dense", "qdrant_sparse", "exact"],
    "producing_legs":      ["dense", "qdrant_sparse", "exact"],
    "failed_legs":         [],
    "docs_returned":       6,
    "top_k":               6
  }
}
```

The `breakdown` field is available in the stream for debugging but is not
currently displayed in the UI.

---

## UI

`ConfidenceBar` renders immediately when the `2:` event arrives — before the LLM
starts generating the answer. It shows:

- Label: "Retrieval confidence · Very High · 92/100"
- Four rectangular step segments, filled left-to-right based on level
- Colour-coded per level (emerald / green / yellow / orange / red)
- Suggestion text below the bar when present

The bar is hidden for `confidence = "none"` — instead a distinct amber warning
banner is shown, since there are no results to qualify.

---

## Tuning

The weights are in `confidence.py` as plain arithmetic — change them there.
The level thresholds (80 / 55 / 30) are also in the same file.

Signal B (cross-leg agreement) is intentionally the highest weight. If your
deployment uses only one retrieval leg (e.g. dense-only), B will always be 0 and
scores will be structurally capped at 65. In that case consider raising the
weight of A and C to compensate, or enabling additional legs.
