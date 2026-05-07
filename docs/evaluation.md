# RAG Evaluation

Automated evaluation of retrieval and answer quality using an external test harness.
The harness communicates with the app exclusively over HTTP — it has zero imports from
the RAG codebase and can run on any machine that can reach the API.

---

## Architecture

```
┌─────────────────────────────────┐        HTTP only
│       Eval Harness              │ ──────────────────► RAG App (FastAPI)
│  eval/eval_harness.py           │                         │
│                                 │  POST /api/auth/token   │
│  1. login                       │ ◄───────────────────────┤
│  2. create KB                   │  POST /api/knowledge-base
│  3. upload articles             │  POST /api/knowledge-base/{id}/documents/upload
│  4. trigger processing          │  POST /api/knowledge-base/{id}/documents/process
│  5. poll ingest-status          │  GET  /api/query/kb/{id}/ingest-status
│  6. run queries (per config)    │  POST /api/query
│  7. score + write report        │
└─────────────────────────────────┘
```

The RAG app needs no knowledge of evaluation — it just serves normal API requests.

---

## Endpoints

Two endpoints were added specifically to support external evaluation.
They live in `backend/app/api/api_v1/query.py`.

### POST /api/query

Stateless RAG query. No chat session is created, nothing is persisted.

**Request**
```json
{
  "question":       "What is Reciprocal Rank Fusion?",
  "kb_ids":         [1, 2],
  "use_dense":      true,
  "use_sparse":     true,
  "use_exact":      true,
  "use_graph_rag":  false,
  "generate_answer": true
}
```

Per-request leg flags AND with the global `.env` settings — a leg only runs when
both the request flag and the server-side `RETRIEVAL_*_ENABLED` flag are true.
Set `generate_answer: false` to measure retrieval quality only (no LLM tokens consumed).

**Response**
```json
{
  "question": "What is Reciprocal Rank Fusion?",
  "answer":   "RRF is a rank fusion method that combines ...",
  "contexts": [
    { "content": "RRF merges ranked lists by ...", "metadata": { "source": "paper.pdf" } }
  ],
  "confidence": "high",
  "suggestion": null,
  "retrieval_info": {
    "legs": {
      "dense":         { "status": "ok",       "count": 6 },
      "qdrant_sparse": { "status": "ok",       "count": 5 },
      "exact":         { "status": "ok",       "count": 4 },
      "graph":         { "status": "disabled", "count": 0 }
    },
    "failed_legs": []
  },
  "latency_ms": 412
}
```

Leg status values: `ok` | `failed` | `disabled`.

**Confidence levels**

| Value      | Meaning |
|------------|---------|
| `very_high`| Multi-leg agreement, full result set |
| `high`     | Good coverage, minimal failures |
| `medium`   | Partial retrieval or one leg failed |
| `low`      | Sparse results, multiple leg failures |
| `none`     | Zero documents retrieved |

### GET /api/query/kb/{kb_id}/ingest-status

Returns processing readiness for all documents in a knowledge base.
Poll this after triggering document processing; begin queries only when `ready: true`.

**Response**
```json
{
  "kb_id":     3,
  "total":     20,
  "completed": 20,
  "failed":    0,
  "pending":   0,
  "ready":     true
}
```

`ready` is `true` when `total > 0`, `completed == total`, and `failed == 0`.

---

## How RRF works in the eval context

Every chunk that appears in any enabled leg's result list gets a score:

```
rrf_score = Σ  weight_leg / (60 + rank_leg)
             for each leg where the chunk appeared
```

`60` is the smoothing constant from the original paper. Weights come from `.env`:

```
HYBRID_DENSE_WEIGHT          0.5   (dense vectors)
HYBRID_QDRANT_SPARSE_WEIGHT  0.3   (SPLADE sparse vectors)
HYBRID_EXACT_WEIGHT          0.2   (MySQL keyword / FTS)
HYBRID_GRAPH_WEIGHT          0.3   (Neo4j GraphRAG)
```

A chunk absent from a disabled leg contributes 0 from that leg but can still
surface via the remaining legs. With only one leg active, RRF degenerates to
the original ranking of that leg — no fusion happens, which is why single-leg
runs are useful as clean baselines.

**Example — 2 legs enabled (dense + exact):**

```
chunk A: dense_rank=0, exact_rank=2
  score = 0.5/(60+0) + 0.2/(60+2) = 0.00833 + 0.00323 = 0.01156

chunk B: dense_rank=4, exact_rank=0
  score = 0.5/(60+4) + 0.2/(60+0) = 0.00781 + 0.00333 = 0.01114
```

Chunk A wins because top-1 dense (weight 0.5) outweighs top-1 exact (weight 0.2).
Add sparse and the balance shifts again — which is exactly what the harness measures.

---

## Eval Harness

### Location

```
eval/
├── eval_harness.py     standalone script, no RAG app imports
└── requirements.txt    requests, datasets, tqdm
```

### Setup

```bash
cd eval
pip install -r requirements.txt
```

### Modes

The harness runs in two modes depending on whether leg flags are passed.

#### Sweep mode (no leg flags — default)

Runs all predefined retrieval configurations against the same question set and
prints a comparison table. Use this to benchmark the effect of each source.

```bash
python eval_harness.py \
    --base-url  http://localhost:8000/api \
    --username  eval_user \
    --password  yourpassword \
    --articles  20 \
    --questions 60
```

Configs included in a sweep:

| Config name       | Legs active                          |
|-------------------|--------------------------------------|
| `exact_only`      | Keyword only (baseline)              |
| `dense_only`      | Dense vectors only                   |
| `sparse_only`     | Sparse vectors (SPLADE) only         |
| `dense+sparse`    | Dense + Sparse, no keyword           |
| `dense+exact`     | Dense + Keyword                      |
| `sparse+exact`    | Sparse + Keyword                     |
| `all_3`           | Dense + Sparse + Keyword (full hybrid) |
| `all_3+graph`     | Full hybrid + Knowledge Graph (opt-in via `--graph`) |

Add `--graph` to include the `all_3+graph` config (requires Neo4j and GraphRAG to be configured).

#### Single-config mode (leg flags passed)

Runs exactly one combination — the one you specify. Useful for quick targeted tests
without waiting for a full sweep.

```bash
# keyword only (exact is always on, no other flags needed)
python eval_harness.py --kb-id 3 --questions 60

# dense + keyword
python eval_harness.py --kb-id 3 --use-dense --questions 60

# sparse + keyword
python eval_harness.py --kb-id 3 --use-sparse --questions 60

# dense + sparse + keyword (full hybrid, no graph)
python eval_harness.py --kb-id 3 --use-dense --use-sparse --questions 60

# everything including knowledge graph
python eval_harness.py --kb-id 3 --use-dense --use-sparse --use-kg --questions 60
```

Keyword search (`exact`) is always enabled and cannot be turned off — this matches
the application behaviour where keyword search is the always-on baseline.

### All flags

| Flag                | Env var    | Default                     | Description |
|---------------------|------------|-----------------------------|-------------|
| `--base-url`        | `BASE_URL` | `http://localhost:8000/api` | RAG API base URL |
| `--username`        | `USERNAME` | `eval_user`                 | Login username |
| `--password`        | `PASSWORD` | `eval_pass`                 | Login password |
| `--email`           | `EMAIL`    | `eval@example.com`          | Used on first registration |
| `--articles`        | —          | `20`                        | SQuAD articles to ingest |
| `--questions`       | —          | `60`                        | Questions to evaluate |
| `--use-dense`       | —          | off                         | Enable dense vector leg |
| `--use-sparse`      | —          | off                         | Enable sparse vector (SPLADE) leg |
| `--use-kg`          | —          | off                         | Enable knowledge graph (Neo4j) leg |
| `--graph`           | —          | off                         | Include `all_3+graph` in sweep |
| `--generate-answers`| —          | off                         | Run LLM answer generation (costs tokens) |
| `--kb-id`           | —          | —                           | Reuse an existing KB (skip ingest) |
| `--keep-kb`         | —          | off                         | Don't delete the KB after eval |
| `--output`          | —          | `eval_results.json`         | Output file path |

### Reusing a KB

Ingestion is slow and only needs to happen once. After the first run, note the
`kb_id` from the output and pass `--kb-id` on subsequent runs to skip re-upload:

```bash
# First run — ingests and evaluates
python eval_harness.py --articles 20 --questions 60 --keep-kb
# note kb_id from output, e.g. 7

# Subsequent runs — evaluation only, no re-ingest
python eval_harness.py --kb-id 7 --questions 60
python eval_harness.py --kb-id 7 --use-dense --questions 60
python eval_harness.py --kb-id 7 --use-sparse --use-kg --questions 60
```

---

## Dataset — SQuAD 2.0

SQuAD (Stanford Question Answering Dataset) is the standard benchmark for
extractive QA. Each row contains:

- `context`  — a Wikipedia paragraph
- `question` — a question about that paragraph
- `answers`  — one or more correct answer spans extracted from the context

The harness uploads contexts as plain-text documents, queries with the questions,
and compares the LLM answer against the ground truth spans.

SQuAD 2.0 (`squad_v2`) adds unanswerable questions — the harness skips those
automatically (they have empty `answers.text`).

Why SQuAD:
- Free, no API key, cached after first download (~35 MB)
- Ground truth answers are short extractive spans — easy to score without an LLM judge
- Widely used, so scores are comparable across systems

---

## Metrics

### Token-F1

The official SQuAD metric. Computes overlap between predicted and reference answer
at the token level after lowercasing and stripping punctuation.

```
precision = overlap_tokens / predicted_tokens
recall    = overlap_tokens / reference_tokens
F1        = 2 * precision * recall / (precision + recall)
```

Computed as `max(F1)` over all reference answers for a question.
Only meaningful when `--generate-answers` is passed; otherwise 0.

### Exact Match

1 if the normalised prediction equals any normalised reference answer, 0 otherwise.

### Hit Rate

1 if any ground truth answer string appears (substring match, case-insensitive) in
any retrieved chunk. Measures retrieval quality independent of answer generation —
useful when running without `--generate-answers`.

### Interpreting scores

| Token-F1   | Interpretation |
|------------|----------------|
| > 0.65     | Good — retrieval and extraction both working |
| 0.40–0.65  | Retrieval probably OK, answer synthesis weak |
| < 0.40     | Likely retrieval misses — check chunk size, embedding model |

- Low F1 + high `confidence: none` count → retrieval not finding the right chunks.
  Try reducing `CHUNK_SIZE`, switching embedding model, or enabling more legs.
- Low F1 + high `confidence: high` count → retrieval is finding chunks but the LLM
  is not extracting correctly. Check the system prompt or model.
- High hit rate + low F1 → chunks contain the answer but the LLM isn't using it.

---

## Output format

`eval_results.json` top-level structure:

```json
{
  "timestamp":        "2026-05-07T14:30:22Z",
  "kb_id":            7,
  "n_articles":       20,
  "n_questions":      60,
  "generate_answers": false,
  "configs_run":      ["exact_only", "dense_only", "all_3"],
  "summary": [
    {
      "config":          "exact_only",
      "label":           "Keyword only (baseline)",
      "mean_f1":         0.312,
      "mean_em":         0.201,
      "hit_rate":        0.483,
      "mean_latency_ms": 142,
      "errors":          0
    }
  ],
  "details": {
    "exact_only": [
      {
        "question":   "To whom did the Virgin Mary allegedly appear in 1858?",
        "answers":    ["Saint Bernadette Soubirous"],
        "prediction": "",
        "f1":         0.0,
        "em":         0.0,
        "hit":        1.0,
        "confidence": "high",
        "latency_ms": 138,
        "legs": {
          "dense":         { "status": "disabled", "count": 0 },
          "qdrant_sparse": { "status": "disabled", "count": 0 },
          "exact":         { "status": "ok",       "count": 6 },
          "graph":         { "status": "disabled", "count": 0 }
        }
      }
    ]
  }
}
```

The `summary` array is ordered by config (same order as `configs_run`) and is
sufficient for a comparison table. The `details` dict contains per-question
breakdowns keyed by config name for deeper analysis.

---

## Error handling

- A question that gets an HTTP error is recorded with `error: "<message>"` and
  scores of 0 — it does not abort the run.
- If any document fails processing (`failed > 0` in ingest-status), the harness
  logs a warning and continues — partial KBs are valid for benchmarking purposes.
- The eval KB is deleted after the run by default. Pass `--keep-kb` to retain it
  for inspection or reuse with `--kb-id`.
