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
│  6. run queries                 │  POST /api/query
│  7. score + write report        │
└─────────────────────────────────┘
```

The RAG app needs no knowledge of evaluation — it just serves normal API requests.

---

## New Endpoints

Two endpoints were added specifically to support external evaluation.
They live in `backend/app/api/api_v1/query.py`.

### POST /api/query

Stateless RAG query. No chat session is created, nothing is persisted.

**Request**
```json
{
  "question": "What is Reciprocal Rank Fusion?",
  "kb_ids": [1, 2],
  "use_graph_rag": false,
  "generate_answer": true
}
```

Set `generate_answer: false` to measure retrieval quality only (no LLM tokens consumed).

**Response**
```json
{
  "question": "What is Reciprocal Rank Fusion?",
  "answer": "RRF is a rank fusion method that combines ...",
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

**Confidence values**

| Value  | Meaning |
|--------|---------|
| `high` | Full result set, no leg failures |
| `low`  | Fewer than half of `RETRIEVAL_TOP_K` docs returned, or at least one leg failed |
| `none` | Zero documents retrieved |

### GET /api/query/kb/{kb_id}/ingest-status

Returns processing readiness for all documents in a knowledge base.
Poll this after triggering document processing; begin queries only when `ready: true`.

**Response**
```json
{
  "kb_id": 3,
  "total": 20,
  "completed": 20,
  "failed": 0,
  "pending": 0,
  "ready": true
}
```

`ready` is `true` when `total > 0`, `completed == total`, and `failed == 0`.

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

### Usage

```bash
python eval_harness.py \
    --base-url  http://localhost:8000/api \
    --username  eval_user \
    --password  yourpassword \
    --articles  20 \
    --questions 60 \
    --output    eval_results.json
```

All flags can also be set via environment variables:

| Flag          | Env var          | Default                          |
|---------------|------------------|----------------------------------|
| `--base-url`  | `RAG_BASE_URL`   | `http://localhost:8000/api`      |
| `--username`  | `RAG_USERNAME`   | `admin`                          |
| `--password`  | `RAG_PASSWORD`   | `admin`                          |
| `--articles`  | —                | `20`                             |
| `--questions` | —                | `60`                             |
| `--dataset`   | —                | `squad` (`squad_v2` also works)  |
| `--output`    | —                | `eval_results.json`              |
| `--no-cleanup`| —                | KB is deleted after run by default |

### What it does

1. Registers the eval user if they don't exist (idempotent)
2. Logs in and acquires a bearer token
3. Loads the SQuAD validation split from HuggingFace (cached locally after first download)
4. Creates a fresh KB named `eval_squad_YYYYMMDD_HHMMSS`
5. Uploads each article as a `.txt` file and triggers processing
6. Polls `GET /query/kb/{id}/ingest-status` until `ready: true` (10-minute timeout)
7. Sends each question to `POST /query` and collects the JSON response
8. Scores each answer with token-F1 and exact-match against ground truth
9. Writes `eval_results.json` and prints a summary table
10. Deletes the eval KB (unless `--no-cleanup`)

---

## Dataset — SQuAD 2.0

SQuAD (Stanford Question Answering Dataset) is the standard benchmark for
extractive QA. Each row contains:

- `context` — a Wikipedia paragraph
- `question` — a question about that paragraph
- `answers` — one or more correct answer spans extracted from the context

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
at the token level after lowercasing and stripping articles/punctuation.

```
precision = overlap_tokens / predicted_tokens
recall    = overlap_tokens / reference_tokens
F1        = 2 * precision * recall / (precision + recall)
```

Computed as `max(F1)` over all reference answers for a question.

### Exact Match

1 if the normalized prediction equals any normalized reference answer, 0 otherwise.

### Interpreting scores

| Token-F1   | Interpretation |
|------------|----------------|
| > 0.65     | Good — retrieval and extraction both working |
| 0.40–0.65  | Retrieval probably OK, answer synthesis weak |
| < 0.40     | Likely retrieval misses — check chunk size, embedding model |

Low F1 with high `confidence: none` count → retrieval not finding the right chunks.
Try reducing `CHUNK_SIZE` or switching embedding model.

Low F1 with high `confidence: high` count → retrieval is finding chunks but the LLM
is not extracting correctly. Check the system prompt or model.

---

## Output format

`eval_results.json` top-level structure:

```json
{
  "dataset":          "squad",
  "timestamp":        "20260507_143022",
  "rag_base_url":     "http://localhost:8000/api",
  "num_articles":     20,
  "num_questions":    60,
  "mean_f1":          0.6231,
  "exact_match_pct":  41.7,
  "confidence_dist":  { "high": 48, "low": 9, "none": 3 },
  "avg_latency_ms":   387.4,
  "avg_contexts":     5.8,
  "questions": [
    {
      "question_id":  "5733be284776f41900661182",
      "question":     "To whom did the Virgin Mary allegedly appear in 1858?",
      "ground_truth": ["Saint Bernadette Soubirous"],
      "answer":       "According to the context, the Virgin Mary appeared to Saint Bernadette Soubirous.",
      "confidence":   "high",
      "suggestion":   null,
      "token_f1":     0.6667,
      "exact_match":  false,
      "num_contexts": 6,
      "latency_ms":   341,
      "failed_legs":  [],
      "error":        null
    }
  ]
}
```

---

## Error handling during eval

- A question that gets an HTTP error is recorded with `error: "<message>"` and
  `token_f1: 0.0` — it does not abort the run.
- If any document fails processing (`failed > 0` in ingest-status), the harness
  raises immediately rather than running queries against an incomplete KB.
- The eval KB is always cleaned up in the `finally` block even if the run errors.
  Pass `--no-cleanup` to keep it for inspection.
