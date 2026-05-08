# Search Implementation

## Overview

Retrieval uses a **3-leg hybrid pipeline** fused with Reciprocal Rank Fusion (RRF):

- **Leg 1 — Dense**: Qdrant cosine-similarity search on OpenAI-compatible embeddings
- **Leg 2 — Sparse**: Qdrant learned-sparse search (SPLADE via FastEmbed, CPU-local)
- **Leg 3 — Exact**: MySQL InnoDB FULLTEXT search (BM25/TF-IDF, server-side)

All three legs run independently; their ranked lists are merged by weighted RRF. Each leg covers the failure modes of the others: dense handles paraphrases/synonyms, sparse captures term-frequency signal for technical vocabulary, exact matches product codes and precise keywords that embeddings may blur.

Individual legs can be disabled via `.env` without re-indexing — ingestion always writes to all three stores.

---

## Pipeline

```
User query
    │
    ▼
[chat_service.py] Condense with chat history → standalone question
    │
    ▼
[retrieval.py] hybrid_search_with_legs()
    ├── _dense_search()          → Qdrant cosine similarity (dense vectors)
    ├── _qdrant_sparse_search()  → Qdrant SPLADE sparse vectors
    └── _exact_search()          → MySQL InnoDB FULLTEXT (NATURAL LANGUAGE MODE)
    │          ↓
    └── _rrf_merge_candidates()  → weighted RRF score → top-K LangchainDocuments
    │
    ▼ (optional: GRAPHRAG_ENABLED=true and use_graph_rag=True)
[graph_service.py] enrich_docs_with_graph()
    └── Look up Chunk by (document_id, chunk_index) in Neo4j
        → traverse entity relationships
        → append [Graph context] triples to each doc's text
    │
    ▼ (optional: RERANKER_ENABLED=true)
[reranker.py] cross-encoder reranking — threshold filter, no top-N cap
    │
    ▼
[chat_service.py] Build prompt → stream LLM response
```

### GraphRAG enrichment — not a retrieval leg

Graph enrichment runs **after** RRF merge, not as a scored leg alongside dense/sparse/exact. Neo4j is queried by `(document_id, chunk_index)` — the same identifiers stored in every Qdrant point payload, established as a cross-reference link at ingest time.

This matches the Qdrant+Neo4j reference architecture:
- **Qdrant** finds the relevant chunks via vector search
- **Neo4j** enriches those chunks with entity/relationship context
- The enriched text is reranked by the cross-encoder before being sent to the LLM

Neo4j never runs its own vector index — that would duplicate Qdrant's work.

---

## Files

| File | Role |
|------|------|
| `backend/app/services/retrieval.py` | Complete hybrid search implementation |
| `backend/app/services/chat_service.py` | Calls `hybrid_search()`, builds prompt, streams response |
| `backend/app/core/config.py` | All tunable retrieval parameters |
| `.env` / `.env.example` | Runtime configuration |

---

## Implementation Detail

### Entry point — `hybrid_search()`

`backend/app/services/retrieval.py`

```python
async def hybrid_search(query, kb_ids, db) -> List[LangchainDocument]:
    top_k = settings.RETRIEVAL_TOP_K
    pool = top_k * 4   # each leg over-fetches so RRF has room to rerank

    dense         = _dense_search(query, kb_ids, pool)          if enabled["dense"]          else {}
    qdrant_sparse = _qdrant_sparse_search(query, kb_ids, pool)  if enabled["qdrant_sparse"]  else {}
    exact         = _exact_search(query, kb_ids, db, pool)      if enabled["exact"]          else {}

    return _rrf_merge(dense, qdrant_sparse, exact, top_k)
```

`pool = top_k * 4` ensures a document ranked #20 by one leg but #1 by another is not discarded before the merge.

---

### Dense leg — `_dense_search()`

Embeds the query with the configured OpenAI-compatible embedding model, then queries each knowledge base's Qdrant collection using cosine distance on the `dense` named vector.

```python
response = _get_openai_client().embeddings.create(input=query, model=settings.OPENAI_EMBEDDINGS_MODEL)
query_vector = response.data[0].embedding

hits = _get_qdrant_client().query_points(
    collection_name=f"kb_{kb_id}",
    query=query_vector,
    using="dense",
    limit=candidates,
    with_payload=True,
).points
```

Qdrant returns scored points; they are ranked in arrival order (Qdrant returns results sorted by score descending).

---

### Sparse leg — `_qdrant_sparse_search()`

Embeds the query with the FastEmbed SPLADE model to produce a sparse (indices + values) vector, then queries Qdrant's `sparse` named vector index.

```python
sparse_emb = next(iter(_get_sparse_embedder().embed([query])))
query_sparse = SparseVector(
    indices=sparse_emb.indices.tolist(),
    values=sparse_emb.values.tolist(),
)

hits = _get_qdrant_client().query_points(
    collection_name=f"kb_{kb_id}",
    query=query_sparse,
    using="sparse",
    limit=candidates,
    with_payload=True,
).points
```

SPLADE produces term-weighted sparse vectors in BERT vocabulary space. These capture TF-IDF-like signal with learned expansion, beating raw BM25 on recall while remaining interpretable.

---

### Exact leg — `_exact_search()`

MySQL InnoDB FULLTEXT search in `NATURAL LANGUAGE MODE`, which applies server-side BM25/TF-IDF ranking. No client-side index to build or maintain.

```python
sql = text("""
    SELECT chunk_text, chunk_metadata,
           MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS fts_score
    FROM   document_chunks
    WHERE  kb_id IN :kb_ids
      AND  MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
    ORDER  BY fts_score DESC
    LIMIT  :candidates
""").bindparams(bindparam("kb_ids", expanding=True))
```

Only rows with `fts_score > 0` are returned — MySQL omits documents with no query-term overlap, matching BM25 semantics without an in-memory `score == 0` guard.

---

### RRF merge — `_rrf_merge()`

```python
def _rrf_merge(dense, qdrant_sparse, exact, top_k):
    merged = {**dense}
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
    ranked = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
    return [c.doc for c in ranked[:top_k]]
```

#### RRF score formula

```
score(doc) = HYBRID_DENSE_WEIGHT         / (60 + dense_rank)
           + HYBRID_QDRANT_SPARSE_WEIGHT / (60 + qdrant_sparse_rank)
           + HYBRID_EXACT_WEIGHT         / (60 + exact_rank)
```

A leg absent for a document (rank == -1) contributes 0. The constant 60 is from the original RRF paper (Cormack et al., 2009) — it prevents the top-ranked document from dominating disproportionately.

#### The eight cases (three binary legs)

| Dense | Sparse | Exact | Outcome |
|-------|--------|-------|---------|
| hit   | hit    | hit   | All three legs contribute — strongest signal |
| hit   | hit    | miss  | Dense + sparse; keyword may not match exactly |
| hit   | miss   | hit   | Dense + exact; SPLADE missed the query terms |
| miss  | hit    | hit   | Sparse + exact; dense embedding didn't fire |
| hit   | miss   | miss  | Dense only; paraphrase / synonym match |
| miss  | hit    | miss  | Sparse only; unusual term weighting |
| miss  | miss   | hit   | Exact only; keyword match, no semantic overlap |
| miss  | miss   | miss  | Excluded — no signal from any leg |

---

### Scoring dataclass — `_Candidate`

```python
@dataclass
class _Candidate:
    doc: LangchainDocument
    content_hash: str
    dense_rank: int = -1           # -1 = absent from this leg
    qdrant_sparse_rank: int = -1
    exact_rank: int = -1

    @property
    def rrf_score(self) -> float:
        score = 0.0
        if self.dense_rank >= 0:
            score += settings.HYBRID_DENSE_WEIGHT / (_RRF_K + self.dense_rank)
        if self.qdrant_sparse_rank >= 0:
            score += settings.HYBRID_QDRANT_SPARSE_WEIGHT / (_RRF_K + self.qdrant_sparse_rank)
        if self.exact_rank >= 0:
            score += settings.HYBRID_EXACT_WEIGHT / (_RRF_K + self.exact_rank)
        return score
```

`-1` sentinel separates "not ranked" from "ranked last". A rank of 0 is the best possible position.

### Deduplication — `_content_hash()`

Multiple knowledge bases may contain the same chunk (e.g. a shared onboarding document). SHA-256 of the chunk text is the merge key — duplicates are collapsed to one candidate before scoring.

---

## Configuration

All parameters live in `backend/app/core/config.py` and are set via `.env`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `6` | Number of chunks returned to the LLM |
| `HYBRID_DENSE_WEIGHT` | `0.5` | RRF weight for the dense (embedding) leg |
| `HYBRID_QDRANT_SPARSE_WEIGHT` | `0.3` | RRF weight for the SPLADE sparse leg |
| `HYBRID_EXACT_WEIGHT` | `0.2` | RRF weight for the MySQL FTS leg |
| `RETRIEVAL_DENSE_ENABLED` | `true` | Enable/disable the dense leg |
| `RETRIEVAL_QDRANT_SPARSE_ENABLED` | `true` | Enable/disable the sparse leg |
| `RETRIEVAL_EXACT_ENABLED` | `true` | Enable/disable the exact leg |
| `DENSE_EMBEDDING_DIM` | `1024` | Output dimension of the embedding model |
| `SPLADE_MODEL` | `prithivida/Splade_PP_en_v1` | FastEmbed model name for SPLADE |

Weights are relative — they don't need to sum to 1. Raise `HYBRID_EXACT_WEIGHT` for corpora with precise terminology (legal, medical, part numbers). Raise `HYBRID_DENSE_WEIGHT` for conversational or paraphrase-heavy content.

Disabling a leg affects retrieval only. Ingestion always indexes all three stores, so re-enabling a leg later requires no re-indexing.

---

## Where retrieval is called

`backend/app/services/chat_service.py`, inside `generate_response()`:

```python
# Retrieve relevant chunks via 3-leg hybrid search
docs = await hybrid_search(
    query=standalone_question,
    kb_ids=knowledge_base_ids,
    db=db,
)
```

The query passed is the **condensed standalone question** — chat history context has been folded in by the preceding summarisation/sliding-window step. This ensures the retrieval query is self-contained and does not depend on pronouns or conversational references that keyword search would fail to resolve.

---

## Performance notes

- Dense and sparse legs query Qdrant (a compiled Rust service) — both are fast even for large collections.
- The exact leg runs a native MySQL FULLTEXT query; InnoDB FTS indexes are persistent and maintained incrementally on insert.
- The candidate pool (`top_k * 4`) adds some overhead but is necessary for RRF correctness — without headroom, a document ranked #1 by one leg but outside the top-K of another would be invisible to the merge.
- SPLADE model (~500 MB) is loaded once per process and kept in memory as a module-level singleton.
