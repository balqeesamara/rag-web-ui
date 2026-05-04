# Vector Representations: Dense vs Sparse

## Overview

Each chunk produces two vectors stored as named fields on a single Qdrant point:

| Field | Dimensions | Varies per chunk? |
|-------|-----------|-------------------|
| `dense` | Fixed — always `DENSE_EMBEDDING_DIM` (1024) | No |
| `sparse` | Variable — count of non-zero entries | Yes |

---

## Dense Vectors: Fixed Dimensions

Dense vectors are fixed-dim because the embedding model maps any input —
regardless of length or content — to a point in a fixed-dimensional space.
Qwen3-embedding-0.6b always outputs exactly 1024 floats. Every position in that
vector has a learned meaning baked in during model training. The model compresses
the entire semantic content of a chunk into that fixed shape.

Two chunks of completely different content still produce 1024-float vectors — the
values differ, the shape doesn't. This fixed shape is what makes efficient
approximate nearest-neighbour (ANN) search possible: Qdrant indexes all points in
the same geometric space and finds the closest ones by cosine similarity.

---

## Sparse Vectors: Variable Dimensions

Sparse vectors are variable because they represent term weights in vocabulary
space. SPLADE's vocabulary has ~30,000 BERT token IDs. A sparse vector is
conceptually a 30,000-dimensional vector where almost all entries are zero —
so Qdrant stores only the non-zero `(index, value)` pairs.

Different chunks activate different subsets of the vocabulary:

- A chunk about databases might activate tokens for "query", "index", "table",
  "schema" → ~150 non-zero entries
- A chunk about cooking might activate "heat", "oil", "pan", "season" → ~120
  non-zero entries at completely different indices
- A short chunk ("See chapter 3.") might only activate 20–30 tokens

The *space* is always 30,000 dimensions. What varies is how many of those
dimensions are non-zero for a given chunk — determined by its vocabulary richness
and length. What Qdrant exposes in the dashboard as "variable dims" is the count
of non-zero entries, not a different vector space per chunk.

### Why store only non-zeros?

Storing 30,000 floats per chunk when 29,850 of them are zero would be wasteful.
Qdrant's sparse index stores only the occupied `(index, value)` pairs. A typical
chunk occupies 100–300 entries — roughly 1% of the full vocabulary space.

### Why sparse search is efficient

Scoring a query against a stored sparse vector is a dot product over the
*intersection* of their non-zero indices. If the query activates 80 tokens and
the stored chunk has 150 non-zeros, the actual computation touches at most 80
positions — not 30,000. The sparsity is the performance mechanism, not just a
storage optimisation.

---

## Side-by-side Comparison

| Property | Dense | Sparse |
|----------|-------|--------|
| Model | Qwen3-embedding-0.6b (or any configured model) | SPLADE PP en v1 (FastEmbed) |
| Vector space | 1024-dim continuous | ~30,000-dim discrete (BERT vocab) |
| Stored shape | 1024 floats, always | N non-zero (index, value) pairs |
| Typical N | — | 100–300 per chunk |
| What it captures | Semantic meaning — synonyms, paraphrases | Term weights + learned vocabulary expansion |
| Search method | Cosine ANN (Qdrant HNSW) | Dot product over non-zero intersections |
| Context limit | 32,768 tokens (Qwen3) | ~512 tokens (BERT — silent truncation beyond ~1800 chars) |

---

## Related Docs

- [ingestion-pipeline.md](ingestion-pipeline.md) — how both vectors are produced
  and stored per chunk
- [chunking.md](chunking.md) — why chunk size is bounded by SPLADE's 512-token
  limit
- [search-implementation.md](search-implementation.md) — how dense and sparse
  vectors are queried and fused at retrieval time
