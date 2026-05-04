# Document Ingestion Pipeline

## Overview

When a document is uploaded it goes through a 7-step pipeline that ends with each
text chunk stored in two places: Qdrant (for vector search) and MySQL (for
full-text search). Every chunk produces exactly **2 vectors** — one dense, one
sparse — stored as named vector fields on a single Qdrant point.

---

## Steps

### 1. Load

File type determines the loader:

| Extension | Loader |
|-----------|--------|
| `.pdf` | `PyPDFLoader` (pypdf) |
| `.docx` | `Docx2txtLoader` (docx2txt) |
| `.md` | `TextLoader` |
| `.txt` | `TextLoader` |

### 2. Chunk

`RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`
(characters, not tokens).

The splitter tries to break on paragraph boundaries first, then sentences, then
words, falling back to raw characters only as a last resort. The 200-character
overlap repeats the tail of each chunk at the head of the next — preserving
sentence context across boundaries.

### 3. Deduplicate

SHA-256 of `(chunk_text + metadata)` is computed for every chunk. If the hash
already exists in MySQL for that file, the chunk is skipped. Only new or changed
chunks proceed to embedding. This makes re-processing an updated document cheap —
unchanged sections are not re-embedded or re-indexed.

### 4. Dense Embedding

Chunk texts are sent to the configured OpenAI-compatible embedding API
(`OPENAI_EMBEDDINGS_MODEL`) in batches of 32.

Each chunk produces one `List[float]` of length `DENSE_EMBEDDING_DIM` (default
1024 for local models such as `qwen3-embedding-0.6b`; 1536 for
`text-embedding-3-small` / `text-embedding-ada-002`).

This is a semantic embedding — the entire chunk is compressed into a single point
in a continuous vector space where similar meanings cluster together regardless of
exact wording.

### 5. Sparse Embedding (SPLADE)

The same chunk texts are passed through FastEmbed's `SparseTextEmbedding`
(`prithivida/Splade_PP_en_v1` by default). Each chunk produces a `SparseVector`:
two arrays — `indices` and `values` — where the vast majority of entries are zero
and only a small subset carry non-zero weights.

See [What sparse vectors represent](#what-sparse-vectors-represent) below.

### 6. Upsert to Qdrant

Each chunk becomes one Qdrant `PointStruct`:

```
PointStruct(
    id     = UUID derived deterministically from SHA-256(chunk_id),
    vector = {
        "dense":  [float, ...]                             # DENSE_EMBEDDING_DIM floats
        "sparse": SparseVector(indices=[...], values=[...])
    },
    payload = {
        "chunk_text":   "...",
        "kb_id":        <int>,
        "document_id":  <int>,
        "file_name":    "...",
        "chunk_index":  <int>,
        # plus any source metadata — e.g. page number for PDFs
    }
)
```

Points are upserted in batches of 100 (`_QDRANT_UPSERT_BATCH`).

### 7. Store in MySQL

The chunk text and metadata are also inserted into the `document_chunks` table
so that MySQL's InnoDB FULLTEXT index can serve the exact-search leg at query
time. This is the same table the exact retrieval leg queries with
`MATCH(...) AGAINST(... IN NATURAL LANGUAGE MODE)`.

---

## What Sparse Vectors Represent

SPLADE (SParse Lexical AnD Expansion) maps text into a sparse vector in BERT's
vocabulary space — roughly 30,000 dimensions, one per BERT token. Each non-zero
dimension corresponds to a token ID; its value is a learned importance weight.

Two properties make SPLADE more powerful than raw BM25:

**Learned weighting** — weights come from a transformer fine-tuned for retrieval,
not from raw term counts. Tokens that carry more discriminative signal in context
receive higher weights.

**Query/document expansion** — the model assigns non-zero weight to terms that
are semantically related to the text but may not literally appear in it. A chunk
about "automobiles" might get non-zero weights on "car", "vehicle", "engine". This
bridges the vocabulary gap that kills exact keyword search: a query for "cars"
will still match a document that only says "automobile".

The result is dense-model-quality recall expressed in sparse form — efficient to
index in Qdrant's sparse vector store and fast to score without a full matrix
multiply.

---

## Vectors Per Chunk

**2 vectors per chunk**, stored as two named vector fields on a single Qdrant
point:

| Field | Type | Length | What it captures |
|-------|------|--------|-----------------|
| `dense` | `List[float]` | `DENSE_EMBEDDING_DIM` (1024 or 1536) | Semantic meaning — paraphrases, synonyms |
| `sparse` | `SparseVector` (indices + values) | Variable; ~100–300 non-zeros out of ~30k | Term-weighted keywords + learned expansion |

The sparse vector length varies per chunk. Longer, keyword-rich chunks typically
produce more non-zero entries than short ones. Qdrant stores only the non-zero
entries, so the storage cost scales with actual content density, not vocabulary
size.

---

## Files

| File | Role |
|------|------|
| `backend/app/services/document_processor.py` | Full ingestion implementation |
| `backend/app/core/config.py` | `DENSE_EMBEDDING_DIM`, `SPLADE_MODEL`, `FASTEMBED_CACHE_DIR`, batch sizes |
| `backend/app/services/chunk_record.py` | MySQL chunk upsert and deduplication helpers |

---

## Related Docs

- [search-implementation.md](search-implementation.md) — how the stored vectors are
  queried at retrieval time
- [architecture.md](architecture.md) — full system overview
