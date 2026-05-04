# Chunking: Implementation and Rationale

## What Chunking Does and Why It Exists

Embedding models and LLMs both have fixed input size limits. A 50-page PDF cannot
be embedded as one unit — the embedding would lose detail, and even if it could be
stored, a cosine-similarity search against it would return the whole document for
every query. Chunking splits a document into smaller, topically focused pieces so
that:

1. Each piece fits within the embedding model's context window.
2. Retrieved chunks carry a tight, specific signal — the LLM receives only the
   paragraphs relevant to the query, not the whole document.
3. Source citations in the UI can point to a precise section, not just a file.

---

## The Splitter: `RecursiveCharacterTextSplitter`

The codebase uses LangChain's `RecursiveCharacterTextSplitter`. Parameters are
read from `.env` at startup via `settings` and applied consistently across all
ingestion paths.

| Parameter | Env var | Default | Derived from |
|-----------|---------|---------|--------------|
| `chunk_size` | `CHUNK_SIZE` | `1500` | Set directly |
| `chunk_overlap` | — | `CHUNK_SIZE * OVERLAP_PERCENTAGE` | Computed property |
| overlap ratio | `OVERLAP_PERCENTAGE` | `0.20` | 20% of chunk_size |

Both values are **character counts**, not token counts.

> **WARNING — do not change `CHUNK_SIZE` or `OVERLAP_PERCENTAGE` after documents
> have been ingested.** Doing so creates inconsistent chunk sizes across a
> knowledge base. If you change these values, delete and re-upload all existing
> documents to re-index them with the new settings.

### How the splitter works

It tries a prioritised list of separators in order, attempting the first one that
produces chunks within `chunk_size`. If splitting on the current separator still
leaves an oversized piece, it recurses down to the next separator in the list.
LangChain's default separator hierarchy is:

```
"\n\n"   ← paragraph break (preferred)
"\n"     ← single newline
" "      ← word boundary
""       ← individual characters (last resort)
```

The splitter always tries to break on a paragraph boundary first. Only if a single
paragraph exceeds `chunk_size` characters does it fall back to sentence-level
splits, then word-level, then character-level. In practice most chunks end at a
paragraph or sentence boundary.

### Why 1500 characters

The default was chosen based on the constraints of both embedding models in the
pipeline:

**Qwen3-embedding-0.6b** (dense leg) has a 32768-token context window. It can
comfortably embed chunks well beyond 1500 characters with no quality loss.

**SPLADE PP en v1** (sparse leg) is a BERT-derived model with a **512-token hard
limit** (510 usable after `[CLS]` and `[SEP]`). English text tokenizes at roughly
3–4.5 characters per BERT WordPiece token depending on vocabulary richness:

| Text type | Chars/token | Safe char ceiling |
|-----------|-------------|-------------------|
| Plain prose | ~4.5 | ~2295 |
| Mixed technical/prose | ~4.0 | ~2040 |
| Dense technical (jargon, codes) | ~3.0–3.5 | ~1530–1785 |

SPLADE silently truncates anything beyond ~512 tokens — it does not raise an
error. The sparse leg then only represents the first ~500 tokens of the chunk,
diverging from the dense vector which covers the full text.

**1500 characters sits safely within SPLADE's effective range for mixed
content**, including typical technical documentation. The upper safe bound for
general use is ~1800 characters; for dense technical corpora, stay at or below
~1500.

### Why 20% overlap (300 characters)

When a sentence straddles a chunk boundary, the context needed to interpret it may
sit in the previous chunk. A 300-character overlap (≈ 40–50 words at the default)
repeats the tail of one chunk at the head of the next, preventing orphaned
half-sentences that embed poorly and retrieve incorrectly.

The 20% ratio is deliberate: it is large enough to preserve boundary context
across most sentence structures while keeping index bloat predictable. At
`chunk_size=1500` and `overlap=300`, a 15,000-character document produces roughly
12 chunks instead of 10 — a 20% overhead in index size that consistently improves
boundary recall.

The overlap does not interact with the SPLADE token limit. Each chunk is at most
`chunk_size` characters; the overlap is shared text between adjacent chunks, not
additive on top of the limit.

---

## Configuration

All chunking parameters live in `.env` and are read via `backend/app/core/config.py`:

```env
# CHUNK_SIZE: target chunk size in characters.
# Keep <= 1800 for SPLADE (BERT 512-token limit, ~4 chars/token English).
# WARNING: do not change after ingesting documents — re-upload to re-index.
CHUNK_SIZE=1500

# OVERLAP_PERCENTAGE: fraction of CHUNK_SIZE repeated at chunk boundaries.
# 0.20 = 20% = 300 chars at CHUNK_SIZE=1500.
OVERLAP_PERCENTAGE=0.20
```

`chunk_overlap` is a computed property in `Settings` — it is never set directly:

```python
@property
def chunk_overlap(self) -> int:
    return int(self.CHUNK_SIZE * self.OVERLAP_PERCENTAGE)
```

### Choosing chunk size for your content

| Use case | Recommended `CHUNK_SIZE` | Notes |
|----------|--------------------------|-------|
| Plain prose / narrative | 1500–1800 | Near SPLADE ceiling but safe |
| Mixed technical docs | 1500 | Default; safe for all three retrieval legs |
| Dense technical (legal, medical, code) | 1200–1500 | Technical vocabulary tokenizes less efficiently |
| Mostly short snippets / FAQs | 500–800 | Smaller chunks = tighter retrieval signal |

If you push above ~1800 characters, reduce `HYBRID_QDRANT_SPARSE_WEIGHT` to
reflect that the sparse leg no longer covers the full chunk:

```env
CHUNK_SIZE=2500
OVERLAP_PERCENTAGE=0.20
HYBRID_DENSE_WEIGHT=0.7
HYBRID_QDRANT_SPARSE_WEIGHT=0.1
HYBRID_EXACT_WEIGHT=0.2
```

---

## Chunk Identity and Deduplication

Each chunk is assigned a **content-addressed ID** — a SHA-256 hash. The two code
paths compute it slightly differently:

**Background ingestion** (`process_document_background` — first-time upload):
```python
chunk_id = hashlib.sha256(
    f"{kb_id}:{file_name}:{chunk.page_content}".encode()
).hexdigest()
```
Same text in two different knowledge bases → two different IDs.

**Incremental update** (`process_document` — re-processing an existing file):
```python
chunk_hash = hashlib.sha256(
    (chunk.content + str(chunk.metadata)).encode()
).hexdigest()
```
The hash also covers metadata (page number, source path), so a chunk whose
position shifts — even if the text is identical — is treated as changed and
re-indexed.

In both paths, if the hash already exists in MySQL for that file the chunk is
skipped. Only genuinely new or changed chunks are embedded and upserted. Stale
chunks (present in the old index but absent from the current document) are deleted
from both MySQL and Qdrant. The index always reflects the current state of the
document.

---

## What Gets Stored per Chunk

After chunking, each chunk is written to two stores:

**MySQL `document_chunks` table**:
- `id` — SHA-256 chunk ID (primary key)
- `document_id`, `kb_id`, `file_name` — ownership
- `chunk_text` — raw text (covered by the FULLTEXT index)
- `chunk_index` — position in the original document (0-based)
- `chunk_metadata` — JSON blob with variable source metadata (page number, source
  path); fields already stored as proper columns are excluded
- `hash` — deduplication hash

**Qdrant point**:
- `id` — deterministic UUID derived via `uuid.uuid5` from the SHA-256 chunk ID
- `vector["dense"]` — float array of length `DENSE_EMBEDDING_DIM`
- `vector["sparse"]` — `SparseVector(indices, values)` from SPLADE
- `payload` — mirrors MySQL: `chunk_text`, `kb_id`, `document_id`, `file_name`,
  `chunk_index`, source metadata

The text is stored in both places intentionally: Qdrant needs it in the payload
to reconstruct `LangchainDocument` objects at retrieval time without a secondary
MySQL lookup; MySQL needs it as a proper column for the FULLTEXT index.

---

## User-Configurable Preview

The UI exposes a preview endpoint that lets a user see how a document will be
chunked before committing to ingestion:

```python
class PreviewRequest(BaseModel):
    document_ids: List[int]
    # When omitted the server uses CHUNK_SIZE / OVERLAP_PERCENTAGE from .env.
    # Explicitly passing values here overrides defaults for this preview only
    # and does NOT affect what is used during actual ingestion.
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
```

When `None` is passed the server resolves to `settings.CHUNK_SIZE` and
`settings.chunk_overlap`. The preview uses the exact same splitter and parameters
as ingestion — there is no gap between what you see in the preview and what ends
up in the index.

---

## Practical Implications

**Chunk size affects retrieval precision.** Smaller chunks retrieve tighter,
more specific passages but may lose surrounding context. Larger chunks carry more
context but the embedding averages over more content — weakening the similarity
signal for specific phrases buried within the chunk.

**Overlap is not free.** At `CHUNK_SIZE=1500` and `OVERLAP_PERCENTAGE=0.20`,
a 15,000-character document produces ~12 chunks instead of 10. At scale this
increases index size, embedding API cost, and SPLADE compute time proportionally.
The 20% default was chosen as a proven balance point.

**Character-based splitting is model-agnostic.** Token counts vary across
tokenizers and model families. Measuring in characters avoids a dependency on the
tokenizer of the currently configured embedding model. At ~4 chars/token for
English, `CHUNK_SIZE=1500` ≈ 300–375 tokens — comfortably within the 512-token
SPLADE limit.

**Chunk size is baked in at ingestion time.** Changing `.env` does not
retroactively re-chunk existing documents. Use the preview endpoint to verify
new settings against representative documents before changing `CHUNK_SIZE` and
re-ingesting.

---

## Files

| File | Role |
|------|------|
| `backend/app/core/config.py` | `CHUNK_SIZE`, `OVERLAP_PERCENTAGE`, `chunk_overlap` computed property |
| `backend/app/services/document_processor.py` | Splitter instantiation, chunk ID hashing, dedup logic |
| `backend/app/schemas/knowledge.py` | `PreviewRequest` — optional `chunk_size` / `chunk_overlap` override |
| `backend/app/api/api_v1/knowledge_base.py` | Preview endpoint; threads params into `process_document_background` |
| `.env` / `.env.example` | Runtime configuration |

---

## Related Docs

- [ingestion-pipeline.md](ingestion-pipeline.md) — full ingestion pipeline from
  upload to vector upsert
- [search-implementation.md](search-implementation.md) — how chunks are queried
  at retrieval time
