# Document Ingestion Pipeline

## Overview

When a document is uploaded it goes through a 7-step pipeline that ends with
each text chunk stored in two places: Qdrant (for vector search) and MySQL (for
full-text search). Every chunk produces exactly **2 vectors** — one dense, one
sparse — stored as named vector fields on a single Qdrant point.

Document parsing is handled by **[MarkItDown](https://github.com/microsoft/markitdown)**
(Microsoft), which converts every supported file type into clean, consistent
Markdown before chunking. This normalises wildly different input formats into a
single representation, so the rest of the pipeline is format-agnostic.

---

## Supported File Formats

All conversion is performed by `markitdown[all]` + `markitdown-ocr`.

| Category | Extensions |
|----------|-----------|
| Documents | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls` |
| Text / Markup | `.txt`, `.md`, `.html`, `.htm`, `.mhtml` |
| Data formats | `.csv`, `.json`, `.xml` |
| Email | `.msg`, `.eml` |
| Books | `.epub` |
| Images (OCR) | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff` |
| Archives | `.zip` (contents processed recursively) |

Files with unsupported extensions are rejected at the API layer with HTTP 400
before any I/O occurs.

---

## Steps

### 1. Upload & Validate

The file is received by `POST /api/knowledge-base/{kb_id}/documents/upload`.

- Extension is checked against `SUPPORTED_EXTENSIONS` — unsupported types return
  HTTP 400 immediately.
- File content is SHA-256 hashed. If an identical file (same name + same hash)
  already exists in the knowledge base it is returned as `status: "exists"` and
  skipped — no re-processing.
- The file is saved to `uploads/user_{uid}/kb_{kb_id}/temp/{filename}` and a
  `DocumentUpload` record is created in MySQL with `status: "pending"`.

### 2. Convert to Markdown

`_convert_to_markdown(abs_path, file_name)` in `document_processor.py`.

MarkItDown is initialised once as a lazy singleton. When `OPENAI_VISION_MODEL` is
set in `.env`, the `markitdown-ocr` plugin is activated and a `SyncOpenAI` client
is passed as `llm_client`. The vision model then automatically OCRs any images
embedded in uploaded documents — scanned PDF pages, photos in DOCX/PPTX/XLSX, or
standalone image uploads. When `OPENAI_VISION_MODEL` is unset, MarkItDown is
initialised without a client and OCR is silently skipped (identical to the
previous behaviour).

Think-block traces (`<think>...</think>`) emitted by reasoning vision models are
stripped from the converted text before it is chunked. Both complete blocks and
truncated unclosed prefixes are handled.

MarkItDown applies the appropriate converter for each file type:

| Input type | What MarkItDown does |
|------------|---------------------|
| PDF | Extracts text and structure, preserves headings |
| Word (docx/doc) | Extracts paragraphs, headings, tables |
| PowerPoint (pptx/ppt) | Extracts slide text, titles, speaker notes |
| Excel (xlsx/xls) | Converts sheets to Markdown tables |
| HTML / MHTML | Strips tags, preserves document structure |
| CSV | Converts to Markdown table |
| JSON | Pretty-prints as fenced code block |
| XML | Preserves structure as formatted text |
| Email (msg/eml) | Extracts headers, body, and attachments |
| EPUB | Extracts chapter text |
| Images | OCR via `markitdown-ocr` using `OPENAI_VISION_MODEL`; skipped when model is unset |
| ZIP | Recursively converts all contained files |

**Fallback:** if MarkItDown raises any exception, `_convert_to_markdown` falls
back to reading the raw file as UTF-8 (with `errors="replace"`). This ensures
the pipeline never hard-fails on a conversion error — degraded output is always
preferred over a processing failure.

**Think-trace stripping:** after conversion the output is passed through two
regex passes to strip `<think>...</think>` blocks that reasoning vision models
may emit during OCR. This keeps thinking noise out of the chunk index.

The resulting Markdown string is wrapped in a single `LangchainDocument` with
`metadata={"source": file_name}` before chunking.

### 3. Chunk

`RecursiveCharacterTextSplitter` with configurable `chunk_size` and
`chunk_overlap` (set via `.env`).

Default values: `CHUNK_SIZE=1500`, `OVERLAP_PERCENTAGE=0.20` → 300-char overlap.

The splitter tries to break on paragraph boundaries first (`\n\n`), then line
breaks (`\n`), then spaces, falling back to raw characters only as a last
resort. The overlap repeats the tail of each chunk at the head of the next,
preserving sentence context across boundaries.

> **Note:** Because MarkItDown produces Markdown output, chunking naturally
> respects structural boundaries like headings and paragraph breaks that are
> present in the converted text.

> **Warning:** `CHUNK_SIZE` and `OVERLAP_PERCENTAGE` must stay consistent
> across all documents in a knowledge base. If you change them after documents
> have been ingested, delete and re-upload existing documents to re-index with
> the new settings.

### 4. Deduplicate

SHA-256 of `(chunk_text + metadata_string)` is computed for every chunk. If the
hash already exists in MySQL for that file, the chunk is skipped. Only new or
changed chunks proceed to embedding. This makes re-processing an updated
document cheap — unchanged sections are not re-embedded or re-indexed.

### 5. Dense Embedding

Chunk texts are sent to the configured OpenAI-compatible embedding API
(`OPENAI_EMBEDDINGS_MODEL`) in batches of 32.

Each chunk produces one `List[float]` of length `DENSE_EMBEDDING_DIM` (default
1024 for local models such as `qwen3-embedding-0.6b`; 1536 for
`text-embedding-3-small` / `text-embedding-ada-002`).

This is a semantic embedding — the entire chunk is compressed into a single
point in a continuous vector space where similar meanings cluster together
regardless of exact wording.

### 6. Sparse Embedding (SPLADE)

The same chunk texts are passed through FastEmbed's `SparseTextEmbedding`
(`prithivida/Splade_PP_en_v1` by default). Each chunk produces a
`SparseVector`: two arrays — `indices` and `values` — where the vast majority
of entries are zero and only a small subset carry non-zero weights.

See [What sparse vectors represent](#what-sparse-vectors-represent) below.

### 7. Upsert to Qdrant

Each chunk becomes one Qdrant `PointStruct`:

```
PointStruct(
    id      = UUID derived deterministically from SHA-256(chunk_id),
    vector  = {
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

### 8. Store in MySQL

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

**Learned weighting** — weights come from a transformer fine-tuned for
retrieval, not from raw term counts. Tokens that carry more discriminative
signal in context receive higher weights.

**Query/document expansion** — the model assigns non-zero weight to terms that
are semantically related to the text but may not literally appear in it. A chunk
about "automobiles" might get non-zero weights on "car", "vehicle", "engine".
This bridges the vocabulary gap that kills exact keyword search: a query for
"cars" will still match a document that only says "automobile".

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

## Why MarkItDown?

Before MarkItDown, each file type required a separate LangChain loader
(`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`). Adding a new format meant
adding a new dependency and a new code path. The limitations were:

- Only 4 formats supported: PDF, DOCX, MD, TXT
- PDF extraction produced raw text with no structural awareness
- Spreadsheets, presentations, images, and emails were unsupported
- Each loader had different metadata conventions

MarkItDown replaces all loaders with a single call. Benefits:

- **28 formats** from one library, maintained by Microsoft
- **Consistent Markdown output** — headings, tables, and lists are preserved
  as Markdown syntax, which the `RecursiveCharacterTextSplitter` can use as
  natural break points
- **OCR for images** via `markitdown-ocr` — scanned documents and image files
  are readable
- **Graceful fallback** — conversion failures degrade to raw text rather than
  failing the whole task
- **No schema changes** — `chunk_text LONGTEXT` in MySQL and Qdrant's
  text-agnostic vector storage handle Markdown output without modification

---

## Failure Handling & Atomicity

The background processing function (`process_document_background`) is designed
so that any failure leaves the system in a fully clean state — no orphaned
records, no partial vectors, no stranded files.

### What happens on failure

| Stage | What fails | Outcome |
|---|---|---|
| Parse (step 1) | `_convert_to_markdown` returns empty string | Raised as `ValueError` — file still at temp path, nothing committed |
| Parse (step 1) | Chunking produces 0 chunks | Raised as `ValueError` — same as above |
| Move to permanent storage (step 4) | `move_file` raises | File still at temp path; temp path is deleted in cleanup |
| Document record commit (step 5) | DB error | Rolled back; file at permanent path is deleted |
| Qdrant upsert (step 7) | Network / API error | Session rolled back — chunk records (never committed) are discarded; Document record deleted; permanent file deleted |

In every case the task is marked `status="failed"` with the error message written
to `task.error_message`, which the UI surfaces to the user.

### Why chunks are committed after Qdrant, not during the loop

The old implementation committed chunks every 100 rows inside the loop. If
Qdrant upsert then failed, MySQL had committed chunks with no matching vectors —
the two stores would be permanently out of sync.

Chunks are now added to the SQLAlchemy session but **not committed** until the
Qdrant upsert succeeds (step 7). A single `db.commit()` at step 8 atomically
persists chunks + task status + upload status together. A failure anywhere
before that point triggers `db.rollback()`, discarding all pending chunk
records cleanly.

### File cleanup always targets the current location

A `permanent_path` variable tracks whether the file has been moved yet. The
cleanup code deletes `permanent_path` if the move succeeded, or `temp_path` if
it had not — so the file is always deleted from wherever it currently is, and
never left orphaned on disk.

### Empty-parse detection

`_convert_to_markdown` never raises — it degrades to raw UTF-8 text on
conversion failure, and returns `""` only if that also fails. To prevent a
zero-text document from silently ingesting as 0 chunks marked "completed", the
pipeline now explicitly raises `ValueError` if:

- the converted text is empty or whitespace-only
- the splitter produces 0 chunks from the converted text

Both errors surface in `task.error_message` and the document is not stored.

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_SIZE` | Target chunk size in characters | `1500` |
| `OVERLAP_PERCENTAGE` | Fraction of `CHUNK_SIZE` to overlap (0.0–1.0) | `0.20` |
| `DENSE_EMBEDDING_DIM` | Output dimension of embedding model | `1024` |
| `OPENAI_EMBEDDINGS_MODEL` | Embedding model name | local model |
| `OPENAI_VISION_MODEL` | Multimodal model for OCR of embedded images. Leave unset to disable. | unset |
| `OPENAI_VISION_API_BASE` | Base URL for the vision model. Falls back to `OPENAI_API_BASE`. | unset |
| `SPLADE_MODEL` | FastEmbed SPLADE model name | `prithivida/Splade_PP_en_v1` |
| `FASTEMBED_CACHE_DIR` | Where FastEmbed caches ONNX models | `./assets/fastembed` |

---

## Files

| File | Role |
|------|------|
| `backend/app/services/document_processor.py` | Full ingestion implementation including `_convert_to_markdown()` |
| `backend/app/core/config.py` | `DENSE_EMBEDDING_DIM`, `SPLADE_MODEL`, `FASTEMBED_CACHE_DIR`, batch sizes |
| `backend/app/services/chunk_record.py` | MySQL chunk upsert and deduplication helpers |
| `backend/requirements.txt` | `markitdown[all]` and `markitdown-ocr` dependencies |

---

## Related Docs

- [search-implementation.md](search-implementation.md) — how the stored vectors
  are queried at retrieval time
- [architecture.md](architecture.md) — full system overview
- [chunking.md](chunking.md) — chunking strategy detail
