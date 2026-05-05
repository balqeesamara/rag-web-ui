# Plan: Automatic OCR for Embedded Images in Documents

## Goal

Extract text from images embedded in uploaded documents — scanned PDFs, images
inside DOCX, PPTX, XLSX — automatically, with no user involvement.

---

## Current State

`markitdown-ocr` is already in `requirements.txt`. However the singleton is
initialised without `llm_client` or `enable_plugins=True`:

```python
# document_processor.py, _get_markitdown()
_markitdown = MarkItDown()
```

Without `llm_client`, the plugin loads but silently skips all OCR — it's a
no-op. The dependency is there; we just haven't wired it up.

---

## Problem Analysis

### What documents currently fail or partially succeed

| Document type | Current behaviour | Why |
|---|---|---|
| Scanned PDF (image-only pages) | Empty or near-empty chunks | No text layer; MarkItDown extracts nothing without OCR |
| DOCX with embedded photos / diagrams | Image silently dropped | Plugin loaded but `llm_client` is None |
| PPTX where slides are rasterised images | Empty slides | Same |
| XLSX with chart images / embedded screenshots | Images dropped | Same |
| Mixed PDF (text pages + scanned pages) | Text pages OK; scanned pages empty | Page-level fallback only triggers when llm_client present |

### Why automatic detection is the right approach

The user uploading a "quarterly report.pdf" has no reliable way to know whether
it has a text layer. Many PDFs look identical visually whether they are born-
digital or scanned. Asking the user to tick a checkbox is friction that will
often be wrong. The pipeline should handle it transparently.

---

## Proposed Approach

### Core idea

Pass an `AsyncOpenAI`-compatible `llm_client` into `MarkItDown` at singleton
initialisation time. `markitdown-ocr` at priority -1.0 (runs before built-ins)
will then:
1. Extract embedded images from PDF / DOCX / PPTX / XLSX
2. Call the LLM vision endpoint for each image
3. Insert the returned text inline, preserving document structure

For pages / sections that already have a text layer the OCR converter detects
this and falls through to the standard built-in converter. For fully scanned
pages it performs full-page OCR. This is the automatic detection that eliminates
the need for user input.

### LLM client to use

The app already has `OPENAI_API_KEY` + `OPENAI_API_BASE` + `OPENAI_MODEL`
configured. Whether the model supports vision determines whether image OCR works
at all. We need a separate config variable for the vision model because:

- The chat model may be a text-only model (e.g. `qwen2.5-7b`)
- OCR requires a multimodal model (e.g. `gpt-4o`, `llava`, `qwen-vl`)
- These may be served at different endpoints

The variable should be optional. If unset, OCR is disabled and behaviour is
identical to today.

---

## Step-by-Step Plan

### Step 1 — Add config variables

In `backend/app/core/config.py`, add to the `Settings` class:

```python
VISION_API_BASE: Optional[str] = None    # defaults to OPENAI_API_BASE if unset
VISION_MODEL: Optional[str] = None       # e.g. "gpt-4o" or "llava:latest"
```

If `VISION_MODEL` is None → OCR disabled, plugin skipped.
If `VISION_MODEL` is set but `VISION_API_BASE` is not → fall back to
`OPENAI_API_BASE` (both models served from same server, common for local setups).

Add both to `.env.example` with comments explaining when to set them.

### Step 2 — Build the MarkItDown singleton with llm_client

In `document_processor.py`, change `_get_markitdown()`:

```python
def _get_markitdown() -> MarkItDown:
    global _markitdown
    if _markitdown is None:
        vision_model = settings.VISION_MODEL
        if vision_model:
            from openai import AsyncOpenAI
            vision_base = settings.VISION_API_BASE or settings.OPENAI_API_BASE
            client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=vision_base,
            )
            _markitdown = MarkItDown(
                enable_plugins=True,
                llm_client=client,
                llm_model=vision_model,
            )
            logger.info("[markitdown] OCR enabled — model=%s base=%s", vision_model, vision_base)
        else:
            _markitdown = MarkItDown()
            logger.info("[markitdown] OCR disabled — VISION_MODEL not set")
    return _markitdown
```

No changes to `_convert_to_markdown()` or downstream pipeline. Everything else
stays the same.

### Step 3 — Invalidate the singleton on config reload (optional / dev only)

The singleton is module-level. In production this is fine — config is stable.
If there is a future need to reload config at runtime, add a
`reset_markitdown()` helper that sets `_markitdown = None`. Not needed now.

### Step 4 — Update .env.example

```env
# Vision model for OCR of embedded images in PDFs, DOCX, PPTX, XLSX.
# Must be a multimodal (vision-capable) model.
# If not set, embedded images are silently skipped during ingestion.
# VISION_MODEL=gpt-4o
# VISION_API_BASE=http://host.docker.internal:1234/v1  # if different from OPENAI_API_BASE
```

### Step 5 — Update documentation

- `docs/ingestion-pipeline.md`: add subsection under Step 2 (Convert to
  Markdown) explaining OCR behaviour, the two config vars, and what happens when
  disabled
- `README.md` Configuration table: add `VISION_MODEL` and `VISION_API_BASE` rows
- `docs/architecture.md`: note in the ingestion diagram that OCR is conditional
  on VISION_MODEL

---

## Files to Change

| File | Change |
|------|--------|
| `backend/app/core/config.py` | Add `VISION_MODEL`, `VISION_API_BASE` |
| `backend/app/services/document_processor.py` | Update `_get_markitdown()` |
| `.env.example` | Add two commented vars with explanations |
| `docs/ingestion-pipeline.md` | Document OCR behaviour |
| `README.md` | Configuration table |
| `docs/architecture.md` | Ingestion diagram note |

No schema changes. No new dependencies (markitdown-ocr already in
requirements.txt). No frontend changes.

---

## Automatic Detection — How It Works Under the Hood

`markitdown-ocr` registers its converters at priority -1.0 (below 0.0 for
built-ins). When MarkItDown picks a converter for a file:

1. OCR converter is tried first
2. For PDF: it inspects each page — if a page has a text layer it extracts that
   directly; if a page has no text layer it sends the rendered page image to the
   LLM. This is the scanned-PDF detection. Zero user input required.
3. For DOCX/PPTX/XLSX: it extracts embedded image objects and calls the LLM for
   each. Text content is extracted normally by the built-in converter; OCR
   results are inserted inline.
4. If `llm_client` is None (VISION_MODEL not set), the OCR converter still
   loads but falls through immediately — no-op, zero performance impact.

---

## Edge Cases and Risks

| Risk | Mitigation |
|------|-----------|
| Vision model not multimodal (text-only) | LLM call fails; `markitdown-ocr` continues without that image's text — graceful, not fatal |
| Vision API latency adds to ingestion time | Ingestion is already async / background task; user doesn't wait. For heavily image-laden PDFs this will be noticeably slower but it's expected |
| Vision API token costs | OCR calls consume tokens on the vision model. If cost is a concern, `VISION_MODEL` can be pointed at a cheap local model (LLaVA, Qwen-VL via LM Studio) |
| LLM hallucinates during OCR | Risk is inherent to LLM-based OCR. Acceptable given the alternative is no text at all from scanned pages |
| Large PDFs with many scanned pages | Many parallel LLM calls. Consider rate limiting or sequential processing inside the plugin. Monitor first before adding limits |
| markitdown-ocr not yet on PyPI (saw 0.0.1a1 on pypi; full version may be local/private) | Verify exact package name and source before implementation. May need to install from GitHub |

---

## Open Questions

1. **Is the vision model the same as the chat model in this setup?** If the user
   runs `gpt-4o` for chat it's also multimodal, so `VISION_MODEL` could default
   to `OPENAI_MODEL`. But many users run a text-only chat model locally. Safest
   to keep it a separate explicit opt-in var.

2. **markitdown-ocr package availability**: PyPI shows `0.0.1a1` but the project
   `requirements.txt` already has `markitdown-ocr>=0.1.0`. Confirm whether it's
   from a private index, GitHub install, or a newer release before implementing.

3. **OCR prompt customisation**: The plugin supports a custom `llm_prompt`.
   Worth exposing as `VISION_OCR_PROMPT` env var for power users who need to
   tune extraction (e.g. preserve table structure, extract handwriting).
   Not required for the initial implementation.
