# RAG Web UI Architecture

## Overview

A self-hosted knowledge base Q&A system using 3-leg hybrid retrieval (dense vector + SPLADE sparse + MySQL full-text) with any OpenAI-compatible LLM.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RAG WEB UI ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃  FRONTEND   ┃   BACKEND    ┃  VECTOR DB  ┃   DATABASE   ┃
┃ (Next.js)   ┃ (FastAPI)    ┃ (Qdrant)    ┃ (MySQL 8)    ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┛

USER REQUEST → [Frontend:3000] → [Backend API:8000] → [Retrieval Engine] → [LLM] → RESPONSE
```

---

## Data Flow

### 1. Document Ingestion Pipeline

```
Upload (PDF / DOCX / MD / TXT)
    │
    ▼
document_processor.py
    ├── Load & parse (PyPDFLoader / Docx2txtLoader / TextLoader)
    ├── Chunk (RecursiveCharacterTextSplitter)
    ├── Embed chunks — async OpenAI-compatible API → dense vectors
    ├── Embed chunks — FastEmbed SPLADE → sparse vectors
    ├── Upsert to Qdrant (dense + sparse named vectors per collection kb_<id>)
    └── Store chunk text in MySQL document_chunks (for FTS + metadata)
```

### 2. Query / Chat Pipeline

```
User message
    │
    ▼
chat_service.py
    ├── Identity shortcut  (hardcoded response for "who are you?" etc.)
    ├── Sliding-window context (3 most-recent turn-pairs verbatim)
    ├── Rolling summary    (older turns folded into a summary via LLM)
    ├── Standalone question (context folded in → self-contained query)
    │
    ▼
retrieval.py — hybrid_search()
    ├── Leg 1: _dense_search()          → Qdrant cosine similarity (dense)
    ├── Leg 2: _qdrant_sparse_search()  → Qdrant SPLADE sparse vectors
    └── Leg 3: _exact_search()          → MySQL FULLTEXT NATURAL LANGUAGE MODE
                        │
                        ▼
                   _rrf_merge()  — weighted Reciprocal Rank Fusion
                        │
                        ▼
    top-K LangchainDocuments
    │
    ▼
chat_service.py
    ├── Build prompt with retrieved chunks as context
    ├── Stream response via AsyncOpenAI
    └── Strip <think>...</think> blocks (reasoning model support)
```

---

## Component Breakdown

### Backend Structure (`backend/`)

```
app/
├── main.py                    # FastAPI entry point, startup hooks
├── api/
│   └── api_v1/
│       ├── api.py             # Router registration
│       ├── auth.py            # JWT login / register
│       ├── chat.py            # Chat endpoints (create, stream, history)
│       ├── knowledge_base.py  # KB + document CRUD, upload, processing
│       └── api_keys.py        # API key management
├── core/
│   ├── config.py              # All settings (pydantic-settings, reads .env)
│   ├── security.py            # Password hashing, JWT creation/verification
│   └── storage.py             # Local filesystem helpers (save, move, delete)
├── db/
│   └── session.py             # SQLAlchemy engine + SessionLocal
├── models/
│   ├── user.py                # User ORM model
│   ├── knowledge.py           # KnowledgeBase, Document, DocumentChunk, ProcessingTask
│   └── chat.py                # Chat, Message ORM models
├── schemas/
│   ├── user.py                # Pydantic request/response schemas
│   ├── knowledge.py
│   ├── chat.py
│   └── token.py
├── services/
│   ├── document_processor.py  # Ingestion: parse → chunk → embed → index
│   ├── retrieval.py           # 3-leg hybrid search + RRF merge
│   ├── chat_service.py        # Conversation context, prompt, LLM streaming
│   └── chunk_record.py        # MySQL chunk upsert helpers
└── startup/                   # Startup utilities (Alembic auto-migrate etc.)

alembic/                       # Database migration scripts
```

### Frontend Structure (`frontend/`)

Next.js 14 app with TypeScript, Tailwind CSS, shadcn/ui, and the Vercel AI SDK for streaming.

```
src/
├── app/          # Next.js app router pages
├── components/   # React components (chat, KB management, retrieval test UI)
├── lib/          # API clients, utilities
├── styles/       # Global CSS
└── middleware.ts # Route protection (redirect unauthenticated to /login)
```

### Docker Stack

| Service | Image | Purpose |
|---------|-------|---------|
| `backend` | custom (Python FastAPI) | API server; uvicorn with hot-reload in dev |
| `frontend` | custom (Next.js) | Web UI; Next.js dev server or production build |
| `qdrant` | `qdrant/qdrant` | Vector database (dense + sparse collections) |
| `db` | `mysql:8` | Relational data + FULLTEXT chunk index |
| `adminer` | `adminer` | MySQL web GUI (dev compose only) |

---

## Key Architectural Decisions

### 1. 3-Leg Hybrid Retrieval
No single modality dominates all query types. Dense vectors handle paraphrases; SPLADE handles technical terms; MySQL FTS handles exact keywords and product codes. Weighted RRF fuses all three without requiring scores to be on the same scale.

### 2. CPU-First Sparse Embeddings
SPLADE runs locally via FastEmbed (ONNX, CPU-optimised), avoiding any GPU dependency for retrieval while maintaining learned sparse expansion beyond raw BM25.

### 3. Ingestion Always Indexes All Three Stores
Per-leg retrieval can be toggled via `.env` without re-ingestion. This makes A/B testing retrieval configurations cheap — flip a flag, test, flip back.

### 4. Sliding Window + Rolling Summary for Context
Rather than truncating history or stuffing the full chat into the prompt, older turns are summarised by the LLM and folded into a rolling summary. The 3 most-recent turn-pairs are kept verbatim.

### 5. OpenAI-Compatible API for LLM and Embeddings
Both the chat model and the embedding model are called through the same `openai` SDK pointed at `OPENAI_API_BASE`. Works with OpenAI, LM Studio, Ollama, or any compatible server.

---

## Memory & Session Management

- Alembic migrations for MySQL schema evolution (auto-applied on backend startup)
- JWT tokens with configurable expiration (default 7 days)
- Ephemeral `SECRET_KEY` in dev — tokens are invalidated on container restart

---

## Technology Stack Summary

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Vercel AI SDK |
| Backend | Python FastAPI, LangChain, SQLAlchemy, Alembic |
| Vector DB | Qdrant (dense + sparse named vectors) |
| Sparse Embeddings | SPLADE via FastEmbed (CPU, ONNX, local) |
| File Storage | Local filesystem (Docker volume mount) |
| Database | MySQL 8 (ORM data + FULLTEXT index) |
| Auth | JWT (python-jose, bcrypt) |

---

## Quick Start

```bash
git clone https://github.com/tangowhisky-dev/rag-web-ui.git
cd rag-web-ui
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, OPENAI_EMBEDDINGS_MODEL, DENSE_EMBEDDING_DIM
docker compose up -d --build
```

Open **http://localhost:3000**, register an account, and start uploading documents.

See [README.md](../README.md) for full configuration reference and development setup.
