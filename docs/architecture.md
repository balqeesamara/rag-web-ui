# RAG Web UI Architecture

## Overview

A self-hosted knowledge base Q&A system using 3-leg hybrid retrieval (dense vector + SPLADE sparse + exact match) with OpenAI-compatible LLMs.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RAG WEB UI ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃  FRONTEND   ┃   BACKEND    ┃  VECTOR DB  ┃   STORAGE    ┃
┃ (Next.js)   ┃ (FastAPI )   ┃ (Qdrant)    ┃ (Local FS)   ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┛

┌──────────────────────────────────────────────────────────────────────────────┐
│                               DATA FLOW                                      │
└──────────────────────────────────────────────────────────────────────────────┘

USER REQUEST → [Frontend:3000] → [Backend API:8000] → [Retrieval Engine] → [LLM] → RESPONSE

## Data Flow Components

### 1. Document Processing Pipeline

```
┏━━━━━━━━━━━━━━┓  ┌─────────────────────┐  ┌─────────────────────────────┐
┃ DOCUMENT     ┃  │   RETRIEV AL        ┃  │    HYBRID SCORING           │
┃ PROCESSING   ┃  │   PIPELI NE         ┃  │                             │
┗━━━━━━━━━━━━━━┛  └─────────────────────┘  └─────────────────────────────┘

       ┌──────────────────┐     ┌──────────────────────────────────────┐
       │                  │     │  LEG 1: DENSE VECTOR                 │
       │   Upload PDF/    │     │  - Qdrant cosine similarity          │
       │   DOCX/etc.      │◄────┤  - High precision semantic match     │
       │   Chunking       │     └──────────────────────────────────────┘
       │   Embedding      │                 ┌─────────────────────────────┐
       └──────────────────┘                 │                             │ 
                                            │  LEG 2: SPLADE SPARSE       │
                                            │  - FastEmbed CPU local      │
                                            │  - Term-frequency vectors   │
                                            │  - SPLADE model (~500MB)    │
                                            └─────────────────────────────┘

       ┌──────────────────┐     ┌─────────────────────────────┐
       │                  │     │                             │
       │  LEG 3: EXACT    │     │  LEG 4: QDRANT SPA          │
       │  KEYWORD MATCH   │     │  (alternative to SPLAE)     │
       └──────────────────┘     └─────────────────────────────┘

                                            ↓
                                         ┌──────────┐
                                         │ WEIGHTED │
                                         │ AGGREGATE│
                                         │ SCORE    │
                                         └──────────┘
```

### 2. Chat Session & Prompt Construction

```
┏━━━━━━━━━━━━━━┓  ┌─────────────────────┐  ┌─────────────────────────────┐
┃ CHAT SESSION ┃  │   RERANK ING        ┃  │   CONTEXT WINDOW            │
┗━━━━━━━━━━━━━━┛  └─────────────────────┘  └─────────────────────────────┘

┏━━━━━━━━━━━━━┓  ┌─────────────────────┐  ┌─────────────────────────────┐
┃ USER QUERY  ┃  │  QDRANT HYB RID     ┃  │   Retrieved chunks          │
┗━━━━━━━━━━━━━┛  │  SEARCH             ┃  │   - Context window for LLM  │
              └──┴─────────────────────┘  └─────────────────────────────┘
```

## Component Breakdown

### Backend Structure (`backend/`)

```
app/
├── main.py              # FastAPI entry point, route handlers
├── api/                  # API routes (chat, upload, retrieval)
│   ├── chat.py          # Chat endpoints with citations
│   └── rag.py           # Retrieval orchestration
├── core/                 # Configuration & utilities
├── db/                   # SQLAlchemy models (MySQL schema)
├── retrievers/           # Three retrieval legs:
│   ├── dense.py         # Qdrant vector search
│   ├── splade.py       # FastEmbed SPLADE sparse embedder
│   └── exact.py        # MySQL full-text search
├── storage/             # Local filesystem handling (uploads/downloads)
└── services/            # Chunking, embedding orchestration
```

### Docker Stack

- **Backend container** (`uvicorn` + `gunicorn` for production)
- **Qdrant** (vector database with hybrid support built-in)
- **MySQL 8.0** (payload storage + exact match queries)
- **Local filesystem** (file uploads/downloads via mounted volume)

### Configuration Flow

```env → Configuration Module → Retrieval Weights Configured →
HYBRID_DENSE_WEIGHT=0.5, HYBRID_QDRANT_SPARSE_WEIGHT=0.3,
HYBRID_EXACT_WEIGHT=0.2
```

## Memory & Session Management

- **Alembic migrations** for MySQL schema evolution
- **JWT tokens** with configurable expiration (default 7 days)
- **Ephemeral secrets** in dev mode; persistent in production

## Key Architectural Decisions

### 1. 3-Leg Hybrid Retrieval
No single modality dominates; weighted combination ensures robustness when one leg degrades (e.g., sparse vectors fail on long queries).

### 2. CPU-First Design
SPLADE runs locally via FastEmbed, avoiding GPU dependency for retrieval while maintaining high recall with term-frequency features.

### 3. Micro-Services via Docker
Each component isolated; easy scaling of Qdrant or MySQL without touching the Python backend.

## Technology Stack Summary

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Vercel AI SDK |
| Backend | Python FastAPI, LangChain, SQLAlchemy |
| Vector DB | Qdrant (dense + sparse vectors) |
| Sparse Embeddings | SPLADE via FastEmbed (CPU, local) |
| File Storage | Local filesystem (mounted volume) |
| Database | MySQL 8 |

## Quick Start

```bash
git clone https://github.com/tangowhisky-dev/rag-web-ui.git
cd rag-web-ui
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, etc.
docker compose up -d --build
```

Open **http://localhost:3000** and start using the knowledge base system.
