<div align="center">
  <img src="https://raw.githubusercontent.com/rag-web-ui/rag-web-ui/main/docs/images/github-cover-new.png" alt="RAG Web UI">
  <br />
  <p>
    <strong>Knowledge Base Management with Retrieval-Augmented Generation</strong>
  </p>
  <p>
    <a href="https://github.com/rag-web-ui/rag-web-ui/blob/main/LICENSE"><img src="https://img.shields.io/github/license/rag-web-ui/rag-web-ui" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/node-%3E%3D18-green.svg" alt="Node"></a>
    <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  </p>
</div>

## Introduction

RAG Web UI is a self-hosted knowledge base Q&A system. Upload your documents, then chat with them. Uses any **OpenAI-compatible API** for LLM and embeddings — works with OpenAI, LM Studio, or any local model server.

Retrieval uses **3-leg hybrid search**: dense vector (Qdrant cosine), sparse vector (SPLADE via Qdrant), and exact keyword (MySQL full-text), combined with configurable weights.

> **Based on:** This is an opinionated, slimmed-down fork of [rag-web-ui/rag-web-ui](https://github.com/rag-web-ui/rag-web-ui). All credit for the original design and implementation goes to the original authors. The goal of this fork is to serve as a learning resource for understanding the RAG pipeline end-to-end — keeping minimal dependencies, removing abstraction layers, and adding visibility into individual RAG components (retrieval legs, reranking, prompt construction, token flow).


## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Vercel AI SDK |
| Backend | Python FastAPI, LangChain, SQLAlchemy |
| Vector DB | Qdrant (dense + sparse vectors) |
| Sparse Embeddings | SPLADE via FastEmbed (CPU, local) |
| File Storage | Local Folder mapped to Docker as Volume |
| Database | MySQL 8 |

## Quick Start

**Prerequisites:** Docker & Docker Compose v2+

```bash
git clone https://github.com/tangowhisky-dev/rag-web-ui.git
cd rag-web-ui
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, OPENAI_EMBEDDINGS_MODEL, DENSE_EMBEDDING_DIM
docker compose up -d --build
```

Open **http://localhost:3000** — register an account and start uploading documents.

> **First run note:** The SPLADE model (~500 MB) is downloaded on first document ingestion if not pre-cached. To pre-download it into `./assets/fastembed/` see [Pre-downloading the SPLADE model](#pre-downloading-the-splade-model).

## Configuration

Copy `.env.example` to `.env` and set these values:

### LLM & Embeddings

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` | API key for your LLM provider | `sk-...` or `lmstudio` |
| `OPENAI_API_BASE` | Base URL of OpenAI-compatible API | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Chat model name | `gpt-4o` |
| `OPENAI_EMBEDDINGS_MODEL` | Embedding model name | `text-embedding-3-small` |
| `DENSE_EMBEDDING_DIM` | Output dimension of the embedding model | `1536` for OpenAI, `1024` for qwen3-0.6b |

**Using a local model server (e.g. LM Studio):**
```env
OPENAI_API_KEY=lmstudio
OPENAI_API_BASE=http://host.docker.internal:1234/v1
OPENAI_MODEL=your-model-name
OPENAI_EMBEDDINGS_MODEL=your-embedding-model-name
DENSE_EMBEDDING_DIM=1024
```

### Retrieval

| Variable | Description | Default |
|---|---|---|
| `RETRIEVAL_TOP_K` | Number of chunks returned per query | `6` |
| `HYBRID_DENSE_WEIGHT` | Weight for dense vector leg | `0.5` |
| `HYBRID_QDRANT_SPARSE_WEIGHT` | Weight for SPLADE sparse leg | `0.3` |
| `HYBRID_EXACT_WEIGHT` | Weight for MySQL full-text leg | `0.2` |
| `RETRIEVAL_DENSE_ENABLED` | Enable/disable dense retrieval leg | `true` |
| `RETRIEVAL_QDRANT_SPARSE_ENABLED` | Enable/disable SPLADE sparse leg | `true` |
| `RETRIEVAL_EXACT_ENABLED` | Enable/disable MySQL FTS leg | `true` |

### Vector DB (Qdrant)

| Variable | Description | Default |
|---|---|---|
| `QDRANT_HOST` | Qdrant service hostname | `qdrant` |
| `QDRANT_PORT` | Qdrant HTTP port | `6333` |
| `QDRANT_GRPC_PORT` | Qdrant gRPC port | `6334` |

### SPLADE Sparse Embedder

| Variable | Description | Default |
|---|---|---|
| `SPLADE_MODEL` | FastEmbed model name | `prithivida/Splade_PP_en_v1` |
| `FASTEMBED_CACHE_DIR` | Where to persist downloaded model | `./assets/fastembed` |

### Storage & Auth

| Variable | Description | Example |
|---|---|---|
| `MYSQL_SERVER` | MySQL host | `db` |
| `MYSQL_USER` | MySQL username | `ragwebui` |
| `MYSQL_PASSWORD` | MySQL password | `ragwebui` |
| `MYSQL_DATABASE` | MySQL database name | `ragwebui` |
| `SECRET_KEY` | JWT signing secret (auto-generated in dev if left as placeholder) | random string |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT lifetime | `10080` (7 days) |

## Pre-downloading the SPLADE model

To avoid a ~500 MB download on first ingestion, pre-download the model into `./assets/fastembed/`:

```bash
pip install fastembed
python download_assets.py
```

The directory is bind-mounted into the container, so the model will be available immediately on next start.

## Admin & Developer Tools

The following web UIs are available when running locally:

| Tool | URL | Purpose |
|---|---|---|
| **RAG Web UI** | http://localhost:3000 | Main application |
| **Backend API Docs** | http://localhost:8000/redoc | OpenAPI reference |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Browse vector collections, inspect points, run queries |
| **File Storage** | Local Folder mapped to Docker as Volume | Browse uploaded files |
| **Adminer** | http://localhost:8080 | MySQL web GUI (dev only — see below) |

### Adminer (MySQL web GUI)

Adminer is included in `docker-compose.dev.yml`. It's a lightweight single-container MySQL browser.

Start it:
```bash
docker compose -f docker-compose.dev.yml up -d adminer
```

Then open http://localhost:8080 and log in:

| Field | Value |
|---|---|
| System | MySQL |
| Server | `db` |
| Username | `ragwebui` |
| Password | `ragwebui` |
| Database | `ragwebui` |

### Qdrant Dashboard

Open http://localhost:6333/dashboard — no login required. From here you can:
- Browse collections and their vector counts
- Inspect individual points and their payloads
- Run search queries manually
- View collection configuration (vector dimensions, distance metric)

## Development

Hot reload for both frontend and backend:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

Services:

| Service | URL |
|---|---|
| Frontend (Next.js dev) | http://localhost:3000 |
| Backend (uvicorn reload) | http://localhost:8000 |
| API Docs | http://localhost:8000/redoc |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Adminer (MySQL) | http://localhost:8080 |

**Stop without losing state:**
```bash
docker compose -f docker-compose.dev.yml stop
```

**Start again (no rebuild):**
```bash
docker compose -f docker-compose.dev.yml start
```

**Useful commands:**
```bash
# Restart a single service
docker compose -f docker-compose.dev.yml restart backend

# Check status
docker compose -f docker-compose.dev.yml ps

# Tail logs
docker compose -f docker-compose.dev.yml logs -f backend
docker compose -f docker-compose.dev.yml logs -f frontend

# Rebuild only when requirements.txt or Dockerfile changes
docker compose -f docker-compose.dev.yml up -d --build backend
```

## Features

- Upload PDF, DOCX, Markdown, and plain text files
- Automatic chunking, embedding, and incremental updates
- 3-leg hybrid search: dense vector + SPLADE sparse + MySQL full-text
- Multi-turn chat with source citations
- Streaming responses with think-block collapsing for reasoning models
- Retrieval quality testing UI
- Route protection: unauthenticated users are redirected to `/login`
- JWT invalidated on container restart (ephemeral secret key in dev)

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).

## License

[Apache-2.0](LICENSE)

---

<div align="center">If this project helps you, please give it a ⭐️</div>
