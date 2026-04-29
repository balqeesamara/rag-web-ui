# Troubleshooting Guide

## Admin & Developer Tools

Quick reference for the web UIs available in development:

| Tool | URL | Credentials |
|---|---|---|
| RAG Web UI | http://localhost:3000 | Your registered account |
| Backend API Docs | http://localhost:8000/redoc | — |
| Qdrant Dashboard | http://localhost:6333/dashboard | — (no login) |
| Adminer (MySQL) | http://localhost:8080 | Server: `db`, User: `ragwebui`, Pass: `ragwebui`, DB: `ragwebui` |

Start Adminer if it's not running:
```bash
docker compose -f docker-compose.dev.yml up -d adminer
```

## Database Issues

### Tables Not Found / Migration Errors

```bash
# Check MySQL container status
docker compose -f docker-compose.dev.yml ps db

# Check MySQL logs
docker compose -f docker-compose.dev.yml logs db

# Run migrations manually
docker exec -it ragwebui-backend-1 alembic upgrade head

# Check current migration state
docker exec -it ragwebui-backend-1 alembic current
docker exec -it ragwebui-backend-1 alembic history
```

### Connect to MySQL CLI

```bash
docker exec -it ragwebui-db-1 mysql -u ragwebui -pragwebui ragwebui
```

### Backup and Restore

```bash
# Backup
docker exec ragwebui-db-1 mysqldump -u ragwebui -pragwebui ragwebui > backup.sql

# Restore
docker exec -i ragwebui-db-1 mysql -u ragwebui -pragwebui ragwebui < backup.sql
```

## Qdrant Issues

### Inspect Collections

Open http://localhost:6333/dashboard — the Collections tab shows all collections, point counts, and vector config.

Or via CLI:
```bash
# List collections
curl http://localhost:6333/collections

# Inspect a specific collection
curl http://localhost:6333/collections/knowledge_1
```

### Collection Out of Sync

If vectors exist in Qdrant but documents appear missing, re-process the affected documents from the Knowledge Base UI. The ingestion pipeline is idempotent — re-processing overwrites existing points.

## Backend Issues

### Document Processing Failures

```bash
# Watch backend logs during ingestion
docker compose -f docker-compose.dev.yml logs -f backend
```

Common causes:
- **Embedding model mismatch**: `DENSE_EMBEDDING_DIM` must match the actual output dimension of `OPENAI_EMBEDDINGS_MODEL`. Check the model's spec.
- **SPLADE model not downloaded**: On first run, FastEmbed downloads `prithivida/Splade_PP_en_v1` (~500 MB). If the container restarts mid-download, delete the incomplete entry in `./assets/fastembed/` and retry.
- **LM Studio not running**: Verify `OPENAI_API_BASE` is reachable: `curl http://localhost:1234/v1/models`

### SPLADE Model Download Issues

If you see errors loading the sparse embedding model:
```bash
# Check what's in the cache directory
ls -la ./assets/fastembed/

# Remove any incomplete downloads and retry
rm -rf ./assets/fastembed/models--Qdrant--Splade_PP_en_v1

# Pre-download on host (avoids container restart issues)
pip install fastembed
python -c "
from fastembed import SparseTextEmbedding
SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1', cache_dir='$(pwd)/assets/fastembed')
print('Done.')
"
```

### API Endpoints Not Responding

```bash
# Check backend health
curl http://localhost:8000/api/health

# Check backend logs
docker compose -f docker-compose.dev.yml logs backend --tail=50
```

### Memory Issues

```bash
# Monitor container resource usage
docker stats
```

## Frontend Issues

### Authentication / Token Issues

The backend generates a new `SECRET_KEY` on every container restart (if `.env` has the placeholder `your-secret-key-here`). This invalidates all existing JWTs — simply log in again.

To persist JWT tokens across restarts, set a fixed `SECRET_KEY` in `.env`:
```env
SECRET_KEY=your-actual-random-secret-here
```

### Route Redirecting to Login Unexpectedly

The middleware checks for a `token` cookie (not localStorage). If you previously logged in before the middleware was added, clear cookies for `localhost:3000` and log in again.

## Container Issues

### Port Conflicts

```bash
# Check if a port is already in use
lsof -i :3000
lsof -i :8000
lsof -i :6333
lsof -i :8080
```

### Check All Service Status

```bash
docker compose -f docker-compose.dev.yml ps
```

### View All Logs

```bash
docker compose -f docker-compose.dev.yml logs --tail=50
```

## Complete Reset

Wipes all data (database, vectors, uploads) and starts fresh:

```bash
# Stop everything
docker compose -f docker-compose.dev.yml down

# Remove persisted data
rm -rf ./docker-data/mysql ./docker-data/qdrant ./uploads/*

# Rebuild and start
docker compose -f docker-compose.dev.yml up -d --build
```

> This does **not** remove `./assets/fastembed/` — the SPLADE model cache is preserved intentionally.

