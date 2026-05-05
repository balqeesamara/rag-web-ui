# GraphRAG Implementation Plan

## Overview

Integrate Neo4j knowledge graph construction and retrieval into the existing
3-leg hybrid pipeline. Document ingestion automatically builds a knowledge
graph in Neo4j alongside the current Qdrant + MySQL indexing. When creating a
chat, the user can opt in to a 4th retrieval leg that traverses the graph.

The implementation is additive — every existing code path, data structure, and
API is preserved. GraphRAG is a new optional leg, not a replacement.

---

## Architecture After Integration

```
INGESTION (automatic, no user action needed)
  Upload → MarkItDown → Chunks
      │
      ├── [existing] Dense embed  → Qdrant  (dense vectors)
      ├── [existing] SPLADE embed → Qdrant  (sparse vectors)
      ├── [existing] Store text   → MySQL   (FTS)
      └── [NEW]      LLM extract  → Neo4j   (entity/relationship graph)
                     + shared UUIDs link Neo4j nodes to Qdrant points

RETRIEVAL (4-leg hybrid, graph leg optional per chat)
  Query → standalone question
      ├── Leg 1: Qdrant dense (existing)
      ├── Leg 2: Qdrant SPLADE sparse (existing)
      ├── Leg 3: MySQL FTS (existing)
      └── Leg 4: [NEW, opt-in] QdrantNeo4jRetriever
                  ↳ Qdrant finds semantic matches
                  ↳ Neo4j expands to entity subgraph (1-2 hops)
                  ↳ Returns graph-enriched context
      │
      └── Weighted RRF merge → top-K chunks → LLM → answer
```

---

## 1. New Docker Service: Neo4j

### docker-compose.yml / docker-compose.dev.yml

Add Neo4j Community Edition container:

```yaml
neo4j:
  image: neo4j:2026.04.0-community-trixie
  restart: always
  ports:
    - "7474:7474"   # HTTP browser UI
    - "7687:7687"   # Bolt driver port
  environment:
    - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-ragwebui_neo4j}
  volumes:
    - ./docker-data/neo4j/data:/data
    - ./docker-data/neo4j/logs:/logs
```

Notes:
- Community Edition is free, no license env var required.
- Bolt port 7687 is what the Python driver uses.
- Browser at http://localhost:7474 for dev inspection.

### Accessing Neo4j

**Neo4j Browser (web, no install needed)**

Navigate to http://localhost:7474 and log in:

```
Username: neo4j
Password: ragwebui_neo4j
```

**Neo4j Desktop (native app)**

Add a remote connection with these details:

```
Name:           RAG Web UI (or anything)
Connect URL:    neo4j://localhost:7687
Username:       neo4j
Password:       ragwebui_neo4j
```

Steps: open Neo4j Desktop → **Add** → **Remote connection** → fill in the fields above → **Connect**.

Use `neo4j://` as the protocol (the newer routing protocol). If it fails, fall back to `bolt://localhost:7687` — both point to the same Bolt port.

Note: port 7474 is the HTTP browser UI only. Port 7687 is the Bolt wire protocol used by Desktop, all drivers, and the Python backend.

### .env additions

```env
# Neo4j / GraphRAG
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ragwebui_neo4j

# Whether to auto-build the knowledge graph during ingestion.
# Set to false to skip graph extraction (saves LLM calls during development).
GRAPHRAG_ENABLED=true

# Model used for entity/relationship extraction during ingestion.
# Should be the same model as OPENAI_MODEL or a cheaper variant.
# Defaults to OPENAI_MODEL when unset.
GRAPHRAG_LLM_MODEL=

# Number of graph hops to traverse per retrieved seed node at query time.
GRAPHRAG_RETRIEVAL_HOPS=2

# RRF weight for the graph retrieval leg.
HYBRID_GRAPH_WEIGHT=0.3
```

---

## 2. Python Dependencies

```
# requirements.txt additions
neo4j>=5.26
neo4j-graphrag[qdrant]>=1.8.0
```

`neo4j-graphrag[qdrant]` installs:
- `neo4j-graphrag` (SimpleKGPipeline, VectorCypherRetriever, QdrantNeo4jRetriever)
- `qdrant-client` (already present — no version conflict expected)

---

## 3. Config Changes

`backend/app/core/config.py` — add:

```python
# Neo4j / GraphRAG
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "ragwebui_neo4j")
GRAPHRAG_ENABLED: bool = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"
GRAPHRAG_LLM_MODEL: str = os.getenv("GRAPHRAG_LLM_MODEL", "")  # falls back to OPENAI_MODEL
GRAPHRAG_RETRIEVAL_HOPS: int = int(os.getenv("GRAPHRAG_RETRIEVAL_HOPS", "2"))
HYBRID_GRAPH_WEIGHT: float = float(os.getenv("HYBRID_GRAPH_WEIGHT", "0.3"))
RETRIEVAL_GRAPH_ENABLED: bool = os.getenv("RETRIEVAL_GRAPH_ENABLED", "true").lower() == "true"

@property
def graphrag_model(self) -> str:
    return self.GRAPHRAG_LLM_MODEL or self.OPENAI_MODEL
```

---

## 4. MySQL Schema Change: Chat Table

Add `use_graph_rag` column to `chats` table so each chat remembers the user's
opt-in preference.

### Alembic migration

```python
# alembic/versions/xxxx_add_use_graph_rag_to_chats.py
def upgrade():
    op.add_column("chats", sa.Column("use_graph_rag", sa.Boolean(),
                                      nullable=False, server_default="0"))

def downgrade():
    op.drop_column("chats", "use_graph_rag")
```

### ORM model update

`backend/app/models/chat.py`:

```python
class Chat(Base, TimestampMixin):
    ...
    use_graph_rag = Column(Boolean, nullable=False, default=False)
```

### Schema update

`backend/app/schemas/chat.py` — add to `ChatCreate` and `ChatResponse`:

```python
class ChatCreate(BaseModel):
    title: str
    knowledge_base_ids: List[int] = []
    use_graph_rag: bool = False

class ChatResponse(BaseModel):
    ...
    use_graph_rag: bool
```

---

## 5. New Backend Service: graph_service.py

`backend/app/services/graph_service.py`

Responsibilities:
- Lazy Neo4j driver singleton
- Lazy graph builder singleton (SimpleKGPipeline)
- `build_graph_for_document()` — called from ingestion pipeline
- `graph_search()` — called from retrieval pipeline

### Key implementation points

```python
import neo4j
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document as LangchainDocument
from app.core.config import settings

_neo4j_driver = None
_kg_builder = None

def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
    return _neo4j_driver


def _get_kg_builder():
    global _kg_builder
    if _kg_builder is None:
        llm = OpenAILLM(
            model_name=settings.graphrag_model,
            model_params={"response_format": {"type": "json_object"}, "temperature": 0},
            # point at the same base URL as the rest of the app
            base_url=settings.OPENAI_API_BASE,
            api_key=settings.OPENAI_API_KEY,
        )
        embedder = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDINGS_MODEL,
            base_url=settings.OPENAI_API_BASE,
            api_key=settings.OPENAI_API_KEY,
        )
        _kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=_get_driver(),
            embedder=embedder,
            schema="EXTRACTED",   # auto-infer schema per document
            from_pdf=False,       # we pass pre-converted text, not raw files
        )
    return _kg_builder


async def build_graph_for_document(
    kb_id: int,
    document_id: int,
    file_name: str,
    chunks: list[str],          # plain text chunks from the existing splitter
    chunk_ids: list[str],       # SHA-256 chunk IDs (same as Qdrant point IDs)
) -> None:
    """
    Extract entity/relationship graph from document chunks and store in Neo4j.
    Also writes the shared chunk_id into each Neo4j Chunk node so that
    QdrantNeo4jRetriever can cross-reference Qdrant results to Neo4j nodes.

    Skips gracefully if Neo4j is unreachable or GRAPHRAG_ENABLED=false.
    """
    if not settings.GRAPHRAG_ENABLED:
        return
    builder = _get_kg_builder()
    full_text = "\n\n".join(chunks)
    # SimpleKGPipeline.run_async accepts raw text via `text=` parameter
    await builder.run_async(text=full_text, metadata={
        "kb_id": kb_id,
        "document_id": document_id,
        "file_name": file_name,
    })
    # After building, tag each Chunk node with its Qdrant chunk_id so the
    # QdrantNeo4jRetriever can bridge the two stores.
    _tag_chunks_with_ids(_get_driver(), kb_id, document_id, chunk_ids)


def _tag_chunks_with_ids(driver, kb_id, document_id, chunk_ids):
    """
    Write the qdrant_chunk_id property onto Neo4j Chunk nodes for this document,
    ordered by chunk_index. This bridges Qdrant and Neo4j at retrieval time.
    """
    with driver.session() as session:
        for i, cid in enumerate(chunk_ids):
            session.run(
                """
                MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
                SET c.qdrant_chunk_id = $cid
                """,
                doc_id=document_id, idx=i, cid=cid
            )


def graph_search(
    query_text: str,
    kb_ids: list[int],
    top_k: int,
) -> list[LangchainDocument]:
    """
    Run QdrantNeo4jRetriever to find graph-enriched results for the query.
    Returns a list of LangchainDocument objects compatible with the existing
    RRF merger in retrieval.py.
    """
    from qdrant_client import QdrantClient
    from neo4j_graphrag.retrievers import VectorCypherRetriever

    # Build retrieval query — 1-2 hop traversal from matched chunk nodes
    hops = settings.GRAPHRAG_RETRIEVAL_HOPS
    retrieval_query = f"""
    WITH node AS chunk
    MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[rel:!FROM_CHUNK*1..{hops}]-(neighbor)
    WITH chunk, collect(DISTINCT rel) AS rels
    RETURN chunk.text +
      '\\n\\n[Graph context]\\n' +
      reduce(s='', r in rels |
        s + startNode(r).name + ' -[' + type(r) + ']-> ' + endNode(r).name + '\\n'
      ) AS info
    """

    # One retriever per knowledge base collection, merge results
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as GREmbeddings
    embedder = GREmbeddings(
        model=settings.OPENAI_EMBEDDINGS_MODEL,
        base_url=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
    )
    qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    driver = _get_driver()

    results = []
    for kb_id in kb_ids:
        retriever = VectorCypherRetriever(
            driver=driver,
            index_name=f"kb_{kb_id}_text_embeddings",  # Neo4j vector index per KB
            embedder=embedder,
            retrieval_query=retrieval_query,
        )
        try:
            r = retriever.search(query_text=query_text, top_k=top_k)
            for item in r.items:
                results.append(LangchainDocument(
                    page_content=item.content,
                    metadata={"kb_id": kb_id, "source": "graph"},
                ))
        except Exception as e:
            logging.getLogger(__name__).warning(
                "GraphRAG retrieval failed for kb_%s: %s", kb_id, e
            )
    return results
```

---

## 6. Neo4j Vector Index Creation

Neo4j needs a vector index on Chunk nodes per knowledge base to support
VectorCypherRetriever. Create it when a knowledge base is first used for
graph ingestion.

`graph_service.py` — add helper called from `build_graph_for_document`:

```python
def _ensure_neo4j_vector_index(driver, kb_id: int, dims: int) -> None:
    index_name = f"kb_{kb_id}_text_embeddings"
    with driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dims},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
```

Call this inside `build_graph_for_document` before running the pipeline.

---

## 7. Ingestion Pipeline Changes

`backend/app/services/document_processor.py` — in `process_document_background`:

After step 8 (chunks committed to MySQL), add:

```python
# ── Step 9: Build Neo4j knowledge graph (async, non-blocking) ────────────
if settings.GRAPHRAG_ENABLED:
    try:
        from app.services.graph_service import build_graph_for_document
        chunk_texts = [c.page_content for c in chunks]
        await build_graph_for_document(
            kb_id=kb_id,
            document_id=document.id,
            file_name=file_name,
            chunks=chunk_texts,
            chunk_ids=[p[0] for p in qdrant_payloads],
        )
        logger.info(f"Task {task_id}: Knowledge graph built in Neo4j")
    except Exception as e:
        # Graph build failure is non-fatal — log and continue.
        # The document is still fully searchable via the other 3 legs.
        logger.warning(
            f"Task {task_id}: Neo4j graph build failed (non-fatal): {e}"
        )
```

Graph build is placed after the atomic commit so a Neo4j failure never
triggers a DB rollback. The document remains fully indexed and searchable
via the other 3 legs even if graph extraction fails.

---

## 8. Retrieval Pipeline Changes

`backend/app/services/retrieval.py`:

```python
async def hybrid_search(query, kb_ids, db, use_graph_rag=False) -> List[LangchainDocument]:
    top_k = settings.RETRIEVAL_TOP_K
    pool  = top_k * 4

    dense         = _dense_search(...)          if enabled["dense"]          else {}
    qdrant_sparse = _qdrant_sparse_search(...)  if enabled["qdrant_sparse"]  else {}
    exact         = _exact_search(...)          if enabled["exact"]          else {}

    graph_docs = []
    if use_graph_rag and settings.RETRIEVAL_GRAPH_ENABLED:
        from app.services.graph_service import graph_search
        graph_docs = graph_search(query, kb_ids, pool)

    return _rrf_merge(dense, qdrant_sparse, exact, graph_docs, top_k)
```

`_rrf_merge` is extended to accept a 4th ranked list. Graph results get
`HYBRID_GRAPH_WEIGHT` in the RRF score formula, same pattern as existing legs.

`_Candidate` dataclass gains one new field: `graph_rank: int = -1`.

---

## 9. Chat Service Changes

`backend/app/services/chat_service.py` — pass `use_graph_rag` through to
`hybrid_search`:

```python
docs = await hybrid_search(
    query=standalone_question,
    kb_ids=knowledge_base_ids,
    db=db,
    use_graph_rag=chat.use_graph_rag,  # read from Chat ORM object
)
```

---

## 10. API Changes

### Chat creation endpoint

`backend/app/api/api_v1/chat.py` — `POST /api/chats`:

Accept `use_graph_rag: bool = False` in the request body and persist to the
`Chat` row.

```python
@router.post("/", response_model=ChatResponse)
def create_chat(chat_in: ChatCreate, ...):
    chat = Chat(
        title=chat_in.title,
        user_id=current_user.id,
        use_graph_rag=chat_in.use_graph_rag,  # new
    )
    ...
```

### Chat response

`ChatResponse` already returns the full `Chat` ORM object — adding the column
to the model is enough for it to appear in responses (verify `from_orm=True`
is set or use `model_validate`).

---

## 11. Frontend Changes

### New Chat creation dialog

`frontend/src/components/chat/new-chat-dialog.tsx` (or wherever the new-chat
form lives):

Add a checkbox below the knowledge base multi-select:

```tsx
<div className="flex items-center gap-2 mt-3">
  <Checkbox
    id="use-graph-rag"
    checked={useGraphRag}
    onCheckedChange={(v) => setUseGraphRag(!!v)}
  />
  <label htmlFor="use-graph-rag" className="text-sm text-muted-foreground cursor-pointer">
    Enable GraphRAG — deeper multi-hop graph retrieval (slower, richer context)
  </label>
</div>
```

Pass `use_graph_rag: useGraphRag` to the chat creation API call.

### Chat header / indicator

Show a small graph icon or badge in the chat header when `use_graph_rag` is
`true` on the loaded chat, so users know the graph leg is active.

---

## 12. Document Cleanup

When a knowledge base or document is deleted, Neo4j nodes must also be pruned
to prevent orphaned graph data.

`backend/app/api/api_v1/knowledge_base.py` — in the delete document handler:

```python
if settings.GRAPHRAG_ENABLED:
    from app.services.graph_service import delete_graph_for_document
    delete_graph_for_document(kb_id=kb_id, document_id=document.id)
```

`graph_service.py` — add:

```python
def delete_graph_for_document(kb_id: int, document_id: int) -> None:
    with _get_driver().session() as session:
        session.run("""
            MATCH (c:Chunk {document_id: $doc_id})
            DETACH DELETE c
        """, doc_id=document_id)
        session.run("""
            MATCH (n)
            WHERE NOT EXISTS { MATCH (n)-[:FROM_CHUNK]->() }
              AND NOT n:Chunk AND NOT n:Document
              AND n.kb_id = $kb_id
            DETACH DELETE n
        """, kb_id=kb_id)
```

This removes all Chunk nodes for the document and any entity nodes that no
longer have connections to any remaining chunks.

---

## 13. Environment Variable Reference (complete additions)

```env
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ragwebui_neo4j

# GraphRAG ingestion
# Set false to skip graph extraction during ingestion (saves LLM calls).
GRAPHRAG_ENABLED=true
# Model for entity/relationship extraction. Defaults to OPENAI_MODEL if unset.
# A smaller/cheaper model works well for extraction (e.g. gpt-4o-mini).
GRAPHRAG_LLM_MODEL=

# GraphRAG retrieval
RETRIEVAL_GRAPH_ENABLED=true
# How many relationship hops to traverse from seed nodes at query time.
GRAPHRAG_RETRIEVAL_HOPS=2
# RRF weight for the graph retrieval leg.
HYBRID_GRAPH_WEIGHT=0.3
```

---

## 14. Summary of File Changes

| File | Change |
|------|--------|
| `docker-compose.yml` / `docker-compose.dev.yml` | Add `neo4j` service |
| `.env` / `.env.example` | Add Neo4j + GraphRAG vars |
| `backend/requirements.txt` | Add `neo4j`, `neo4j-graphrag[qdrant]` |
| `backend/app/core/config.py` | Add Neo4j/GraphRAG settings |
| `backend/app/models/chat.py` | Add `use_graph_rag` column |
| `backend/app/schemas/chat.py` | Add `use_graph_rag` to Create/Response |
| `backend/app/services/graph_service.py` | **New file** — Neo4j driver, KG builder, ingestion, retrieval, cleanup |
| `backend/app/services/document_processor.py` | Call `build_graph_for_document` in step 9 (non-fatal) |
| `backend/app/services/retrieval.py` | Add graph leg to `hybrid_search`, extend RRF merge |
| `backend/app/services/chat_service.py` | Pass `use_graph_rag` to `hybrid_search` |
| `backend/app/api/api_v1/chat.py` | Accept + persist `use_graph_rag` on chat creation |
| `alembic/versions/xxxx_add_use_graph_rag.py` | **New migration** — add column |
| `frontend/src/components/chat/new-chat-dialog.tsx` | Add GraphRAG checkbox |
| `frontend/src/components/chat/chat-header.tsx` | Graph active indicator |
| `docs/graphrag-implementation-plan.md` | This document |

---

## 15. Implementation Order

1. **Neo4j container** — add to compose, confirm it boots, run `bolt://neo4j:7687` health check.
2. **Dependencies + config** — `requirements.txt`, `config.py`, `.env.example`.
3. **DB migration** — Alembic migration for `use_graph_rag`, verify column added.
4. **graph_service.py** — implement and unit-test `build_graph_for_document` and `graph_search` in isolation against a local Neo4j.
5. **Ingestion hook** — wire step 9 into `process_document_background`. Test with a small document. Confirm entities appear in Neo4j browser.
6. **Retrieval hook** — extend `hybrid_search` + `_rrf_merge`. Test with `use_graph_rag=True`.
7. **API + schema** — update chat creation endpoint and `ChatCreate`/`ChatResponse`.
8. **Frontend** — checkbox in new-chat dialog, indicator in chat header.
9. **Cleanup hook** — delete graph nodes on document deletion.
10. **End-to-end test** — ingest a document, create a graph-enabled chat, query it, confirm graph context appears in retrieved chunks.

---

## 16. Design Decisions and Rationale

**Graph build is non-fatal.** A failure in Neo4j extraction does not roll back
the document. The user gets a fully indexed document with 3-leg retrieval even
if the graph step fails. This avoids making ingestion dependent on LLM
availability for extraction.

**Graph leg is opt-in per chat.** Graph retrieval is slower (LLM extraction
at ingest, graph traversal at query) and adds cost. Users who don't need it
pay nothing. The `use_graph_rag` flag is stored on the Chat row so the choice
is permanent per conversation without per-message configuration.

**Schema auto-inferred (`"EXTRACTED"` mode).** This avoids maintaining a
domain-specific schema for general-purpose document ingestion. Neo4j GraphRAG
v1.8.0's `SchemaFromTextExtractor` infers an appropriate schema per document.
Users with a specific domain can supply explicit node/relationship types via
a future `GRAPHRAG_SCHEMA_PATH` config option pointing to a YAML file.

**Shared Qdrant collection, separate Neo4j index.** The Qdrant collection
`kb_{id}` is unchanged. Neo4j gets a named vector index `kb_{id}_text_embeddings`
on Chunk nodes. Both are scoped per knowledge base, consistent with the
existing `kb_{id}` naming convention.

**VectorCypherRetriever over QdrantNeo4jRetriever.** Both classes are
available in `neo4j-graphrag`. `VectorCypherRetriever` is preferable here
because it uses Neo4j's own vector index (seeded at ingest), returns a
graph-traversal-enriched text string natively compatible with the existing
`LangchainDocument` pipeline, and does not require a Qdrant → Neo4j ID
bridge. `QdrantNeo4jRetriever` would be a better fit if we needed to reuse
the existing Qdrant dense index as the seed — that can be added as a
follow-up optimisation.

---

## 17. Known Limitations and Follow-ups

- **LLM cost at ingest.** Entity extraction calls the LLM once per document
  chunk. For large documents this adds latency and API cost. Mitigate by
  pointing `GRAPHRAG_LLM_MODEL` at a cheap fast model (e.g. `gpt-4o-mini`).

- **Graph staleness on re-ingest.** Re-processing an existing document should
  delete the old graph nodes before rebuilding. The current plan calls
  `delete_graph_for_document` + rebuild. This should be wired into the
  `process_document` (incremental update) path, not just `process_document_background`.

- **Multi-KB graph queries.** The current graph_search loops over kb_ids and
  merges results. An optimisation would be to store `kb_id` as a Chunk node
  property and query all KBs in a single `WHERE c.kb_id IN $kb_ids` pass.
