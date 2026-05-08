# GraphRAG Implementation

## Architecture

GraphRAG uses Neo4j exclusively for entity/relationship storage and post-retrieval
context enrichment. It does NOT perform vector search — Qdrant owns all vector
search. This matches the Qdrant+Neo4j reference architecture described at
https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/

```
Ingestion:
  document text
      │
      ├── Qdrant  ← dense + sparse vectors (existing 3-leg pipeline)
      ├── MySQL   ← chunk text for FTS (existing)
      └── Neo4j   ← Chunk nodes (keyed by document_id + chunk_index)
                     + Entity nodes + relationships (via SimpleKGPipeline)

Query time:
  user query
      │
      ▼
  3-leg RRF (Qdrant dense + sparse + MySQL FTS)
      │
      ▼  (document_id, chunk_index from Qdrant payload)
  Neo4j graph traversal — enrich_docs_with_graph()
      │ MATCH Chunk by (document_id, chunk_index)
      │ → FROM_CHUNK → Entity → relationships → neighbor Entity
      │ → append [Graph context] triples to chunk text
      ▼
  cross-encoder reranker
      │
      ▼
  LLM response generation
```

The cross-reference between Qdrant and Neo4j is the `(document_id, chunk_index)`
pair. Every Qdrant point payload contains these fields; every Neo4j Chunk node
is written with these same fields at ingest time. No UUID synchronisation is needed.

---

## Configuration

```env
# Enable Neo4j graph building at ingest and enrichment at query time
GRAPHRAG_ENABLED=true

# Neo4j connection
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Model used for entity/relationship extraction (falls back to OPENAI_MODEL)
GRAPHRAG_MODEL=

# Hop limit for entity traversal at query time
GRAPHRAG_RETRIEVAL_HOPS=2

# Per-retrieval enable/disable toggle (separate from GRAPHRAG_ENABLED)
RETRIEVAL_GRAPH_ENABLED=true
```

---

## Key files

| File | Role |
|------|------|
| backend/app/services/graph_service.py | All Neo4j logic: ingest, enrich, delete |
| backend/app/services/retrieval.py | Calls enrich_docs_with_graph() after RRF merge |
| backend/app/services/document_processor.py | Calls build_graph_for_document() at step 9 |
| backend/app/core/config.py | All GRAPHRAG_* settings |

---

## Ingestion — build_graph_for_document()

Called from `document_processor.py` step 9, after the chunk has been indexed in
Qdrant and MySQL.

1. Write `Chunk` nodes: `MERGE (c:Chunk {document_id: $doc_id, chunk_index: $idx})`
   with `qdrant_id`, `text`, `kb_id`, `file_name` properties.
2. Run `SimpleKGPipeline` with `embedder=None` to extract `Entity` nodes and
   relationships. `embedder=None` prevents it from calling `setNodeVectorProperty` —
   Neo4j stores no embeddings; Qdrant owns all vectors.
3. Entities are linked to Chunks via `FROM_CHUNK` edges by the pipeline.

Non-fatal by design. A Neo4j failure never blocks the core 3-leg RAG pipeline.

---

## Retrieval — enrich_docs_with_graph()

Called from `retrieval.py` after RRF merge, before cross-encoder reranking.

For each doc in the merged list:
1. Read `document_id` and `chunk_index` from `doc.metadata` (written there by
   the Qdrant payload parser at search time).
2. Cypher query:
   ```cypher
   MATCH (c:Chunk {document_id: $document_id, chunk_index: $chunk_index})
   OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(entity)
   OPTIONAL MATCH (entity)-[r]-(neighbor)
   WHERE type(r) <> 'FROM_CHUNK' AND type(r) <> 'FROM_DOCUMENT' AND type(r) <> 'NEXT_CHUNK'
   WITH c, coalesce(entity.name, ...) AS ename, type(r) AS rel, coalesce(neighbor.name, ...) AS nname
   WHERE ename IS NOT NULL AND nname IS NOT NULL
   WITH c, collect(DISTINCT [ename, rel, nname])[..40] AS triples
   RETURN triples
   ```
3. Append `[Graph context]\nA -[R]-> B\n...` to the chunk's text.
4. Store `_graph_triples` count in metadata for confidence scoring / logging.

Docs that fail graph lookup are returned unchanged. Non-fatal.

---

## Deletion

### delete_graph_for_document(kb_id, document_id)
- Deletes all `Chunk` nodes for the document (`DETACH DELETE`)
- Cleans up orphaned `__Entity__` nodes (no remaining `FROM_CHUNK` connections)
- NOT gated on `GRAPHRAG_ENABLED` — data may exist from a prior run

### delete_graph_for_kb(kb_id)
- Deletes all nodes with `kb_id` property (`DETACH DELETE`)
- Cleans up fully isolated `__Entity__` nodes
- NOT gated on `GRAPHRAG_ENABLED`

Both functions check `NEO4J_URI` and return early if unset, so they're safe
to call even in environments with no Neo4j configured.

---

## Design decisions

**Why not a 4th RRF leg?**
The original implementation ran Neo4j's own vector search as a 4th RRF leg.
This was redundant — Neo4j's vector index was doing the same job as Qdrant's
dense index, using a separate copy of the embeddings. It added latency, disk
usage, and ingest complexity with no retrieval quality benefit.

The correct use of Neo4j in this architecture is as a context enrichment layer,
not as a retrieval competitor. Qdrant finds what's relevant; Neo4j explains
how the relevant entities relate to each other.

**Why (document_id, chunk_index) as the cross-reference key?**
Qdrant point IDs are UUIDs assigned at ingest time. These are available in
`chunk_ids` passed to `build_graph_for_document()` and stored as `qdrant_id`
on the Chunk node (for tracing). However, `document_id` and `chunk_index` are
more stable and are already written into every Qdrant point payload, which
means graph lookup requires no schema changes to the Qdrant payload.

**Why embedder=None in SimpleKGPipeline?**
SimpleKGPipeline by default calls `setNodeVectorProperty` on Chunk and Entity
nodes. With `embedder=None`, this step is skipped — no embeddings are written
to Neo4j, eliminating the `List{}` error that occurred with neo4j_graphrag 1.16.0
when `embedding_properties` was empty.
