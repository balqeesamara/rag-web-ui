"""
Neo4j knowledge graph service.

Architecture
------------
Neo4j is used exclusively for entity/relationship storage and graph-context
enrichment. It is NOT used for vector search — Qdrant owns all vector search.

Ingestion (build_graph_for_document):
  1. Write Chunk nodes keyed by (document_id, chunk_index) — the same
     identifiers stored in every Qdrant point payload.
  2. Use SimpleKGPipeline to extract Entity nodes and relationships from the
     full document text. Entities are linked to Chunks via FROM_CHUNK edges.
     Embeddings are written to Chunk nodes by SimpleKGPipeline (required by
     the library) but are never queried — Qdrant owns all vector search.

Retrieval (enrich_docs_with_graph):
  Called after RRF merge (and before reranking) on the merged doc list.
  For each doc, looks up its Chunk node by (document_id, chunk_index), then
  traverses entity relationships outward. Appends a [Graph context] block of
  entity triples to the chunk text so the reranker and LLM see richer context.

  This matches the Qdrant+Neo4j reference architecture:
    vector search (Qdrant) → IDs → graph traversal (Neo4j) → enriched context

Deletion:
  delete_graph_for_document / delete_graph_for_kb — always run regardless of
  the GRAPHRAG_ENABLED flag so data is never orphaned when the flag is toggled.
"""

import logging
from typing import Optional

import neo4j
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_neo4j_driver: Optional[neo4j.Driver] = None


def _get_driver() -> neo4j.Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _neo4j_driver


def _make_kg_builder() -> SimpleKGPipeline:
    """Create a fresh SimpleKGPipeline per document (no shared state).

    SimpleKGPipeline requires an embedder — we provide a real one but do NOT
    create a Neo4j vector index.  The embeddings land on Chunk nodes but are
    never queried; Qdrant owns all vector search.
    """
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    llm = OpenAILLM(
        model_name=settings.graphrag_model,
        model_params={"temperature": 0},
        base_url=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
    )
    embedder = OpenAIEmbeddings(
        base_url=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_EMBEDDINGS_MODEL,
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=_get_driver(),
        embedder=embedder,
        from_pdf=False,
    )


# ── Ingest ─────────────────────────────────────────────────────────────────────

async def build_graph_for_document(
    kb_id: int,
    document_id: int,
    file_name: str,
    chunks: list[str],
    chunk_ids: list[str],
) -> None:
    """
    Extract entity/relationship graph from document text and store in Neo4j.

    Called from document_processor.py step 9. Non-fatal — callers catch all
    exceptions so a Neo4j failure never breaks the core retrieval pipeline.

    Chunk nodes are keyed by (document_id, chunk_index) which matches the
    payload stored in every Qdrant point — this is the cross-reference link
    used by enrich_docs_with_graph() at query time.
    """
    if not settings.GRAPHRAG_ENABLED:
        return

    driver = _get_driver()

    # 1. Write Chunk nodes with the identifiers Qdrant stores in its payload.
    #    qdrant_id is stored for debugging/tracing but not used for lookup
    #    (document_id + chunk_index is the stable key used at retrieval time).
    logger.info(
        "GraphService: writing %d Chunk nodes for doc %d (kb=%d)",
        len(chunks), document_id, kb_id,
    )
    with driver.session() as session:
        for idx, (text, qdrant_id) in enumerate(zip(chunks, chunk_ids)):
            session.run(
                """
                MERGE (c:Chunk {document_id: $document_id, chunk_index: $chunk_index})
                SET c.kb_id      = $kb_id,
                    c.text       = $text,
                    c.qdrant_id  = $qdrant_id,
                    c.file_name  = $file_name
                """,
                document_id=str(document_id),
                chunk_index=idx,
                kb_id=str(kb_id),
                text=text,
                qdrant_id=qdrant_id,
                file_name=file_name,
            )

    # 2. Run SimpleKGPipeline for entity/relationship extraction.
    #    embedder=None so it never calls setNodeVectorProperty — Qdrant owns
    #    embeddings. The pipeline writes __KGBuilder__ entity nodes and edges.
    builder = _make_kg_builder()
    full_text = "\n\n".join(chunks)

    logger.info(
        "GraphService: extracting entities for doc %d (%s) | %d chars",
        document_id, file_name, len(full_text),
    )

    await builder.run_async(
        text=full_text,
        document_metadata={
            "kb_id": str(kb_id),
            "document_id": str(document_id),
            "file_name": file_name,
        },
    )
    logger.info("GraphService: SimpleKGPipeline complete for doc %d", document_id)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def enrich_docs_with_graph(docs: list[LangchainDocument]) -> list[LangchainDocument]:
    """
    Append [Graph context] entity triples to each doc's text using Neo4j.

    Looks up the Chunk node by (document_id, chunk_index) — both fields are
    stored in every Qdrant payload and carried in doc.metadata. Traverses
    FROM_CHUNK edges to entities, then one hop of entity-to-entity
    relationships to build a compact triple list.

    Non-fatal: docs that fail graph lookup are returned unchanged.
    Called after RRF merge, before reranking.
    """
    if not settings.GRAPHRAG_ENABLED or not docs:
        return docs

    driver = _get_driver()

    enriched = []
    graph_hits = 0

    for doc in docs:
        doc_id  = doc.metadata.get("document_id")
        chunk_idx = doc.metadata.get("chunk_index")

        if doc_id is None or chunk_idx is None:
            enriched.append(doc)
            continue

        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chunk {document_id: $document_id, chunk_index: $chunk_index})
                    OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(entity)
                    OPTIONAL MATCH (entity)-[r]-(neighbor)
                    WHERE type(r) <> 'FROM_CHUNK'
                      AND type(r) <> 'FROM_DOCUMENT'
                      AND type(r) <> 'NEXT_CHUNK'
                    WITH c,
                         coalesce(entity.name, entity.title, entity.id) AS ename,
                         type(r)                                         AS rel,
                         coalesce(neighbor.name, neighbor.title, neighbor.id) AS nname
                    WHERE ename IS NOT NULL AND nname IS NOT NULL
                    WITH c, collect(DISTINCT [ename, rel, nname])[..40] AS triples
                    RETURN triples
                    """,
                    document_id=str(doc_id),
                    chunk_index=int(chunk_idx),
                )
                record = result.single()
                triples = record["triples"] if record else []

            if triples:
                graph_ctx = "\n[Graph context]\n" + "".join(
                    f"{t[0]} -[{t[1]}]-> {t[2]}\n" for t in triples
                )
                enriched_doc = LangchainDocument(
                    page_content=doc.page_content + graph_ctx,
                    metadata={**doc.metadata, "_graph_triples": len(triples)},
                )
                graph_hits += 1
            else:
                enriched_doc = doc

        except Exception as exc:
            logger.warning(
                "GraphService: enrichment failed for doc_id=%s chunk=%s: %s",
                doc_id, chunk_idx, exc,
            )
            enriched_doc = doc

        enriched.append(enriched_doc)

    logger.info(
        "GraphService: graph enrichment complete | %d/%d docs enriched with triples",
        graph_hits, len(docs),
    )
    return enriched


# ── Deletion ───────────────────────────────────────────────────────────────────

def delete_graph_for_document(kb_id: int, document_id: int) -> None:
    """
    Remove all Neo4j nodes for a deleted document.

    Intentionally NOT gated on GRAPHRAG_ENABLED — data may exist from a prior
    ingest run when the flag was on.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        # Delete Chunk nodes for this document
        result = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            doc_id=str(document_id),
        )
        record = result.single()
        deleted = record["deleted"] if record else 0
        logger.info(
            "GraphService: deleted %d Chunk nodes for doc %d", deleted, document_id
        )

        # Clean up entity nodes no longer connected to any Chunk
        result = session.run(
            """
            MATCH (n:__Entity__)
            WHERE NOT EXISTS { MATCH (n)-[:FROM_CHUNK]-() }
            DETACH DELETE n
            RETURN count(n) AS cleaned
            """
        )
        record = result.single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after doc %d deletion",
            record["cleaned"] if record else 0, document_id,
        )


def delete_graph_for_kb(kb_id: int) -> None:
    """
    Remove all Neo4j nodes for an entire deleted knowledge base.

    Intentionally NOT gated on GRAPHRAG_ENABLED — same reasoning as
    delete_graph_for_document.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n {kb_id: $kb_id})
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            kb_id=str(kb_id),
        )
        record = result.single()
        logger.info(
            "GraphService: deleted %d nodes for kb_%d",
            record["deleted"] if record else 0, kb_id,
        )

        # Orphaned entities (no kb_id) that lost all their Chunk connections
        result = session.run(
            """
            MATCH (n:__Entity__)
            WHERE NOT EXISTS { MATCH (n)-[]-() }
            DETACH DELETE n
            RETURN count(n) AS cleaned
            """
        )
        record = result.single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after kb_%d deletion",
            record["cleaned"] if record else 0, kb_id,
        )
