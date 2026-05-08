"""
Neo4j knowledge graph service.

Responsibilities:
  - Lazy Neo4j driver singleton
  - Lazy SimpleKGPipeline singleton (entity/relationship extraction at ingest)
  - build_graph_for_document()   — called from ingestion pipeline (step 9)
  - graph_search()               — called from retrieval pipeline (leg 4)
  - delete_graph_for_document()  — called on document/KB deletion

All public functions are non-fatal by design: callers catch exceptions and log
warnings so that Neo4j failures never break the core 3-leg RAG pipeline.
"""

import logging
from typing import Optional

import neo4j
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_neo4j_driver: Optional[neo4j.Driver] = None
_embedder: Optional[OpenAIEmbeddings] = None
# Note: SimpleKGPipeline is NOT a singleton — a fresh instance is created per
# document build to avoid shared state across concurrent ingestion tasks.


def _get_driver() -> neo4j.Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _neo4j_driver


def _get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDINGS_MODEL,
            base_url=settings.OPENAI_API_BASE,
            api_key=settings.OPENAI_API_KEY,
        )
    return _embedder


def _make_kg_builder() -> SimpleKGPipeline:
    """Create a new SimpleKGPipeline instance.

    Called per-document rather than as a singleton — SimpleKGPipeline holds
    internal pipeline state that is not safe to share across concurrent runs.
    The driver and embedder are still singletons (stateless / thread-safe).
    """
    llm = OpenAILLM(
        model_name=settings.graphrag_model,
        model_params={"temperature": 0},
        base_url=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=_get_driver(),
        embedder=_get_embedder(),
        from_pdf=False,
    )


def _ensure_neo4j_vector_index(driver: neo4j.Driver, kb_id: int, dims: int) -> None:
    """Create a vector index on Chunk nodes for this KB if it doesn't exist."""
    index_name = f"kb_{kb_id}_text_embeddings"
    with driver.session() as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dims},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
    logger.info("GraphService: ensured vector index %s (dims=%d)", index_name, dims)


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
    exceptions so a Neo4j failure never prevents the document from being
    indexed via the other 3 legs.

    Args:
        kb_id:       Knowledge base ID (used to scope the vector index).
        document_id: MySQL document ID (written to Chunk node for linking).
        file_name:   Human-readable filename for logging.
        chunks:      Plain-text chunk strings from the existing splitter.
        chunk_ids:   UUID hex strings (same IDs used as Qdrant point IDs).
    """
    if not settings.GRAPHRAG_ENABLED:
        return

    driver = _get_driver()
    _ensure_neo4j_vector_index(driver, kb_id, settings.DENSE_EMBEDDING_DIM)

    builder = _make_kg_builder()
    full_text = "\n\n".join(chunks)

    logger.info(
        "GraphService: building graph for doc %d (%s) | kb=%d | chunks=%d | chars=%d",
        document_id, file_name, kb_id, len(chunks), len(full_text),
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


def graph_search(
    query_text: str,
    kb_ids: list[int],
    top_k: int,
) -> list[LangchainDocument]:
    """
    Run VectorCypherRetriever per KB and return graph-enriched results.

    Uses Neo4j's own vector index (seeded at ingest by SimpleKGPipeline) to
    find the nearest Chunk nodes, then traverses up to GRAPHRAG_RETRIEVAL_HOPS
    hops of entity relationships and appends a graph-context summary to each
    chunk's text.

    Returns LangchainDocument objects compatible with the existing RRF merger.
    """
    from neo4j_graphrag.retrievers import VectorCypherRetriever
    from neo4j_graphrag.types import RetrieverResultItem

    def _graph_result_formatter(record) -> RetrieverResultItem:
        """Extract the 'info' column string from the retrieval_query result."""
        return RetrieverResultItem(content=record.get("info") or "", metadata={})

    hops = settings.GRAPHRAG_RETRIEVAL_HOPS

    # Traverse entity relationships outward from seed chunk nodes.
    # Variable-length rel patterns cannot use type negation (!FROM_CHUNK*1..N) in Neo4j.
    # Instead: hop from chunk → entities via FROM_CHUNK, then traverse any relationship
    # between entities (excluding FROM_CHUNK explicitly via WHERE).
    retrieval_query = f"""
    WITH node AS chunk
    OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
    OPTIONAL MATCH (entity)-[r]-(neighbor)
    WHERE type(r) <> 'FROM_CHUNK'
      AND type(r) <> 'FROM_DOCUMENT'
      AND type(r) <> 'NEXT_CHUNK'
    WITH chunk,
         entity,
         r,
         neighbor,
         coalesce(entity.name, entity.title, entity.id) AS ename,
         coalesce(neighbor.name, neighbor.title, neighbor.id) AS nname
    WHERE ename IS NOT NULL AND nname IS NOT NULL
    WITH chunk, collect(DISTINCT [ename, type(r), nname]) AS triples
    WITH chunk, triples[..40] AS triples
    RETURN chunk.text +
      CASE WHEN size(triples) > 0
        THEN '\\n\\n[Graph context]\\n' +
             reduce(s='', t IN triples |
               s + t[0] + ' -[' + t[1] + ']-> ' + t[2] + '\\n'
             )
        ELSE ''
      END AS info
    """

    embedder = _get_embedder()
    driver = _get_driver()
    results: list[LangchainDocument] = []

    for kb_id in kb_ids:
        index_name = f"kb_{kb_id}_text_embeddings"
        try:
            retriever = VectorCypherRetriever(
                driver=driver,
                index_name=index_name,
                embedder=embedder,
                retrieval_query=retrieval_query,
                result_formatter=_graph_result_formatter,
            )
            r = retriever.search(query_text=query_text, top_k=top_k)
            for item in r.items:
                results.append(LangchainDocument(
                    page_content=item.content,
                    metadata={"kb_id": kb_id, "source": "graph"},
                ))
            logger.info(
                "GraphService: graph_search kb_%d returned %d items",
                kb_id, len(r.items),
            )
        except Exception as e:
            logger.warning(
                "GraphService: VectorCypherRetriever failed for kb_%d: %s", kb_id, e
            )

    return results


def delete_graph_for_document(kb_id: int, document_id: int) -> None:
    """
    Remove all Neo4j nodes for a deleted document.

    Intentionally NOT gated on GRAPHRAG_ENABLED — same reasoning as
    delete_graph_for_kb: data may exist from a prior ingest run when the
    flag was on.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        # Delete Chunk nodes that belong to this document (via FROM_DOCUMENT)
        result = session.run(
            """
            MATCH (d:Document {document_id: $doc_id})-[:FROM_DOCUMENT]-(c:Chunk)
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

        # Delete the Document node itself
        session.run(
            "MATCH (d:Document {document_id: $doc_id}) DETACH DELETE d",
            doc_id=str(document_id),
        )

        # Clean up entity nodes no longer connected to any Chunk
        session.run(
            """
            MATCH (n:__Entity__)
            WHERE NOT EXISTS { MATCH (n)-[:FROM_CHUNK]-() }
            DETACH DELETE n
            """,
        )

        # Clean up any orphaned Chunk nodes (no FROM_DOCUMENT link — debris from failed ingests)
        session.run(
            """
            MATCH (c:Chunk)
            WHERE NOT EXISTS { MATCH (c)-[:FROM_DOCUMENT]-() }
            DETACH DELETE c
            """,
        )
    logger.info(
        "GraphService: orphan cleanup complete for kb_%d after doc %d deletion",
        kb_id, document_id,
    )


def delete_graph_for_kb(kb_id: int) -> None:
    """
    Remove all Neo4j nodes for an entire deleted knowledge base.
    Also drops the vector index for this KB.

    Intentionally NOT gated on GRAPHRAG_ENABLED — deletion must run regardless
    of whether GraphRAG is currently enabled, since data may have been written
    when the flag was on. Skipping deletion based on a runtime flag causes
    orphaned nodes whenever the flag is toggled between ingest and delete.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        # 1. Delete all Chunk and Document nodes for this KB, along with any
        #    entity nodes exclusively connected to those chunks (DETACH DELETE
        #    removes the relationships; the WHERE NOT EXISTS guard below catches
        #    entities whose *only* connections were to this KB's chunks).
        result = session.run(
            """
            MATCH (n {kb_id: $kb_id})
            WITH collect(n) AS kb_nodes
            UNWIND kb_nodes AS n
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            kb_id=str(kb_id),
        )
        record = result.single()
        deleted_kb_nodes = record["deleted"] if record else 0
        logger.info(
            "GraphService: deleted %d nodes with kb_id for kb_%d",
            deleted_kb_nodes, kb_id
        )

        # 2. Clean up entity nodes that are now fully disconnected (their only
        #    FROM_CHUNK links pointed to the chunks we just deleted above).
        result = session.run(
            """
            MATCH (e:__Entity__)
            WHERE NOT EXISTS { MATCH (e)-[]-() }
            DETACH DELETE e
            RETURN count(e) AS cleaned_orphans
            """,
        )
        record = result.single()
        cleaned_orphans = record["cleaned_orphans"] if record else 0
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after kb_%d deletion",
            cleaned_orphans, kb_id
        )

        # Drop the vector index for this KB
        index_name = f"kb_{kb_id}_text_embeddings"
        try:
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            logger.info("GraphService: dropped vector index %s", index_name)
        except Exception as e:
            logger.warning(
                "GraphService: could not drop index %s: %s", index_name, e
            )
