"""
Neo4j knowledge graph service.

Architecture
------------
Neo4j stores entities and relationships only — zero vectors.
Qdrant owns all vector search.

Ingestion (build_graph_for_document):
  1. Write Chunk nodes keyed by (document_id, chunk_index) — the same
     identifiers stored in every Qdrant point payload.
  2. Run ReLiK on each chunk to extract Entity nodes and typed relationships.
     Entities are linked to Chunks via FROM_CHUNK edges.
     No embeddings are written to Neo4j.

  This matches the Qdrant+Neo4j reference architecture:
    Qdrant: vectors + (document_id, chunk_index) payload
    Neo4j:  Entity nodes + relationships + Chunk nodes (no vectors)

Retrieval (enrich_docs_with_graph):
  Called after RRF merge (and before reranking) on the merged doc list.
  Looks up each Chunk by (document_id, chunk_index), traverses FROM_CHUNK
  edges to Entity nodes, then one hop of entity-to-entity relationships.
  Appends a [Graph context] block of triples to the chunk text.

Deletion:
  delete_graph_for_document / delete_graph_for_kb — always run regardless of
  the GRAPHRAG_ENABLED flag so data is never orphaned when the flag is toggled.
"""

import logging
from typing import Optional

import neo4j
from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_neo4j_driver: Optional[neo4j.Driver] = None
_relik_model = None   # loaded once on first ingest call


def _get_driver() -> neo4j.Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _neo4j_driver


def _get_relik():
    """Lazy-load ReLiK model singleton (CPU, small variant)."""
    global _relik_model
    if _relik_model is None:
        from relik import Relik
        logger.info("GraphService: loading ReLiK model (first call)...")
        _relik_model = Relik.from_pretrained(
            "relik-ie/relik-relation-extraction-small",
            device="cpu",
        )
        logger.info("GraphService: ReLiK model loaded")
    return _relik_model


# ── Ingest ─────────────────────────────────────────────────────────────────────

async def build_graph_for_document(
    kb_id: int,
    document_id: int,
    file_name: str,
    chunks: list[str],
    chunk_ids: list[str],
) -> None:
    """
    Extract entity/relationship graph from document chunks and store in Neo4j.

    Called from document_processor.py step 9. Non-fatal — callers catch all
    exceptions so a Neo4j failure never breaks the core retrieval pipeline.

    Per-chunk ReLiK extraction writes:
      - Entity nodes  (label=Entity, properties: name, type)
      - Typed relationship edges between entities
      - FROM_CHUNK edges linking Entity → Chunk

    Zero embeddings are written to Neo4j.
    """
    if not settings.GRAPHRAG_ENABLED:
        return

    driver = _get_driver()
    relik = _get_relik()

    # 1. Write Chunk nodes — keyed by (document_id, chunk_index) to match
    #    the payload Qdrant stores on every point.
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

    # 2. Extract entities and relationships per chunk with ReLiK.
    total_entities = 0
    total_relations = 0

    with driver.session() as session:
        for idx, text in enumerate(chunks):
            out = relik(text)

            # Write Entity nodes and FROM_CHUNK edges
            for span in out.spans:
                entity_name = span.text.strip()
                entity_type = span.label if span.label and span.label != "--NME--" else "Entity"
                if not entity_name:
                    continue
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type
                    WITH e
                    MATCH (c:Chunk {document_id: $document_id, chunk_index: $chunk_index})
                    MERGE (e)-[:FROM_CHUNK]->(c)
                    """,
                    name=entity_name,
                    type=entity_type,
                    document_id=str(document_id),
                    chunk_index=idx,
                )
                total_entities += 1

            # Write typed relationship edges between entity pairs
            for triplet in out.triplets:
                subj = triplet.subject.text.strip()
                obj  = triplet.object.text.strip()
                rel  = triplet.label.upper().replace(" ", "_") if triplet.label else "RELATED_TO"
                if not subj or not obj:
                    continue
                # Use APOC-safe dynamic relationship via Cypher parameter workaround:
                # Neo4j doesn't support parameterised rel types, so we whitelist via
                # a safe identifier (alphanumeric + underscore only).
                rel_safe = "".join(c if c.isalnum() or c == "_" else "_" for c in rel)
                session.run(
                    f"""
                    MERGE (a:Entity {{name: $subj}})
                    MERGE (b:Entity {{name: $obj}})
                    MERGE (a)-[:`{rel_safe}`]->(b)
                    """,
                    subj=subj,
                    obj=obj,
                )
                total_relations += 1

    logger.info(
        "GraphService: doc %d — %d entities, %d relations written to Neo4j",
        document_id, total_entities, total_relations,
    )


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
        doc_id    = doc.metadata.get("document_id")
        chunk_idx = doc.metadata.get("chunk_index")

        if doc_id is None or chunk_idx is None:
            enriched.append(doc)
            continue

        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chunk {document_id: $document_id, chunk_index: $chunk_index})
                    OPTIONAL MATCH (entity:Entity)-[:FROM_CHUNK]->(c)
                    OPTIONAL MATCH (entity)-[r]-(neighbor:Entity)
                    WITH c,
                         entity.name                                            AS ename,
                         type(r)                                                AS rel,
                         neighbor.name                                          AS nname
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
        result = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            doc_id=str(document_id),
        )
        record = result.single()
        logger.info(
            "GraphService: deleted %d Chunk nodes for doc %d",
            record["deleted"] if record else 0, document_id,
        )

        # Clean up Entity nodes no longer connected to any Chunk
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """
        )
        record = result.single()
        logger.info(
            "GraphService: cleaned %d orphaned Entity nodes after doc %d deletion",
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

        # Orphaned Entity nodes with no remaining connections
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE NOT EXISTS { MATCH (e)-[]-() }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """
        )
        record = result.single()
        logger.info(
            "GraphService: cleaned %d orphaned Entity nodes after kb_%d deletion",
            record["cleaned"] if record else 0, kb_id,
        )
