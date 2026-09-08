from __future__ import annotations

import importlib
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import networkx as nx

from .config import DEFAULT_MODEL, GRAPH_B_EXPORT_PATH, K_HOPS, Neo4jConfig, REPO_ROOT
from .normalize import normalize_entity


GRAPH_RAG_PATHS = [
    REPO_ROOT / "graphrag_engine",
    REPO_ROOT / "graph_rag_latest_upstream",
]


class GraphRAGAdapterError(RuntimeError):
    pass


class NoOpStatus:
    def update(self, **_: Any) -> None:
        return None


@contextmanager
def _graph_rag_import_path():
    existing = list(sys.path)
    for path in GRAPH_RAG_PATHS:
        if path.exists():
            sys.path.insert(0, str(path))
            break
    try:
        yield
    finally:
        sys.path[:] = existing


def _load_graph_module():
    with _graph_rag_import_path():
        return importlib.import_module("graph")


def _load_app_module():
    with _graph_rag_import_path():
        return importlib.import_module("app")


def _default_ontology() -> str:
    return """@prefix : <http://example.org/honeywell#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Product a owl:Class .
:Protocol a owl:Class .
:Voltage a owl:Class .
:Component a owl:Class .
:ErrorCode a owl:Class .

:supports_protocol a owl:ObjectProperty ;
  rdfs:domain :Product ;
  rdfs:range :Protocol .

:requires_voltage a owl:ObjectProperty ;
  rdfs:domain :Product ;
  rdfs:range :Voltage .

:has_component a owl:ObjectProperty ;
  rdfs:domain :Product ;
  rdfs:range :Component .

:reports_error a owl:ObjectProperty ;
  rdfs:domain :Product ;
  rdfs:range :ErrorCode .
"""


def generate_or_load_ontology(chunks: list[dict[str, Any]], model: str = DEFAULT_MODEL, ontology_path: Path | None = None) -> str:
    if ontology_path and ontology_path.exists():
        return ontology_path.read_text(encoding="utf-8")
    try:
        app = _load_app_module()
        if hasattr(app, "call_prompt_1a") and hasattr(app, "validate_ttl"):
            text = "\n\n".join(str(c.get("text", "")) for c in chunks[:20])
            client = app.OpenAI()
            ttl = app.call_prompt_1a(client, model, text)
            valid, _ = app.validate_ttl(ttl)
            if valid:
                if ontology_path:
                    ontology_path.parent.mkdir(parents=True, exist_ok=True)
                    ontology_path.write_text(ttl, encoding="utf-8")
                return ttl
    except Exception:
        pass
    ttl = _default_ontology()
    if ontology_path:
        ontology_path.parent.mkdir(parents=True, exist_ok=True)
        ontology_path.write_text(ttl, encoding="utf-8")
    return ttl


def _as_documents(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "name": str(chunk.get("chunk_id", "")),
            "text": str(chunk.get("text", "")),
        }
        for chunk in sorted(chunks, key=lambda c: str(c.get("chunk_id", "")))
    ]


def build_semantic_graph(
    chunks: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    neo4j_config: Neo4jConfig | None = None,
    ontology_ttl: str | None = None,
) -> dict[str, Any]:
    graph_mod = _load_graph_module()
    cfg = neo4j_config or Neo4jConfig()
    client = graph_mod.Neo4jClient(uri=cfg.uri or "", user=cfg.username or "", password=cfg.password or "")
    driver = client()
    try:
        ttl = ontology_ttl or generate_or_load_ontology(chunks, model=model)
        schema = graph_mod.ontology_to_schema(ttl)
        graph_mod.build_knowledge_graph(driver, schema, _as_documents(chunks), model=model, chunk_size=500, chunk_overlap=100)
        graph_mod.normalize_entity_names(driver)
        enrich_stats = graph_mod.enrich_relationships(driver, schema, model=model)
        weight_stats = graph_mod.compute_edge_weights(driver)
        evidence_stats = graph_mod.create_evidence_layer(driver)
        confidence_stats = graph_mod.aggregate_evidence(driver)
        stats = graph_mod.get_graph_stats(driver)
        return {
            "stats": stats,
            "enrichment": enrich_stats,
            "weights": weight_stats,
            "evidence": evidence_stats,
            "confidence": confidence_stats,
        }
    finally:
        client.close()


def _parse_relationship_text(relationship: str) -> tuple[str, str, str] | None:
    match = re.match(r"(.+?)\s+-\[(.+?)(?:\s|])", relationship)
    if not match:
        return None
    source = normalize_entity(match.group(1))
    relation = match.group(2).strip()
    target_match = re.search(r"\]->\s+(.+)$", relationship)
    target = normalize_entity(target_match.group(1)) if target_match else ""
    if not source or not target:
        return None
    return source, relation, target


def query_semantic_graph(question: str, model: str = DEFAULT_MODEL, hops: int = K_HOPS) -> dict[str, Any]:
    graph_mod = _load_graph_module()
    client = graph_mod.Neo4jClient()
    started = time.perf_counter()
    try:
        result = graph_mod.query_graph_rag(
            client(),
            question=question,
            model=model,
            hops=hops,
            weight_threshold=0.0,
            confidence_threshold=0.0,
            include_web_sources=False,
        )
        result["latency"] = time.perf_counter() - started
        return result
    finally:
        client.close()


def export_semantic_graph(driver: Any | None = None, path: Path = GRAPH_B_EXPORT_PATH) -> nx.DiGraph:
    close_client = None
    if driver is None:
        graph_mod = _load_graph_module()
        close_client = graph_mod.Neo4jClient()
        driver = close_client()

    query = """
    MATCH (a)-[r]->(b)
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
      AND NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO', 'EVIDENCE_SOURCE', 'EVIDENCE_TARGET']
    RETURN a.name AS source, type(r) AS relation, b.name AS target,
           coalesce(r.weight, 1.0) AS weight,
           coalesce(r.chunk_id, '') AS chunk_id
    ORDER BY source, relation, target
    """
    try:
        with driver.session() as session:
            rows = session.run(query).data()
    finally:
        if close_client is not None:
            close_client.close()

    graph = nx.DiGraph(graph_type="semantic_relation")
    for row in rows:
        source = normalize_entity(row.get("source"))
        target = normalize_entity(row.get("target"))
        if not source or not target:
            continue
        graph.add_node(source, name=source)
        graph.add_node(target, name=target)
        graph.add_edge(
            source,
            target,
            relation=str(row.get("relation", "")),
            weight=float(row.get("weight", 1.0) or 1.0),
            supporting_chunk_ids=str(row.get("chunk_id", "")),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)
    return graph


def get_semantic_graph_metadata() -> dict[str, Any]:
    from .graph_stats import compute_graph_stats

    graph = export_semantic_graph()
    return compute_graph_stats(graph)


def relationships_from_context(context: Any) -> list[tuple[str, str, str]]:
    relationships: list[tuple[str, str, str]] = []
    records = getattr(context, "records", None) or getattr(context, "items", None) or context
    if not isinstance(records, list):
        return relationships
    for item in records:
        data = item if isinstance(item, dict) else getattr(item, "data", {})
        for rel_text in data.get("relationships", []) if isinstance(data, dict) else []:
            parsed = _parse_relationship_text(str(rel_text))
            if parsed:
                relationships.append(parsed)
    return relationships

