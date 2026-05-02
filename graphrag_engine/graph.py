"""Neo4j graph layer for the Honeywell GraphRAG baseline.

Public API consumed by the build and prediction scripts:
    Neo4jClient            – driver lifecycle wrapper
    ontology_to_schema     – parse TTL → {node_types, rel_types}
    build_knowledge_graph  – extract entities from docs and write to Neo4j
    tag_new_nodes_with_domain – stamp new nodes with a domain label
    query_graph_rag        – retrieve graph context and generate an answer
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase
from rdflib import OWL, RDF, Graph as RDFGraph

from config import make_llm_client


# ---------------------------------------------------------------------------
# Result types consumed by run_graphrag_predictions._context_rows
# ---------------------------------------------------------------------------

@dataclass
class GraphRAGContextItem:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRAGContext:
    items: list[GraphRAGContextItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Neo4j client
# ---------------------------------------------------------------------------

class Neo4jClient:
    """Thin wrapper around the Neo4j driver with lazy initialisation."""

    def __init__(self) -> None:
        self._uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
        self._password = os.getenv("NEO4J_PASSWORD", "")
        self._driver = None

    def __call__(self):
        """Return (and lazily create) the Neo4j driver.

        When NEO4J_PASSWORD is blank the driver is created without credentials,
        which works for instances started with NEO4J_AUTH=none.
        """
        if self._driver is not None:
            return self._driver
        auth = (self._user, self._password) if self._password else None
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=auth)
            # Eagerly verify connectivity so auth errors surface here with a
            # clear message rather than on the first Cypher call.
            self._driver.verify_connectivity()
        except Exception as exc:
            self._driver = None
            raise RuntimeError(
                f"Cannot connect to Neo4j at {self._uri}.\n"
                "Set NEO4J_URI / NEO4J_USER or NEO4J_USERNAME / NEO4J_PASSWORD in "
                "graphrag_engine/.env (leave NEO4J_PASSWORD blank for "
                "instances started with NEO4J_AUTH=none).\n"
                f"Original error: {exc}"
            ) from exc
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


# ---------------------------------------------------------------------------
# Ontology → schema
# ---------------------------------------------------------------------------

_FALLBACK_NODE_TYPES = ["Product", "Specification", "Feature", "Application", "Component"]
_FALLBACK_REL_TYPES = ["HAS_SPEC", "HAS_FEATURE", "COMPATIBLE_WITH", "MOUNTS_WITH", "CONNECTS_TO"]


def ontology_to_schema(ttl: str) -> dict[str, list[str]]:
    """Extract node types and relationship types from a Turtle ontology string."""
    g = RDFGraph()
    g.parse(data=ttl, format="turtle")

    node_types = [
        _local_name(str(cls))
        for cls in g.subjects(RDF.type, OWL.Class)
        if not str(cls).startswith("_")
    ]
    node_types = [n for n in node_types if n]

    rel_types = [
        _camel_to_upper_snake(_local_name(str(prop)))
        for prop in g.subjects(RDF.type, OWL.ObjectProperty)
        if not str(prop).startswith("_")
    ]
    rel_types = [r for r in rel_types if r]

    return {
        "node_types": node_types or _FALLBACK_NODE_TYPES,
        "rel_types": rel_types or _FALLBACK_REL_TYPES,
    }


def _local_name(uri: str) -> str:
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


def _camel_to_upper_snake(name: str) -> str:
    snaked = re.sub(r"([A-Z])", r"_\1", name).upper().lstrip("_")
    return snaked or name.upper()


# ---------------------------------------------------------------------------
# Graph build
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    driver,
    schema: dict[str, list[str]],
    documents: list[dict[str, str]],
    model_name: str,
) -> None:
    """Extract entities and relationships from every document and write to Neo4j.

    Each document dict must have ``name`` (str) and ``text`` (str) keys.
    Nodes are written WITHOUT a domain tag so that a subsequent call to
    ``tag_new_nodes_with_domain`` can stamp them atomically.
    """
    client = make_llm_client()
    _ensure_constraints(driver, schema.get("node_types", []))

    for doc_idx, doc in enumerate(documents, 1):
        chunks = _chunk_text(doc["text"], max_chars=3000, overlap=200)
        print(
            f"  doc {doc_idx}/{len(documents)}  {doc['name']}  ({len(chunks)} chunks)",
            flush=True,
        )
        for chunk_idx, chunk in enumerate(chunks, 1):
            print(f"    chunk {chunk_idx}/{len(chunks)} ...", end=" ", flush=True)
            extractions = _extract_entities(client, model_name, chunk, schema, doc["name"])
            entities = len(extractions.get("entities") or [])
            rels = len(extractions.get("relationships") or [])
            _write_extractions(driver, extractions, doc["name"])
            print(f"{entities} entities, {rels} rels", flush=True)


def _ensure_constraints(driver, node_types: list[str]) -> None:
    with driver.session() as session:
        for label in node_types:
            safe = _safe_label(label)
            try:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{safe}`) REQUIRE n.name IS UNIQUE"
                )
            except Exception:
                pass  # constraint may already exist or Neo4j edition may not support it


def _chunk_text(text: str, max_chars: int = 3000, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


_EXTRACTION_SYSTEM = """\
You are an information-extraction engine for industrial product documentation.
Extract named entities and relationships from the provided text.

Return a JSON object with exactly two keys:
  "entities"      – list of {{"label": str, "name": str, "properties": dict}}
  "relationships" – list of {{"from": str, "rel": str, "to": str}}

Guidelines:
- label must be one of the allowed node types listed below.
- name should be the canonical identifier (e.g. model number or concise term).
- properties may include description, value, unit, source_doc, etc.
- Only emit a relationship when BOTH nodes appear in the entities list.
- Use UPPER_SNAKE_CASE for relationship types.
- Keep names short and consistent across chunks."""


def _extract_entities(
    client,
    model_name: str,
    text: str,
    schema: dict[str, list[str]],
    doc_name: str,
) -> dict:
    node_types = schema.get("node_types", _FALLBACK_NODE_TYPES)
    rel_types = schema.get("rel_types", _FALLBACK_REL_TYPES)
    system = (
        _EXTRACTION_SYSTEM
        + f"\n\nAllowed node types: {', '.join(node_types)}"
        + f"\nSuggested relationship types: {', '.join(rel_types)}"
    )
    prompt = f"Document: {doc_name}\n\nText:\n{text}\n\nExtract entities and relationships. Output only valid JSON."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        # Fallback: try without json_object enforcement
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {"entities": [], "relationships": []}


def _write_extractions(driver, extractions: dict, doc_name: str) -> None:
    entities = extractions.get("entities") or []
    relationships = extractions.get("relationships") or []

    with driver.session() as session:
        for entity in entities:
            label = _safe_label(str(entity.get("label", "Entity")))
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            props: dict[str, Any] = dict(entity.get("properties") or {})
            props["name"] = name
            props["source_doc"] = doc_name
            try:
                session.run(
                    f"MERGE (n:`{label}` {{name: $name}}) SET n += $props",
                    name=name,
                    props=props,
                )
            except Exception:
                pass

        for rel in relationships:
            from_name = str(rel.get("from", "")).strip()
            to_name = str(rel.get("to", "")).strip()
            rel_type = _safe_rel_type(str(rel.get("rel", "RELATED_TO")))
            if not from_name or not to_name:
                continue
            try:
                session.run(
                    f"MATCH (a {{name: $from_name}}), (b {{name: $to_name}})"
                    f" MERGE (a)-[:`{rel_type}`]->(b)",
                    from_name=from_name,
                    to_name=to_name,
                )
            except Exception:
                pass


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", label)


def _safe_rel_type(rel: str) -> str:
    return re.sub(r"[^A-Z0-9_]", "_", rel.upper().replace(" ", "_"))


# ---------------------------------------------------------------------------
# Domain tagging
# ---------------------------------------------------------------------------

def tag_new_nodes_with_domain(driver, domain_name: str) -> int:
    """Stamp every node that has no domain with ``domain_name``.

    Returns the number of nodes tagged.
    """
    with driver.session() as session:
        record = session.run(
            "MATCH (n) WHERE n.domain IS NULL SET n.domain = $domain RETURN count(n) AS c",
            domain=domain_name,
        ).single()
    return int(record["c"]) if record else 0


# ---------------------------------------------------------------------------
# Query / answer
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = """\
You are a Honeywell product assistant.
Answer the question using ONLY the graph context provided.
If the answer is not present, respond with: "Not found in provided context."
Be concise and factual. Quote exact numeric values and units when available."""


def query_graph_rag(
    driver,
    question: str,
    model_name: str,
    domain_name: str,
) -> dict[str, Any]:
    """Retrieve graph context for *question* and generate an answer.

    Returns::

        {
            "answer":  str,
            "context": GraphRAGContext,   # .items is a list[GraphRAGContextItem]
        }
    """
    client = make_llm_client()
    context_items = _retrieve_graph_context(driver, question, domain_name)
    context_text = _format_context(context_items)
    answer = _generate_answer(client, model_name, question, context_text)
    return {"answer": answer, "context": GraphRAGContext(items=context_items)}


# ---------------------------------------------------------------------------
# Graph retrieval helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "what", "is", "the", "are", "a", "an", "of", "for", "in", "on",
    "and", "or", "to", "how", "many", "does", "do", "can", "with",
    "from", "its", "it", "be", "that", "this", "which", "where", "does",
}


def _extract_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*", question)
    return [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]


def _retrieve_graph_context(
    driver,
    question: str,
    domain_name: str,
    max_items: int = 8,
) -> list[GraphRAGContextItem]:
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    items: list[GraphRAGContextItem] = []
    with driver.session() as session:
        for keyword in keywords[:6]:
            records = session.run(
                """MATCH (n)
                   WHERE n.domain = $domain
                     AND (
                       toLower(n.name) CONTAINS toLower($kw)
                       OR toLower(coalesce(n.description, '')) CONTAINS toLower($kw)
                       OR toLower(coalesce(n.value, '')) CONTAINS toLower($kw)
                     )
                   RETURN n, labels(n) AS lbls
                   LIMIT 3""",
                domain=domain_name,
                kw=keyword,
            ).data()

            for record in records:
                node = record["n"]
                labels = record["lbls"]
                content = _node_to_text(node, labels)
                neighbors = _get_neighbors(session, node["name"], domain_name)
                if neighbors:
                    content += "\n  Related: " + "; ".join(neighbors)
                score = _keyword_overlap(question, content)
                items.append(
                    GraphRAGContextItem(
                        content=content,
                        metadata={"score": score, "labels": labels},
                    )
                )

    # Deduplicate by content then rank by keyword overlap score
    seen: set[str] = set()
    deduped: list[GraphRAGContextItem] = []
    for item in items:
        if item.content not in seen:
            seen.add(item.content)
            deduped.append(item)

    deduped.sort(key=lambda i: i.metadata.get("score", 0.0), reverse=True)
    return deduped[:max_items]


def _get_neighbors(session, node_name: str, domain_name: str) -> list[str]:
    records = session.run(
        """MATCH (n {name: $name})-[r]-(m)
           WHERE m.domain = $domain
           RETURN type(r) AS rel, m.name AS neighbor
           LIMIT 6""",
        name=node_name,
        domain=domain_name,
    ).data()
    return [f"{rec['rel']} {rec['neighbor']}" for rec in records]


def _node_to_text(node: dict, labels: list[str]) -> str:
    label_str = "/".join(labels) if labels else "Entity"
    parts = [f"[{label_str}] {node.get('name', '')}"]
    for key, value in node.items():
        if key not in ("name", "domain", "source_doc") and value is not None:
            parts.append(f"  {key}: {value}")
    return "\n".join(parts)


def _format_context(items: list[GraphRAGContextItem]) -> str:
    if not items:
        return "(no graph context found)"
    return "\n\n".join(item.content for item in items)


def _keyword_overlap(question: str, text: str) -> float:
    keywords = _extract_keywords(question)
    if not keywords:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matched / len(keywords)


def _generate_answer(client, model_name: str, question: str, context: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": f"Graph context:\n{context}\n\nQuestion: {question}\n\nAnswer in 3-6 lines."},
            ],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Not found in provided context. (Error: {exc})"
