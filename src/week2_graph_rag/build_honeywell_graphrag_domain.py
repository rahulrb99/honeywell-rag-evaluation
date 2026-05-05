from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = ROOT_DIR / "graphrag_engine"
OUTPUT_DIR = ROOT_DIR / "outputs" / "graph_rag"

sys.path.insert(0, str(GRAPH_DIR))

load_dotenv(ROOT_DIR / ".env")

from app import call_prompt_1a, call_prompt_2, validate_ttl  # noqa: E402
from config import make_llm_client  # noqa: E402
from graph import (  # noqa: E402
    Neo4jClient,
    aggregate_evidence,
    build_knowledge_graph,
    compute_edge_weights,
    create_evidence_layer,
    get_graph_stats,
    normalize_entity_names,
    ontology_to_schema,
)


DOC_PATHS = [
    ROOT_DIR / "data" / "raw" / "Zone_expander Data Sheet.pdf",
    ROOT_DIR / "data" / "raw" / "L-Series_Speakers_Strobes_Wall_DataSheet_AVDS867.pdf",
    ROOT_DIR / "data" / "raw" / "Public Address Speakers datasheet.pdf",
    ROOT_DIR / "data" / "raw" / "FirstCommand Fire Fighter Telephone_Honeywell Building Automation.pdf",
    ROOT_DIR / "data" / "raw" / "Notifier First Command Remote Microphone_Honeywell Building Automation.pdf",
    ROOT_DIR / "data" / "raw" / "lsecs info guide.pdf",
    ROOT_DIR / "data" / "raw" / "slc wiring manual.pdf",
    ROOT_DIR / "data" / "raw" / "DAA2 amplifier Manual.pdf",
]


def _extract_pdf_text(path: Path) -> str:
    docs = PDFPlumberLoader(str(path)).load()
    text = "\n".join(doc.page_content for doc in docs if str(doc.page_content or "").strip())
    return text.encode("ascii", "ignore").decode("ascii")


def _load_documents() -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for idx, path in enumerate(DOC_PATHS, 1):
        if not path.exists():
            print(f"  [{idx}/{len(DOC_PATHS)}] Skipped (not found): {path.name}", flush=True)
            continue
        print(f"  [{idx}/{len(DOC_PATHS)}] Parsing {path.name} ...", flush=True)
        text = _extract_pdf_text(path)
        if not text.strip():
            print(f"  [{idx}/{len(DOC_PATHS)}] Skipped (no extractable text)", flush=True)
            continue
        documents.append({"name": path.name, "text": text})
        print(f"  [{idx}/{len(DOC_PATHS)}] OK — {len(text):,} chars", flush=True)
    if not documents:
        raise ValueError("No extractable documents found for GraphRAG domain build.")
    return documents


def _generate_ttl(documents: list[dict[str, str]], model_name: str) -> str:
    print("Generating TTL ontology via LLM ...", flush=True)
    client = make_llm_client()
    document_block = "\n\n".join(
        f"--- Document {idx}: {doc['name']} ---\n{doc['text']}"
        for idx, doc in enumerate(documents, start=1)
    )
    ttl = call_prompt_1a(client, model_name, document_block)
    is_valid, result = validate_ttl(ttl)
    repair_attempts = int(os.getenv("GRAPH_TTL_REPAIR_ATTEMPTS", "3"))
    for attempt in range(1, repair_attempts + 1):
        if is_valid:
            break
        raw_path = OUTPUT_DIR / f"invalid_ontology_attempt_{attempt}.ttl"
        raw_path.write_text(ttl, encoding="utf-8")
        print(
            f"TTL validation failed on attempt {attempt}/{repair_attempts}: {result}",
            flush=True,
        )
        print(f"  Saved invalid TTL -> {raw_path}", flush=True)
        print("  Asking LLM syntax validator to repair TTL ...", flush=True)
        ttl = call_prompt_2(client, model_name, ttl, errors=result)
        is_valid, result = validate_ttl(ttl)
    if not is_valid:
        raise RuntimeError(f"Generated TTL is invalid after repair attempts: {result}")
    print("TTL ontology OK", flush=True)
    return result


def _schema_labels(schema) -> tuple[list[str], list[str]]:
    node_labels = [node.label for node in getattr(schema, "node_types", [])]
    rel_labels = [rel.label for rel in getattr(schema, "relationship_types", [])]
    return node_labels, rel_labels


def _clear_graph(driver) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def main() -> None:
    domain_name = os.getenv("GRAPH_DOMAIN_NAME", "honeywell_fire_comms_subset_20260425")
    model_name = os.getenv("PIPELINE_GRAPH_MODEL_NAME") or os.getenv(
        "GRAPH_MODEL_NAME", "gpt-4o-mini"
    )
    if model_name.startswith("llama-"):
        model_name = "gpt-4o-mini"

    print(f"Domain : {domain_name}")
    print(f"Model  : {model_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1/4  Loading PDFs", flush=True)
    documents = _load_documents()

    print(f"\nStep 2/4  Generating ontology ({len(documents)} docs)", flush=True)
    ttl = _generate_ttl(documents, model_name)
    schema = ontology_to_schema(ttl)
    node_labels, rel_labels = _schema_labels(schema)
    print(f"  node types : {node_labels}")
    print(f"  rel types  : {rel_labels}")

    print("\nStep 3/4  Connecting to Neo4j", flush=True)
    neo4j_client = Neo4jClient()
    driver = neo4j_client()
    print("  Connected", flush=True)

    print("\nStep 4/4  Building knowledge graph", flush=True)
    try:
        if os.getenv("CLEAR_GRAPH_BEFORE_BUILD", "true").lower() in {"1", "true", "yes", "on"}:
            print("  Clearing existing Neo4j graph before latest Graph_RAG build", flush=True)
            _clear_graph(driver)
        build_knowledge_graph(
            driver,
            schema,
            documents,
            model_name,
            chunk_size=int(os.getenv("GRAPH_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("GRAPH_CHUNK_OVERLAP", "250")),
        )
        normalized = normalize_entity_names(driver)
        weights = compute_edge_weights(driver)
        evidence = create_evidence_layer(driver)
        evidence_agg = aggregate_evidence(driver)
        graph_stats = get_graph_stats(driver)
    finally:
        neo4j_client.close()

    summary_path = OUTPUT_DIR / f"{domain_name}_build_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"built_at_utc={datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
                f"domain_name={domain_name}",
                f"model_name={model_name}",
                f"documents={len(documents)}",
                f"node_types={len(node_labels)}",
                f"relationship_types={len(rel_labels)}",
                f"normalized_entities={normalized.get('fixed', 0)}",
                f"weighted_relationships={weights.get('updated', 0)}",
                f"evidence_nodes={evidence.get('created', 0)}",
                f"evidence_relationships_aggregated={evidence_agg.get('updated', 0)}",
                f"graph_entities={graph_stats.get('entities', 0)}",
                f"graph_relationships={graph_stats.get('relationships', 0)}",
            ]
            + [f"doc={doc['name']}" for doc in documents]
        ),
        encoding="utf-8",
    )

    print(f"Built GraphRAG domain -> {domain_name}")
    print(f"Used {len(documents)} documents")
    print(f"Graph stats -> {graph_stats}")
    print(f"Saved build summary -> {summary_path}")


if __name__ == "__main__":
    main()
