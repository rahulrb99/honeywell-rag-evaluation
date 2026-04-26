from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader


ROOT_DIR = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT_DIR / "graph_rag_baseline"
OUTPUT_DIR = ROOT_DIR / "outputs" / "graph_rag"

sys.path.insert(0, str(GRAPH_DIR))

load_dotenv(GRAPH_DIR / ".env", override=True)
load_dotenv(ROOT_DIR / ".env")  # pick up GROQ_API_KEY from project root

from app import call_prompt_1a, validate_ttl  # noqa: E402
from config import make_llm_client  # noqa: E402
from graph import (  # noqa: E402
    Neo4jClient,
    build_knowledge_graph,
    ontology_to_schema,
    tag_new_nodes_with_domain,
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
    if not is_valid:
        raise RuntimeError(f"Generated TTL is invalid: {result}")
    print("TTL ontology OK", flush=True)
    return result


def main() -> None:
    domain_name = os.getenv("GRAPH_DOMAIN_NAME", "honeywell_fire_comms_subset_20260425")
    model_name = os.getenv("GRAPH_MODEL_NAME", "llama-3.3-70b-versatile")

    print(f"Domain : {domain_name}")
    print(f"Model  : {model_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1/4  Loading PDFs", flush=True)
    documents = _load_documents()

    print(f"\nStep 2/4  Generating ontology ({len(documents)} docs)", flush=True)
    ttl = _generate_ttl(documents, model_name)
    schema = ontology_to_schema(ttl)
    print(f"  node types : {schema['node_types']}")
    print(f"  rel types  : {schema['rel_types']}")

    print("\nStep 3/4  Connecting to Neo4j", flush=True)
    neo4j_client = Neo4jClient()
    driver = neo4j_client()
    print("  Connected", flush=True)

    print("\nStep 4/4  Building knowledge graph", flush=True)
    try:
        build_knowledge_graph(driver, schema, documents, model_name)
        tagged_nodes = tag_new_nodes_with_domain(driver, domain_name)
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
                f"tagged_nodes={tagged_nodes}",
            ]
            + [f"doc={doc['name']}" for doc in documents]
        ),
        encoding="utf-8",
    )

    print(f"Built GraphRAG domain -> {domain_name}")
    print(f"Used {len(documents)} documents")
    print(f"Tagged nodes -> {tagged_nodes}")
    print(f"Saved build summary -> {summary_path}")


if __name__ == "__main__":
    main()
