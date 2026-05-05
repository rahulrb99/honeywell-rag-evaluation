from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "graphrag_engine"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"

sys.path.insert(0, str(GRAPH_DIR))
load_dotenv(ROOT / ".env")

from graph import Neo4jClient  # noqa: E402


INFRA_LABELS = ["Document", "Chunk", "WebDocument", "WebChunk", "Evidence"]
INFRA_RELS = [
    "FROM_CHUNK",
    "FROM_DOCUMENT",
    "NEXT_CHUNK",
    "SIMILAR_TO",
    "EVIDENCE_SOURCE",
    "EVIDENCE_TARGET",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Neo4j GraphRAG domain.")
    parser.add_argument(
        "--domain",
        default=os.getenv("GRAPH_DOMAIN_NAME", "honeywell_fire_comms_subset_20260425"),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise instead of writing a warning summary when Neo4j audit fails.",
    )
    return parser.parse_args()


def _run_query(driver, query: str, **params) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [dict(row) for row in session.run(query, **params)]


def _write_warning(out_dir: Path, domain: str, error: Exception) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "domain": domain,
        "status": "unavailable",
        "error": f"{type(error).__name__}: {error}",
        "node_count": 0,
        "relationship_count": 0,
    }
    (out_dir / "graph_extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(columns=["label", "count"]).to_csv(
        out_dir / "graph_entity_counts.csv", index=False
    )
    pd.DataFrame(columns=["relationship_type", "count"]).to_csv(
        out_dir / "graph_relationship_counts.csv", index=False
    )
    pd.DataFrame(columns=["source", "relationship", "target"]).to_csv(
        out_dir / "sample_triples.csv", index=False
    )


def run_audit(domain: str, out_dir: Path, strict: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Neo4jClient()
    try:
        driver = client()
        node_rows = _run_query(
            driver,
            """
            MATCH (n)
            WHERE NONE(label IN labels(n) WHERE label IN $infra_labels)
            RETURN count(n) AS count
            """,
            infra_labels=INFRA_LABELS,
        )
        rel_rows = _run_query(
            driver,
            """
            MATCH (a)-[r]->(b)
            WHERE NOT type(r) IN $infra_rels
            RETURN count(r) AS count
            """,
            infra_rels=INFRA_RELS,
        )
        label_rows = _run_query(
            driver,
            """
            MATCH (n)
            WHERE NONE(label IN labels(n) WHERE label IN $infra_labels)
            UNWIND labels(n) AS label
            WITH label
            WHERE NOT label IN ['__Entity__', '__KGBuilder__']
            RETURN label, count(*) AS count
            ORDER BY count DESC, label
            """,
            infra_labels=INFRA_LABELS,
        )
        rel_type_rows = _run_query(
            driver,
            """
            MATCH (a)-[r]->(b)
            WHERE NOT type(r) IN $infra_rels
            RETURN type(r) AS relationship_type, count(*) AS count
            ORDER BY count DESC, relationship_type
            """,
            infra_rels=INFRA_RELS,
        )
        triple_rows = _run_query(
            driver,
            """
            MATCH (a)-[r]->(b)
            WHERE NOT type(r) IN $infra_rels
            RETURN coalesce(a.name, elementId(a)) AS source,
                   type(r) AS relationship,
                   coalesce(b.name, elementId(b)) AS target
            LIMIT 20
            """,
            infra_rels=INFRA_RELS,
        )
        summary = {
            "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                "+00:00", "Z"
            ),
            "domain": domain,
            "status": "ok",
            "node_count": int(node_rows[0]["count"]) if node_rows else 0,
            "relationship_count": int(rel_rows[0]["count"]) if rel_rows else 0,
            "entity_label_count": len(label_rows),
            "relationship_type_count": len(rel_type_rows),
            "claim_count": None,
            "claim_note": "Dedicated claim extraction is not implemented in this pass.",
        }
        (out_dir / "graph_extraction_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        pd.DataFrame(label_rows).to_csv(out_dir / "graph_entity_counts.csv", index=False)
        pd.DataFrame(rel_type_rows).to_csv(
            out_dir / "graph_relationship_counts.csv", index=False
        )
        pd.DataFrame(triple_rows).to_csv(out_dir / "sample_triples.csv", index=False)
        return summary
    except Exception as exc:
        if strict:
            raise
        _write_warning(out_dir, domain, exc)
        return {"domain": domain, "status": "unavailable", "error": str(exc)}
    finally:
        client.close()


def main() -> None:
    args = _parse_args()
    summary = run_audit(args.domain, Path(args.out_dir), strict=args.strict)
    print(f"Saved graph audit -> {Path(args.out_dir) / 'graph_extraction_summary.json'}")
    if summary.get("status") != "ok":
        print(f"Graph audit warning: {summary.get('error')}")


if __name__ == "__main__":
    main()
