from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import get_settings


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = ROOT_DIR / "graphrag_engine"
sys.path.insert(0, str(GRAPH_DIR))

load_dotenv(GRAPH_DIR / ".env", override=True)
load_dotenv(ROOT_DIR / ".env")  # pick up GROQ_API_KEY from project root

from graph import Neo4jClient, query_graph_rag  # noqa: E402


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_contexts(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    return []


def _context_rows(result) -> list[dict[str, object]]:
    items = getattr(result["context"], "items", []) or []
    rows: list[dict[str, object]] = []
    for item in items:
        content = str(getattr(item, "content", ""))
        metadata = getattr(item, "metadata", {}) or {}
        rows.append(
            {
                "text": content[:1200],
                "score": metadata.get("score"),
            }
        )
    return rows


def _domain_exists(driver, domain_name: str) -> bool:
    query = "MATCH (n) WHERE n.domain = $domain RETURN count(n) AS c"
    with driver.session() as session:
        record = session.run(query, domain=domain_name).single()
    return bool(record and record["c"] > 0)


def main() -> None:
    settings = get_settings()
    eval_csv = Path(os.getenv("GRAPH_EVAL_CSV", "data/eval/week2_honeywell_eval.csv"))
    output_csv = Path(
        os.getenv("GRAPH_PREDICTIONS_CSV", "outputs/predictions/week2_graphrag_predictions.csv")
    )
    domain_name = os.getenv("GRAPH_DOMAIN_NAME", "honeywell_fire_comms_subset_20260425")
    model_name = os.getenv("GRAPH_MODEL_NAME", "llama-3.3-70b-versatile")

    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing eval CSV: {eval_csv}")

    eval_df = pd.read_csv(eval_csv)
    required_cols = {"id", "question", "ground_truth", "contexts", "category"}
    if not required_cols.issubset(eval_df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")

    core_eval_cols = {"id", "question", "ground_truth", "contexts", "category"}
    passthrough_cols = [col for col in eval_df.columns if col not in core_eval_cols]

    client = Neo4jClient()
    driver = client()
    try:
        if not _domain_exists(driver, domain_name):
            raise ValueError(
                f"GraphRAG domain '{domain_name}' does not exist in Neo4j. Build the domain first."
            )

        run_id = str(uuid4())
        rows = []
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Running GraphRAG"):
            started_at = time.perf_counter()
            result = query_graph_rag(driver, str(row["question"]), model_name, domain_name)
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            record = {
                "id": row["id"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "contexts": _parse_contexts(row["contexts"]),
                "category": row["category"],
                "answer": result["answer"],
                "retrieved_contexts": _context_rows(result),
                "latency_ms": latency_ms,
                "model_name": model_name,
                "domain_name": domain_name,
                "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                    "+00:00", "Z"
                ),
                "run_id": run_id,
            }
            for col in passthrough_cols:
                record[col] = row.get(col, "")
            rows.append(record)
    finally:
        client.close()

    out_df = pd.DataFrame(rows)
    export_df = out_df.copy()
    for column in ("contexts", "retrieved_contexts"):
        export_df[column] = export_df[column].apply(json.dumps)

    _ensure_parent(output_csv)
    export_df.to_csv(output_csv, index=False)
    print(f"Loaded eval CSV -> {eval_csv}")
    print(f"Queried GraphRAG domain -> {domain_name}")
    print(f"Saved GraphRAG predictions -> {output_csv}")


if __name__ == "__main__":
    main()
