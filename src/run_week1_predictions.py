"""Generate `outputs/week1_predictions.csv` from the gold eval CSV using the baseline RAG."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

from src.config import get_settings
from src.rag_pipeline import answer_question


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_contexts(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            return [raw]
    return []


def main() -> None:
    settings = get_settings()
    eval_path = Path(settings.eval_csv)
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {settings.eval_csv}")

    eval_df = pd.read_csv(eval_path)
    required_cols = {"id", "question", "ground_truth", "contexts", "category"}
    if not required_cols.issubset(eval_df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")

    run_id = str(uuid4())
    rows = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Running baseline RAG"):
        output = answer_question(str(row["question"]))
        rows.append(
            {
                "id": row.get("id", output["query_id"]),
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "contexts": _parse_contexts(row["contexts"]),
                "category": row.get("category", ""),
                "answer": output["answer"],
                "metadata_filter_used": json.dumps(output.get("metadata_filter_used")),
                "retrieved_contexts": output["retrieved_contexts"],
                "latency_ms": output["latency_ms"],
                "model_name": output["model_name"],
                "timestamp_utc": output["timestamp_utc"],
                "run_id": run_id,
            }
        )

    pred_df = pd.DataFrame(rows)
    _ensure_parent(settings.predictions_csv)
    export_df = pred_df.copy()
    for column in ("contexts", "retrieved_contexts"):
        export_df[column] = export_df[column].apply(json.dumps)
    export_df.to_csv(settings.predictions_csv, index=False)
    print(f"Saved predictions -> {settings.predictions_csv}")


if __name__ == "__main__":
    main()
