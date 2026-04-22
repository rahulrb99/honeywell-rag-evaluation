"""Run groundedness analysis on week1_predictions.csv.

Reads predictions + judge scores, computes per-row chunk attribution using
dual-threshold cosine similarity, and writes week1_groundedness.csv.

Usage:
    python -m src.run_groundedness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import get_settings
from src.groundedness import analyse_row


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_json_list_cell(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def main() -> None:
    settings = get_settings()

    pred_path = Path(settings.predictions_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    required = {"id", "question", "answer", "retrieved_contexts"}
    if not required.issubset(pred_df.columns):
        raise ValueError(f"Predictions CSV must contain: {sorted(required)}")

    # Load judge scores for correctness lookup (optional)
    judge_path = Path(settings.judge_scores_csv)
    correctness_map: dict[Any, int] = {}
    if judge_path.exists():
        judge_df = pd.read_csv(judge_path)
        if "correctness_score" in judge_df.columns and "id" in judge_df.columns:
            correctness_map = dict(
                zip(judge_df["id"], judge_df["correctness_score"].astype(int))
            )

    rows = []
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Groundedness"):
        retrieved_contexts = _parse_json_list_cell(row["retrieved_contexts"])
        correctness_score = correctness_map.get(row["id"])

        result = analyse_row(
            answer=str(row["answer"]),
            retrieved_contexts=retrieved_contexts,
            correctness_score=int(correctness_score) if correctness_score is not None else None,
        )
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "failure_tag": result["failure_tag"],
                "max_chunk_sim": result["max_chunk_sim"],
                "used_status": result["used_status"],
                "used_chunk_ids": json.dumps(result["used_chunk_ids"]),
                "top_chunk_text": result["top_chunk_text"],
            }
        )

    result_df = pd.DataFrame(rows)
    _ensure_parent(settings.groundedness_csv)
    result_df.to_csv(settings.groundedness_csv, index=False)

    tag_counts = result_df["failure_tag"].value_counts().to_dict()
    print(f"Saved groundedness -> {settings.groundedness_csv}")
    print("Failure tag counts:", tag_counts)


if __name__ == "__main__":
    main()
