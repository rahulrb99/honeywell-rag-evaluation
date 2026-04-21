from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import get_settings
from src.llm_judge import METRIC_KEYS, score_answer


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
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    required = {"id", "question", "ground_truth", "answer", "retrieved_contexts"}
    if not required.issubset(pred_df.columns):
        raise ValueError(f"Predictions CSV must contain: {sorted(required)}")

    rows = []
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Running LLM judge"):
        retrieved_contexts = _parse_json_list_cell(row["retrieved_contexts"])
        scored = score_answer(
            question=str(row["question"]),
            answer=str(row["answer"]),
            ground_truth=str(row["ground_truth"]),
            retrieved_contexts=retrieved_contexts,
        )
        out_row: dict[str, Any] = {
            "id": row["id"],
            "question": row["question"],
            "judge_model": scored["judge_model"],
            "judge_mean_1_to_5": scored["mean_score_1_to_5"],
            "judge_mean_0_to_1": scored["mean_score_0_to_1"],
        }
        for metric in METRIC_KEYS:
            out_row[f"{metric}_score"] = scored["metrics"][metric]["score"]
            out_row[f"{metric}_rationale"] = scored["metrics"][metric]["rationale"]
        rows.append(out_row)

    result_df = pd.DataFrame(rows)
    aggregates = {
        f"{metric}_mean_1_to_5": round(float(result_df[f"{metric}_score"].mean()), 4)
        for metric in METRIC_KEYS
    }
    aggregates["judge_mean_1_to_5"] = round(float(result_df["judge_mean_1_to_5"].mean()), 4)
    aggregates["judge_mean_0_to_1"] = round(float(result_df["judge_mean_0_to_1"].mean()), 4)

    _ensure_parent(settings.judge_scores_csv)
    result_df.to_csv(settings.judge_scores_csv, index=False)

    _ensure_parent(settings.judge_scores_json)
    payload = {
        "input_predictions_csv": str(pred_path),
        "judge_scores_csv": settings.judge_scores_csv,
        "judge_model": settings.judge_model,
        "total_rows": int(len(result_df)),
        "metric_means": aggregates,
    }
    with open(settings.judge_scores_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved judge row scores -> {settings.judge_scores_csv}")
    print(f"Saved judge aggregate scores -> {settings.judge_scores_json}")
    print("Judge means:", aggregates)


if __name__ == "__main__":
    main()

