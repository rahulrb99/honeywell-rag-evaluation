"""Save a point-in-time experiment snapshot to outputs/experiments/.

Reads whichever eval output files currently exist and writes a single
timestamped JSON capturing all available metrics. Run after each pipeline
step to track improvement over time.

Usage:
    python -m src.save_experiment_snapshot --step step1_baseline
    python -m src.save_experiment_snapshot --step step5_reranker
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_settings


def _read_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _ragas_summary(ragas_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if ragas_json is None:
        return None
    scores = ragas_json.get("scores", ragas_json)
    return {
        "faithfulness": scores.get("faithfulness"),
        "answer_relevancy": scores.get("answer_relevancy"),
        "context_precision": scores.get("context_precision"),
        "context_recall": scores.get("context_recall"),
    }


def _judge_summary(judge_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if judge_json is None:
        return None
    means = judge_json.get("metric_means", {})
    return {
        "faithfulness_mean": means.get("faithfulness_mean_1_to_5"),
        "correctness_mean": means.get("correctness_mean_1_to_5"),
        "completeness_mean": means.get("completeness_mean_1_to_5"),
        "answer_relevance_mean": means.get("answer_relevance_mean_1_to_5"),
        "conciseness_mean": means.get("conciseness_mean_1_to_5"),
        "judge_mean_0_to_1": means.get("judge_mean_0_to_1"),
    }


def _groundedness_summary(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or "failure_tag" not in df.columns:
        return None
    return df["failure_tag"].value_counts().to_dict()


def _recall_summary(recall_json: dict[str, Any] | None) -> float | None:
    if recall_json is None:
        return None
    return recall_json.get("recall_at_k")


def build_snapshot(step: str, settings) -> dict[str, Any]:
    ragas_json = _read_json(settings.ragas_scores_json)
    judge_json = _read_json(settings.judge_scores_json)
    ground_df = _read_csv(settings.groundedness_csv)
    recall_json = _read_json(
        str(Path(settings.retrieval_recall_csv).with_suffix(".json"))
    )

    return {
        "step": step,
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ragas": _ragas_summary(ragas_json),
        "judge": _judge_summary(judge_json),
        "retrieval_recall_at_k": _recall_summary(recall_json),
        "groundedness_failure_counts": _groundedness_summary(ground_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Save experiment snapshot.")
    parser.add_argument("--step", required=True, help="Step name, e.g. step1_baseline")
    args = parser.parse_args()

    settings = get_settings()
    experiments_dir = Path(settings.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(args.step, settings)

    out_path = experiments_dir / f"{args.step}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Snapshot saved -> {out_path}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
