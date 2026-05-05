from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.week3_benchmark.run_week3_eval import _parse_json_cell, _tokens


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"
CLASS_POINTS = {
    "single_hop_fact": 0,
    "local_specific": 0,
    "manual_curated": 1,
    "comparison": 2,
    "multi_hop": 2,
    "relationship_reasoning": 2,
    "theme_summary": 2,
    "global_specific": 2,
    "global_thematic": 3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score benchmark query difficulty.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def _list_len(value: Any) -> int:
    parsed = _parse_json_cell(value, [])
    return len(parsed) if isinstance(parsed, list) else 0


def difficulty_score(row: pd.Series) -> tuple[int, str, list[str]]:
    points = CLASS_POINTS.get(str(row.get("query_class", "")), 1)
    reasons = [f"class:{row.get('query_class', '')}"]
    expected_entities = _list_len(row.get("expected_entities", "[]"))
    source_fields = _list_len(row.get("source_fields", "[]"))
    contexts = _list_len(row.get("contexts", "[]"))
    answer_tokens = len(_tokens(row.get("ground_truth", "")))
    if expected_entities >= 2:
        points += 1
        reasons.append("multiple_expected_entities")
    if source_fields >= 2:
        points += 1
        reasons.append("multiple_source_fields")
    if contexts >= 2:
        points += 1
        reasons.append("multiple_supporting_contexts")
    if answer_tokens >= 24:
        points += 1
        reasons.append("long_reference_answer")
    score = max(1, min(5, 1 + points))
    band = "easy" if score <= 2 else "medium" if score == 3 else "hard" if score == 4 else "very_hard"
    return score, band, reasons


def build_query_difficulty(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in results.iterrows():
        score, band, reasons = difficulty_score(row)
        rows.append(
            {
                "id": row["id"],
                "question": row.get("question", ""),
                "query_class": row.get("query_class", ""),
                "difficulty_score": score,
                "difficulty_band": band,
                "difficulty_reasons": ";".join(reasons),
                "winner": row.get("winner", ""),
                "graph_score": row.get("graph_score", math.nan),
                "vector_score": row.get("vector_score", math.nan),
                "failure_mode": row.get("failure_mode", ""),
            }
        )
    return pd.DataFrame(rows)


def _summary_by_class(difficulty: pd.DataFrame) -> pd.DataFrame:
    if difficulty.empty:
        return pd.DataFrame()
    return (
        difficulty.groupby("query_class", dropna=False)
        .agg(
            rows=("id", "count"),
            mean_difficulty=("difficulty_score", "mean"),
            graph_wins=("winner", lambda s: int((s == "graph").sum())),
            vector_wins=("winner", lambda s: int((s == "vector").sum())),
        )
        .reset_index()
        .round({"mean_difficulty": 4})
    )


def _difficulty_vs_win_rate(difficulty: pd.DataFrame) -> pd.DataFrame:
    if difficulty.empty:
        return pd.DataFrame()
    rows = []
    for band, group in difficulty.groupby("difficulty_band", dropna=False):
        rows.append(
            {
                "difficulty_band": band,
                "rows": len(group),
                "mean_difficulty": round(float(group["difficulty_score"].mean()), 4),
                "graph_win_rate": round(float((group["winner"] == "graph").mean()), 4),
                "vector_win_rate": round(float((group["winner"] == "vector").mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    results = pd.read_csv(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    difficulty = build_query_difficulty(results)
    difficulty.to_csv(out_dir / "query_difficulty.csv", index=False)
    _summary_by_class(difficulty).to_csv(out_dir / "difficulty_summary_by_class.csv", index=False)
    _difficulty_vs_win_rate(difficulty).to_csv(out_dir / "difficulty_vs_win_rate.csv", index=False)
    print(f"Saved query difficulty artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
