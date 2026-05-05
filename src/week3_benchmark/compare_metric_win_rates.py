from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"
METRICS = [
    "entity_recall",
    "context_precision",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]
TIE_THRESHOLD = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GraphRAG and vector RAG per metric.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tie-threshold", type=float, default=TIE_THRESHOLD)
    return parser.parse_args()


def classify_delta(delta: float, tie_threshold: float = TIE_THRESHOLD) -> str:
    if abs(delta) <= tie_threshold:
        return "tie"
    return "graph" if delta > 0 else "vector"


def build_per_metric_comparison(
    results: pd.DataFrame, tie_threshold: float = TIE_THRESHOLD
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in results.iterrows():
        for metric in METRICS:
            graph_col = f"graph_{metric}"
            vector_col = f"vector_{metric}"
            if graph_col not in results.columns or vector_col not in results.columns:
                continue
            graph_value = float(row[graph_col])
            vector_value = float(row[vector_col])
            delta = round(graph_value - vector_value, 4)
            rows.append(
                {
                    "id": row["id"],
                    "query_class": row.get("query_class", ""),
                    "metric": metric,
                    "graph_value": graph_value,
                    "vector_value": vector_value,
                    "delta": delta,
                    "winner": classify_delta(delta, tie_threshold),
                    "row_winner": row.get("winner", ""),
                    "failure_mode": row.get("failure_mode", ""),
                }
            )
    return pd.DataFrame(rows)


def _summarize(
    grouped: pd.core.groupby.generic.DataFrameGroupBy, group_names: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        counts = Counter(group["winner"])
        if not isinstance(key, tuple):
            key = (key,)
        row = {
            "graph_wins": counts.get("graph", 0),
            "vector_wins": counts.get("vector", 0),
            "ties": counts.get("tie", 0),
            "rows": len(group),
            "graph_win_rate": round(counts.get("graph", 0) / max(1, len(group)), 4),
            "vector_win_rate": round(counts.get("vector", 0) / max(1, len(group)), 4),
            "mean_delta": round(float(group["delta"].mean()), 4),
        }
        for idx, name in enumerate(group_names):
            row[name] = key[idx]
        rows.append(row)
    cols = group_names + [
        "rows",
        "graph_wins",
        "vector_wins",
        "ties",
        "graph_win_rate",
        "vector_win_rate",
        "mean_delta",
    ]
    return pd.DataFrame(rows)[cols].sort_values(group_names)


def run_compare(results_path: Path, out_dir: Path, tie_threshold: float = TIE_THRESHOLD) -> None:
    results = pd.read_csv(results_path)
    comparison = build_per_metric_comparison(results, tie_threshold)
    by_class = _summarize(
        comparison.groupby(["query_class", "metric"], dropna=False),
        ["query_class", "metric"],
    )
    overall = _summarize(comparison.groupby(["metric"], dropna=False), ["metric"])

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "per_metric_comparison.csv", index=False)
    by_class.to_csv(out_dir / "metric_win_rates_by_class.csv", index=False)
    overall.to_csv(out_dir / "metric_win_rates_overall.csv", index=False)


def main() -> None:
    args = _parse_args()
    run_compare(Path(args.results), Path(args.out_dir), args.tie_threshold)
    print(f"Saved per-metric comparison outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
