from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.week1_vector_rag.run_eval_week1 import load_rag_outputs_df
from src.week2_graph_rag.run_eval_week2 import run_ragas_week2
from src.week3_benchmark.compare_metric_win_rates import classify_delta


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "outputs" / "predictions" / "honeywell_hard_labels_28_graphrag.csv"
DEFAULT_VECTOR = ROOT / "outputs" / "predictions" / "vector_rag_hard_labels_28_predictions.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full RAGAS for GraphRAG and vector RAG.")
    parser.add_argument("--graph-predictions", default=str(DEFAULT_GRAPH))
    parser.add_argument("--vector-predictions", default=str(DEFAULT_VECTOR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--smoke-rows", type=int, default=5)
    return parser.parse_args()


def _run_system(predictions_path: Path, rows_csv: Path, scores_json: Path, limit: int | None) -> None:
    if rows_csv.exists() and scores_json.exists():
        print(f"Reusing existing RAGAS artifacts -> {rows_csv}, {scores_json}")
        return
    pred_df = load_rag_outputs_df(predictions_path)
    if limit is not None:
        pred_df = pred_df.head(limit).copy()
    scores, rows = run_ragas_week2(pred_df)
    id_cols = [col for col in ["id", "query_class"] if col in pred_df.columns]
    for col in reversed(id_cols):
        rows.insert(0, col, list(pred_df[col].values))
    rows_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(rows_csv, index=False)
    scores_json.write_text(json.dumps(scores, indent=2), encoding="utf-8")


def _metric_value(row: pd.Series, metric: str, prefix: str) -> float:
    value = row.get(f"{prefix}_{metric}", row.get(f"{metric}_{prefix}", row.get(metric, 0.0)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_ragas_comparison(
    graph_rows: pd.DataFrame,
    vector_rows: pd.DataFrame,
    graph_scores: dict[str, float],
    vector_scores: dict[str, float],
    tie_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    graph = graph_rows.copy()
    vector = vector_rows.copy()
    if "id" not in graph.columns or "id" not in vector.columns:
        graph["id"] = range(1, len(graph) + 1)
        vector["id"] = range(1, len(vector) + 1)
    merged = graph.merge(vector, on="id", suffixes=("_graph", "_vector"))
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        metric_winners: list[str] = []
        out = {
            "id": row["id"],
            "query_class": row.get("query_class_graph", row.get("query_class_vector", "")),
        }
        for metric in RAGAS_METRICS:
            graph_value = _metric_value(row, metric, "graph")
            vector_value = _metric_value(row, metric, "vector")
            delta = round(graph_value - vector_value, 4)
            winner = classify_delta(delta, tie_threshold)
            metric_winners.append(winner)
            out[f"graph_{metric}"] = graph_value
            out[f"vector_{metric}"] = vector_value
            out[f"{metric}_winner"] = winner
        graph_wins = metric_winners.count("graph")
        vector_wins = metric_winners.count("vector")
        out["ragas_winner"] = (
            "graph"
            if graph_wins > vector_wins
            else "vector"
            if vector_wins > graph_wins
            else "tie"
        )
        rows.append(out)

    summary = {
        "rows": int(len(rows)),
        "graph_scores": graph_scores,
        "vector_scores": vector_scores,
        "winner_counts": pd.Series([row["ragas_winner"] for row in rows]).value_counts().to_dict()
        if rows
        else {},
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    limit = args.smoke_rows if args.mode == "smoke" else None
    graph_rows_csv = out_dir / "ragas_graphrag_rows.csv"
    graph_scores_json = out_dir / "ragas_graphrag_scores.json"
    vector_rows_csv = out_dir / "ragas_vector_rows.csv"
    vector_scores_json = out_dir / "ragas_vector_scores.json"

    _run_system(Path(args.graph_predictions), graph_rows_csv, graph_scores_json, limit)
    _run_system(Path(args.vector_predictions), vector_rows_csv, vector_scores_json, limit)

    graph_scores = json.loads(graph_scores_json.read_text(encoding="utf-8"))
    vector_scores = json.loads(vector_scores_json.read_text(encoding="utf-8"))
    comparison, summary = build_ragas_comparison(
        pd.read_csv(graph_rows_csv),
        pd.read_csv(vector_rows_csv),
        graph_scores,
        vector_scores,
    )
    comparison.to_csv(out_dir / "ragas_system_comparison.csv", index=False)
    (out_dir / "ragas_system_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved full RAGAS comparison -> {out_dir}")


if __name__ == "__main__":
    main()
