from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.week3_benchmark.bootstrap_confidence_intervals import build_bootstrap_intervals, summarize_bootstrap
from src.week3_benchmark.audit_cost_latency import summarize_cost_latency
from src.week3_benchmark.compare_metric_win_rates import run_compare
from src.week3_benchmark.run_week3_eval import _metric_summary, _summarize_by_query_class


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "outputs" / "eval_outputs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine manual and AutoQ evaluation folders.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--datasets", nargs="+", default=["manual_28", "autoq_35"])
    parser.add_argument("--output-name", default="combined")
    return parser.parse_args()


def _combine_csv(base_dir: Path, dataset_names: list[str], filename: str, out_dir: Path) -> pd.DataFrame:
    frames = []
    for name in dataset_names:
        path = base_dir / name / filename
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "dataset_name", name)
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(out_dir / filename, index=False)
    return combined


def _write_metric_summary(results: pd.DataFrame, out_dir: Path) -> None:
    if results.empty or "winner" not in results.columns:
        return
    summary_by_class = _summarize_by_query_class(results)
    summary_by_class.to_csv(out_dir / "summary_by_query_class.csv", index=False)
    summary = _metric_summary(results, summary_by_class)
    summary["datasets"] = sorted(results["dataset_name"].unique().tolist())
    (out_dir / "metric_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_bootstrap(results: pd.DataFrame, out_dir: Path) -> None:
    if results.empty:
        return
    intervals = build_bootstrap_intervals(results)
    intervals.to_csv(out_dir / "bootstrap_confidence_intervals.csv", index=False)
    (out_dir / "bootstrap_summary.json").write_text(
            json.dumps(summarize_bootstrap(intervals, 1000, 5010, 5), indent=2), encoding="utf-8"
    )


def combine_outputs(base_dir: Path, dataset_names: list[str], output_name: str = "combined") -> Path:
    out_dir = base_dir / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = _combine_csv(base_dir, dataset_names, "eval_results.csv", out_dir)
    for filename in [
        "failure_analysis.csv",
        "review_queue.csv",
        "query_difficulty.csv",
        "difficulty_summary_by_class.csv",
        "difficulty_vs_win_rate.csv",
        "graph_coverage_metrics.csv",
        "graph_coverage_summary_by_class.csv",
        "graph_coverage_vs_winner.csv",
        "metric_disagreement_analysis.csv",
        "cost_latency_audit.csv",
        "ragas_system_comparison.csv",
    ]:
        _combine_csv(base_dir, dataset_names, filename, out_dir)
    _write_metric_summary(results, out_dir)
    if not results.empty:
        run_compare(out_dir / "eval_results.csv", out_dir)
        _write_bootstrap(results, out_dir)
    cost_audit = out_dir / "cost_latency_audit.csv"
    if cost_audit.exists():
        (out_dir / "cost_latency_summary.json").write_text(
            json.dumps(summarize_cost_latency(pd.read_csv(cost_audit)), indent=2),
            encoding="utf-8",
        )
    return out_dir


def main() -> None:
    args = _parse_args()
    out_dir = combine_outputs(Path(args.base_dir), args.datasets, args.output_name)
    print(f"Saved combined evaluation artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
