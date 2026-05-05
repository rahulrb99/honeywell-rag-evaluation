from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.week3_benchmark.run_week3_eval import _write_report


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"
DEFAULT_REPORT = DEFAULT_OUT_DIR / "benchmarking_comparative_analysis.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Week 3 markdown report from artifacts.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--dataset", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    results = pd.read_csv(out_dir / "eval_results.csv")
    if args.dataset:
        results.attrs["benchmark_path"] = args.dataset
    summary_by_class = pd.read_csv(out_dir / "summary_by_query_class.csv")
    summary = json.loads((out_dir / "metric_summary.json").read_text(encoding="utf-8"))
    _write_report(Path(args.report), results, summary_by_class, summary)
    print(f"Rendered report -> {args.report}")


if __name__ == "__main__":
    main()
