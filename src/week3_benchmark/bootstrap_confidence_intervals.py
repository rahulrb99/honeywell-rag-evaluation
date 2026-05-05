from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"
DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 5010
MIN_CLASS_N = 5
HARD_QUERY_CLASSES = {"multi_hop", "theme_summary", "relationship_reasoning", "comparison"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bootstrap confidence intervals.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-class-n", type=int, default=MIN_CLASS_N)
    return parser.parse_args()


def _rate(df: pd.DataFrame, label: str) -> float:
    return float((df["winner"] == label).mean()) if len(df) else 0.0


def _score_delta(df: pd.DataFrame) -> float:
    return float((df["graph_score"] - df["vector_score"]).mean()) if len(df) else 0.0


def _bootstrap(values: pd.DataFrame, metric_fn, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = []
    n = len(values)
    for _ in range(resamples):
        indices = rng.integers(0, n, size=n)
        sample = values.iloc[indices]
        stats.append(metric_fn(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _ci_row(
    name: str,
    scope: str,
    df: pd.DataFrame,
    metric_fn,
    resamples: int,
    seed: int,
    min_n: int = 1,
) -> dict[str, object]:
    point = metric_fn(df) if len(df) else 0.0
    if len(df) < min_n:
        return {
            "metric": name,
            "scope": scope,
            "n": int(len(df)),
            "point_estimate": round(point, 4),
            "ci_lower": "",
            "ci_upper": "",
            "status": "insufficient_n_for_ci",
        }
    lower, upper = _bootstrap(df, metric_fn, resamples, seed)
    return {
        "metric": name,
        "scope": scope,
        "n": int(len(df)),
        "point_estimate": round(point, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "status": "ok",
    }


def build_bootstrap_intervals(
    results: pd.DataFrame,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    min_class_n: int = MIN_CLASS_N,
) -> pd.DataFrame:
    rows = [
        _ci_row("graph_win_rate", "overall", results, lambda df: _rate(df, "graph"), resamples, seed),
        _ci_row("vector_win_rate", "overall", results, lambda df: _rate(df, "vector"), resamples, seed + 1),
        _ci_row("mean_graph_vector_score_delta", "overall", results, _score_delta, resamples, seed + 2),
    ]
    hard = results[results["query_class"].isin(HARD_QUERY_CLASSES)]
    rows.append(
        _ci_row(
            "graph_win_rate",
            "hard_queries",
            hard,
            lambda df: _rate(df, "graph"),
            resamples,
            seed + 3,
        )
    )
    for query_class, group in results.groupby("query_class", dropna=False):
        rows.append(
            _ci_row(
                "graph_win_rate",
                f"query_class:{query_class}",
                group,
                lambda df: _rate(df, "graph"),
                resamples,
                seed + int(hashlib.sha256(str(query_class).encode("utf-8")).hexdigest()[:8], 16) % 100_000,
                min_n=min_class_n,
            )
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(
    intervals: pd.DataFrame, resamples: int, seed: int, min_class_n: int = MIN_CLASS_N
) -> dict[str, object]:
    return {
        "resamples": resamples,
        "seed": seed,
        "min_class_n_for_ci": min_class_n,
        "rows": int(len(intervals)),
        "ok_rows": int((intervals["status"] == "ok").sum()) if not intervals.empty else 0,
        "insufficient_n_rows": int((intervals["status"] == "insufficient_n_for_ci").sum())
        if not intervals.empty
        else 0,
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals = build_bootstrap_intervals(
        pd.read_csv(args.results), args.resamples, args.seed, args.min_class_n
    )
    intervals.to_csv(out_dir / "bootstrap_confidence_intervals.csv", index=False)
    (out_dir / "bootstrap_summary.json").write_text(
        json.dumps(summarize_bootstrap(intervals, args.resamples, args.seed, args.min_class_n), indent=2),
        encoding="utf-8",
    )
    print(f"Saved bootstrap confidence intervals -> {out_dir}")


if __name__ == "__main__":
    main()
