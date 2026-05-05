from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze disagreement across eval signals.")
    parser.add_argument("--results", default=str(DEFAULT_OUT_DIR / "eval_results.csv"))
    parser.add_argument("--ragas", default=str(DEFAULT_OUT_DIR / "ragas_system_comparison.csv"))
    parser.add_argument("--judge", default=str(DEFAULT_OUT_DIR / "pairwise_llm_judge.csv"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--low-confidence-threshold", type=float, default=0.7)
    return parser.parse_args()


def _clean_winner(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"graph", "vector", "tie", "neither"}:
        return raw
    return "missing"


def build_metric_disagreement(
    results: pd.DataFrame,
    ragas: pd.DataFrame,
    judge: pd.DataFrame,
    low_confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    base = results[["id", "question", "query_class", "winner", "failure_mode"]].rename(
        columns={"winner": "deterministic_winner"}
    )
    if not ragas.empty and "id" in ragas.columns:
        base = base.merge(ragas[["id", "ragas_winner"]], on="id", how="left")
    else:
        base["ragas_winner"] = "missing"
    if not judge.empty and "id" in judge.columns:
        keep = [col for col in ["id", "winner", "confidence", "rationale"] if col in judge.columns]
        base = base.merge(judge[keep].rename(columns={"winner": "llm_judge_winner"}), on="id", how="left")
    else:
        base["llm_judge_winner"] = "missing"
        base["confidence"] = 0.0
        base["rationale"] = ""

    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        deterministic = _clean_winner(row.get("deterministic_winner"))
        ragas_winner = _clean_winner(row.get("ragas_winner"))
        judge_winner = _clean_winner(row.get("llm_judge_winner"))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        flags: list[str] = []
        if deterministic != "missing" and judge_winner != "missing" and deterministic != judge_winner:
            flags.append("deterministic_vs_judge")
        if ragas_winner != "missing" and judge_winner != "missing" and ragas_winner != judge_winner:
            flags.append("ragas_vs_judge")
        if deterministic != "missing" and ragas_winner != "missing" and deterministic != ragas_winner:
            flags.append("deterministic_vs_ragas")
        if deterministic == "tie" and judge_winner in {"graph", "vector"} and confidence >= 0.8:
            flags.append("metric_tie_judge_prefers_system")
        if confidence and confidence < low_confidence_threshold:
            flags.append("low_judge_confidence")
        rows.append(
            {
                "id": row["id"],
                "question": row.get("question", ""),
                "query_class": row.get("query_class", ""),
                "deterministic_winner": deterministic,
                "ragas_winner": ragas_winner,
                "llm_judge_winner": judge_winner,
                "judge_confidence": round(confidence, 4),
                "disagreement_flags": ";".join(flags) if flags else "none",
                "has_disagreement": bool(flags and flags != ["low_judge_confidence"]),
                "failure_mode": row.get("failure_mode", ""),
                "judge_rationale": row.get("rationale", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_disagreement(disagreement: pd.DataFrame) -> dict[str, object]:
    if disagreement.empty:
        return {
            "rows": 0,
            "rows_with_disagreement": 0,
            "deterministic_vs_ragas_conflicts": 0,
            "deterministic_vs_judge_conflicts": 0,
            "ragas_vs_judge_conflicts": 0,
            "low_judge_confidence_rows": 0,
            "judge_confidence_buckets": {"high": 0, "medium": 0, "low": 0, "missing": 0},
            "deterministic_winner_counts": {},
            "ragas_winner_counts": {},
            "llm_judge_winner_counts": {},
        }
    flags: Counter[str] = Counter()
    for raw in disagreement["disagreement_flags"]:
        for flag in str(raw).split(";"):
            if flag and flag != "none":
                flags[flag] += 1
    confidence = pd.to_numeric(disagreement["judge_confidence"], errors="coerce")
    buckets = {
        "high": int((confidence >= 0.8).sum()),
        "medium": int(((confidence >= 0.5) & (confidence < 0.8)).sum()),
        "low": int(((confidence >= 0.0) & (confidence < 0.5)).sum()),
        "missing": int(confidence.isna().sum()),
    }
    return {
        "rows": int(len(disagreement)),
        "rows_with_disagreement": int(disagreement["has_disagreement"].sum()),
        "deterministic_vs_ragas_conflicts": flags.get("deterministic_vs_ragas", 0),
        "deterministic_vs_judge_conflicts": flags.get("deterministic_vs_judge", 0),
        "ragas_vs_judge_conflicts": flags.get("ragas_vs_judge", 0),
        "low_judge_confidence_rows": flags.get("low_judge_confidence", 0),
        "judge_confidence_buckets": buckets,
        "deterministic_winner_counts": disagreement["deterministic_winner"].value_counts().to_dict(),
        "ragas_winner_counts": disagreement["ragas_winner"].value_counts().to_dict(),
        "llm_judge_winner_counts": disagreement["llm_judge_winner"].value_counts().to_dict(),
    }


def main() -> None:
    args = _parse_args()
    results = pd.read_csv(args.results)
    ragas = pd.read_csv(args.ragas) if Path(args.ragas).exists() else pd.DataFrame()
    judge = pd.read_csv(args.judge) if Path(args.judge).exists() else pd.DataFrame()
    disagreement = build_metric_disagreement(results, ragas, judge, args.low_confidence_threshold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    disagreement.to_csv(out_dir / "metric_disagreement_analysis.csv", index=False)
    (out_dir / "metric_disagreement_summary.json").write_text(
        json.dumps(summarize_disagreement(disagreement), indent=2), encoding="utf-8"
    )
    print(f"Saved metric disagreement analysis -> {out_dir}")


if __name__ == "__main__":
    main()
