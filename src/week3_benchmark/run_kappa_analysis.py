from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL = ROOT / "outputs" / "comparison" / "honeywell_hard_labels_28_manual_review.csv"
DEFAULT_LLM_JUDGE = ROOT / "outputs" / "comparison" / "honeywell_hard_labels_28_pairwise_judge.csv"
DEFAULT_WEEK3 = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_OUT_CSV = ROOT / "outputs" / "eval_outputs" / "kappa_analysis.csv"
DEFAULT_OUT_JSON = ROOT / "outputs" / "eval_outputs" / "kappa_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's Kappa agreement checks for Week 3 evaluation."
    )
    parser.add_argument("--manual-review", default=str(DEFAULT_MANUAL))
    parser.add_argument("--llm-judge", default=str(DEFAULT_LLM_JUDGE))
    parser.add_argument("--week3-results", default=str(DEFAULT_WEEK3))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    return parser.parse_args()


def _normalize_label(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"graph", "graphrag", "graph_rag"}:
        return "graph"
    if raw in {"baseline", "structured", "structured_baseline", "vector", "vector_rag"}:
        return "baseline_or_vector"
    if raw in {"tie", "both"}:
        return "tie"
    if raw in {"neither", "none", "no_winner"}:
        return "neither"
    return raw or "missing"


def _kappa(left: pd.Series, right: pd.Series) -> float:
    return round(float(cohen_kappa_score(left, right)), 4)


def _agreement_rate(left: pd.Series, right: pd.Series) -> float:
    return round(float((left == right).mean()), 4)


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def main() -> None:
    args = _parse_args()
    manual_path = Path(args.manual_review)
    llm_path = Path(args.llm_judge)
    week3_path = Path(args.week3_results)
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)

    manual = _load_csv(manual_path, "manual review CSV")
    llm = _load_csv(llm_path, "LLM judge CSV")
    week3 = _load_csv(week3_path, "Week 3 results CSV")

    if "manual_verdict" not in manual.columns:
        raise ValueError("Manual review CSV must contain manual_verdict")
    if "winner" not in llm.columns:
        raise ValueError("LLM judge CSV must contain winner")
    if "winner" not in week3.columns:
        raise ValueError("Week 3 results CSV must contain winner")

    base = manual[["id", "question", "query_class", "manual_verdict"]].copy()
    base["human_label"] = base["manual_verdict"].apply(_normalize_label)

    llm_trim = llm[["id", "winner"]].rename(columns={"winner": "llm_judge_winner"})
    llm_trim["llm_judge_label"] = llm_trim["llm_judge_winner"].apply(_normalize_label)

    week3_trim = week3[["id", "winner", "failure_mode"]].rename(
        columns={"winner": "week3_metric_winner"}
    )
    week3_trim["week3_metric_label"] = week3_trim["week3_metric_winner"].apply(_normalize_label)

    merged = base.merge(llm_trim, on="id", how="left").merge(week3_trim, on="id", how="left")
    merged["human_vs_llm_agree"] = merged["human_label"] == merged["llm_judge_label"]
    merged["human_vs_week3_agree"] = merged["human_label"] == merged["week3_metric_label"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    summary = {
        "manual_review_csv": str(manual_path),
        "llm_judge_csv": str(llm_path),
        "week3_results_csv": str(week3_path),
        "rows": int(len(merged)),
        "human_vs_llm_judge": {
            "cohens_kappa": _kappa(merged["human_label"], merged["llm_judge_label"]),
            "agreement_rate": _agreement_rate(merged["human_label"], merged["llm_judge_label"]),
            "label_counts": {
                "human": merged["human_label"].value_counts().to_dict(),
                "llm_judge": merged["llm_judge_label"].value_counts().to_dict(),
            },
        },
        "human_vs_week3_metric": {
            "cohens_kappa": _kappa(merged["human_label"], merged["week3_metric_label"]),
            "agreement_rate": _agreement_rate(
                merged["human_label"], merged["week3_metric_label"]
            ),
            "label_counts": {
                "human": merged["human_label"].value_counts().to_dict(),
                "week3_metric": merged["week3_metric_label"].value_counts().to_dict(),
            },
        },
        "note": (
            "Manual review and LLM judge compare GraphRAG against the structured baseline. "
            "Week 3 metric winners compare GraphRAG against the top-k vector baseline, so that "
            "kappa is an agreement sanity check rather than a strict same-task rater comparison."
        ),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved kappa rows -> {out_csv}")
    print(f"Saved kappa summary -> {out_json}")
    print("Human vs LLM judge kappa:", summary["human_vs_llm_judge"]["cohens_kappa"])
    print("Human vs Week 3 metric kappa:", summary["human_vs_week3_metric"]["cohens_kappa"])


if __name__ == "__main__":
    main()
