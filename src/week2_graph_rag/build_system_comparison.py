from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def _normalize_answer(text: str) -> str:
    lowered = str(text).strip().lower()
    lowered = lowered.replace("vrms", "volts")
    lowered = lowered.replace("vdc", "volts dc")
    lowered = lowered.replace("1 hz", "1 flash per second")
    lowered = lowered.replace("25.0", "25")
    lowered = lowered.replace("4,000", "4000")
    lowered = lowered.replace("x", " by ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_reference(answer: str, reference: str) -> bool:
    normalized_answer = _normalize_answer(answer)
    normalized_reference = _normalize_answer(reference)
    if not normalized_answer or not normalized_reference:
        return False
    return normalized_reference in normalized_answer or normalized_answer in normalized_reference


def _winner(row: pd.Series) -> str:
    graph_ok = bool(row["graph_matches_reference"])
    baseline_ok = bool(row["baseline_matches_reference"])
    if graph_ok and baseline_ok:
        return "tie"
    if graph_ok:
        return "graph"
    if baseline_ok:
        return "baseline"
    return "neither"


def main() -> None:
    graph_csv = Path(os.getenv("GRAPH_PREDICTIONS_CSV", "outputs/graph_rag/week2_graphrag_predictions.csv"))
    baseline_csv = Path(
        os.getenv("BASELINE_PREDICTIONS_CSV", "outputs/comparison/week2_structured_baseline_predictions.csv")
    )
    output_csv = Path(
        os.getenv("COMPARISON_CSV", "outputs/comparison/week2_graph_vs_structured_comparison.csv")
    )
    summary_txt = Path(
        os.getenv("COMPARISON_SUMMARY_TXT", "outputs/comparison/week2_graph_vs_structured_summary.txt")
    )

    if not graph_csv.exists():
        raise FileNotFoundError(f"Missing GraphRAG predictions CSV: {graph_csv}")
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Missing baseline predictions CSV: {baseline_csv}")

    graph_df = pd.read_csv(graph_csv)
    baseline_df = pd.read_csv(baseline_csv)

    keep_cols = [
        "id",
        "question",
        "ground_truth",
        "category",
        "query_class",
        "reasoning_type",
        "source_product",
        "source_split",
    ]
    graph_trim = graph_df[[col for col in keep_cols if col in graph_df.columns] + ["answer"]].rename(
        columns={"answer": "graph_answer"}
    )
    baseline_trim = baseline_df[["id", "answer", "answer_mode", "retrieval_mode", "matched_field_name"]].rename(
        columns={"answer": "baseline_answer"}
    )
    merged = graph_trim.merge(baseline_trim, on="id", how="inner")

    merged["graph_matches_reference"] = merged.apply(
        lambda row: _contains_reference(row["graph_answer"], row["ground_truth"]), axis=1
    )
    merged["baseline_matches_reference"] = merged.apply(
        lambda row: _contains_reference(row["baseline_answer"], row["ground_truth"]), axis=1
    )
    merged["winner_heuristic"] = merged.apply(_winner, axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    counts = merged["winner_heuristic"].value_counts().to_dict()
    lines = [
        f"rows={len(merged)}",
        f"graph={counts.get('graph', 0)}",
        f"baseline={counts.get('baseline', 0)}",
        f"tie={counts.get('tie', 0)}",
        f"neither={counts.get('neither', 0)}",
    ]
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved comparison CSV -> {output_csv}")
    print(f"Saved comparison summary -> {summary_txt}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
