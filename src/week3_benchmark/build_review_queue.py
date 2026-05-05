from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_AUDIT = ROOT / "outputs" / "eval_outputs" / "output_quality_audit.csv"
DEFAULT_OUT = ROOT / "outputs" / "eval_outputs" / "review_queue.csv"
HARD_QUERY_CLASSES = {"multi_hop", "theme_summary", "relationship_reasoning", "comparison"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human-review queue from eval outputs.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    return parser.parse_args()


def _audit_flags(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    flags: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        value = str(row.get("quality_flags", "") or "")
        if value:
            flags[(str(row["id"]), str(row["system"]))] = value
    return flags


def build_review_queue(results: pd.DataFrame, audit_flags: dict[tuple[str, str], str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in results.iterrows():
        reasons: list[str] = []
        row_id = str(row["id"])
        query_class = str(row.get("query_class", "") or "")
        winner = str(row.get("winner", "") or "")

        if winner == "neither":
            reasons.append("neither_system_won")
        if query_class in HARD_QUERY_CLASSES and winner == "vector":
            reasons.append("vector_won_hard_query")
        if float(row.get("graph_entity_recall", 0)) >= 0.75 and float(
            row.get("graph_answer_correctness", 0)
        ) < 0.4:
            reasons.append("graph_retrieved_entities_but_answer_weak")
        if str(row.get("failure_mode", "")) in {
            "retrieval_missed_entities",
            "low_context_precision",
            "unfaithful_answer",
            "irrelevant_or_dodged_answer",
        }:
            reasons.append(str(row["failure_mode"]))

        graph_flags = audit_flags.get((row_id, "graph"), "")
        vector_flags = audit_flags.get((row_id, "vector"), "")
        if graph_flags:
            reasons.append(f"graph_output_flags:{graph_flags}")
        if vector_flags:
            reasons.append(f"vector_output_flags:{vector_flags}")

        if not reasons:
            continue
        rows.append(
            {
                "id": row_id,
                "query_class": query_class,
                "winner": winner,
                "review_reasons": ";".join(dict.fromkeys(reasons)),
                "failure_mode": row.get("failure_mode", ""),
                "graph_score": row.get("graph_score", ""),
                "vector_score": row.get("vector_score", ""),
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", ""),
                "graph_answer": row.get("graph_answer", ""),
                "vector_answer": row.get("vector_answer", ""),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "query_class",
                "winner",
                "review_reasons",
                "failure_mode",
                "graph_score",
                "vector_score",
                "question",
                "ground_truth",
                "graph_answer",
                "vector_answer",
            ]
        )
    return pd.DataFrame(rows).sort_values(["query_class", "id"])


def main() -> None:
    args = _parse_args()
    results = pd.read_csv(args.results)
    queue = build_review_queue(results, _audit_flags(Path(args.audit)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False)
    print(f"Saved review queue -> {output}")


if __name__ == "__main__":
    main()
