from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "data" / "eval" / "honeywell_hard_labels_28.csv"
DEFAULT_GRAPH = ROOT / "outputs" / "predictions" / "honeywell_hard_labels_28_graphrag.csv"
DEFAULT_VECTOR = ROOT / "outputs" / "predictions" / "vector_rag_hard_labels_28_predictions.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"

ERROR_PATTERNS = [
    "error code",
    "rate limit",
    "rate_limit",
    "authenticationerror",
    "apierror",
    "badrequesterror",
    "timeout",
    "exception",
    "traceback",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prediction output quality before scoring.")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--graph-predictions", default=str(DEFAULT_GRAPH))
    parser.add_argument("--vector-predictions", default=str(DEFAULT_VECTOR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when catastrophic output issues are found.",
    )
    return parser.parse_args()


def _parse_json_cell(value: object, fallback: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    raw = str(value).strip()
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _context_texts(value: object) -> list[str]:
    parsed = _parse_json_cell(value, [])
    if not isinstance(parsed, list):
        return []
    texts: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            texts.append(text)
    return texts


def _has_error(answer: object) -> bool:
    text = str(answer or "").lower()
    return any(pattern in text for pattern in ERROR_PATTERNS)


def _is_not_found(answer: object) -> bool:
    text = str(answer or "").lower()
    return "not found in provided context" in text or text.strip() in {
        "not found",
        "i do not know",
        "i don't know",
    }


def _duplicate_context_count(contexts: list[str]) -> int:
    normalized = [re.sub(r"\s+", " ", context.strip().lower()) for context in contexts]
    counts = Counter(normalized)
    return sum(count - 1 for count in counts.values() if count > 1)


def _audit_system(benchmark: pd.DataFrame, predictions: pd.DataFrame, system: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_by_id = {str(row["id"]): row for _, row in predictions.iterrows()}
    for _, bench_row in benchmark.iterrows():
        row_id = str(bench_row["id"])
        pred = pred_by_id.get(row_id)
        if pred is None:
            rows.append(
                {
                    "id": row_id,
                    "system": system,
                    "query_class": bench_row.get("query_class", ""),
                    "answer_empty": True,
                    "answer_error": False,
                    "answer_not_found": False,
                    "missing_prediction": True,
                    "context_count": 0,
                    "context_empty": True,
                    "duplicate_context_count": 0,
                    "answer_length_chars": 0,
                    "quality_flags": "missing_prediction;answer_empty;context_empty",
                }
            )
            continue

        answer = str(pred.get("answer", "") or "").strip()
        contexts = _context_texts(pred.get("retrieved_contexts", ""))
        flags: list[str] = []
        answer_empty = not answer
        answer_error = _has_error(answer)
        answer_not_found = _is_not_found(answer)
        context_empty = len(contexts) == 0
        duplicate_count = _duplicate_context_count(contexts)

        if answer_empty:
            flags.append("answer_empty")
        if answer_error:
            flags.append("answer_error")
        if answer_not_found:
            flags.append("answer_not_found")
        if context_empty:
            flags.append("context_empty")
        if duplicate_count:
            flags.append("duplicate_contexts")

        rows.append(
            {
                "id": row_id,
                "system": system,
                "query_class": bench_row.get("query_class", ""),
                "answer_empty": answer_empty,
                "answer_error": answer_error,
                "answer_not_found": answer_not_found,
                "missing_prediction": False,
                "context_count": len(contexts),
                "context_empty": context_empty,
                "duplicate_context_count": duplicate_count,
                "answer_length_chars": len(answer),
                "quality_flags": ";".join(flags),
            }
        )
    return rows


def _system_summary(rows: pd.DataFrame, system: str) -> dict[str, Any]:
    subset = rows[rows["system"] == system]
    total = max(1, len(subset))
    return {
        "rows": int(len(subset)),
        "missing_predictions": int(subset["missing_prediction"].sum()),
        "empty_answers": int(subset["answer_empty"].sum()),
        "error_answers": int(subset["answer_error"].sum()),
        "not_found_answers": int(subset["answer_not_found"].sum()),
        "empty_context_rows": int(subset["context_empty"].sum()),
        "duplicate_contexts": int(subset["duplicate_context_count"].sum()),
        "avg_context_count": round(float(subset["context_count"].mean()), 4) if len(subset) else 0,
        "avg_answer_length_chars": round(float(subset["answer_length_chars"].mean()), 4)
        if len(subset)
        else 0,
        "error_answer_rate": round(float(subset["answer_error"].sum()) / total, 4),
        "empty_answer_rate": round(float(subset["answer_empty"].sum()) / total, 4),
    }


def run_audit(
    benchmark_path: Path,
    graph_path: Path,
    vector_path: Path,
    out_dir: Path,
    strict: bool = False,
) -> dict[str, Any]:
    benchmark = pd.read_csv(benchmark_path)
    graph = pd.read_csv(graph_path)
    vector = pd.read_csv(vector_path)

    rows = _audit_system(benchmark, graph, "graph") + _audit_system(benchmark, vector, "vector")
    audit_df = pd.DataFrame(rows)
    graph_summary = _system_summary(audit_df, "graph")
    vector_summary = _system_summary(audit_df, "vector")

    catastrophic = []
    for system, summary in (("graph", graph_summary), ("vector", vector_summary)):
        if summary["rows"] and summary["missing_predictions"] == summary["rows"]:
            catastrophic.append(f"{system}: all predictions missing")
        if summary["rows"] and summary["empty_answers"] == summary["rows"]:
            catastrophic.append(f"{system}: all answers empty")
        if summary["rows"] and summary["error_answers"] == summary["rows"]:
            catastrophic.append(f"{system}: all answers look like runtime/API errors")

    summary = {
        "rows": int(len(benchmark)),
        "systems": {
            "graph": graph_summary,
            "vector": vector_summary,
        },
        "catastrophic_issues": catastrophic,
        "strict": strict,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_dir / "output_quality_audit.csv", index=False)
    (out_dir / "output_quality_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if strict and catastrophic:
        raise RuntimeError("Output quality audit failed: " + " | ".join(catastrophic))
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_audit(
        Path(args.benchmark),
        Path(args.graph_predictions),
        Path(args.vector_predictions),
        Path(args.out_dir),
        strict=args.strict,
    )
    print(f"Saved output quality audit -> {Path(args.out_dir) / 'output_quality_audit.json'}")
    if summary["catastrophic_issues"]:
        print("Output quality warnings: " + " | ".join(summary["catastrophic_issues"]))


if __name__ == "__main__":
    main()
