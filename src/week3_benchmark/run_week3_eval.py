from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

from src.week3_benchmark.metrics_metadata import METRICS_VERSION, file_sha256, git_sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "data" / "eval" / "honeywell_hard_labels_28.csv"
DEFAULT_GRAPH = ROOT / "outputs" / "predictions" / "honeywell_hard_labels_28_graphrag.csv"
DEFAULT_VECTOR = ROOT / "outputs" / "predictions" / "vector_rag_hard_labels_28_predictions.csv"
DEFAULT_MANUAL = ROOT / "outputs" / "comparison" / "honeywell_graph_vs_vector_24_manual_comparison.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"
DEFAULT_REPORT = ROOT / "outputs" / "eval_outputs" / "benchmarking_comparative_analysis.md"

HARD_QUERY_CLASSES = {"multi_hop", "theme_summary", "relationship_reasoning", "comparison"}
REQUIRED_BENCHMARK_COLUMNS = {"id", "question", "ground_truth", "contexts", "query_class"}
REQUIRED_PREDICTION_COLUMNS = {"id", "answer", "retrieved_contexts"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 3 GraphRAG vs vector RAG benchmark evaluation."
    )
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--graph-predictions", default=str(DEFAULT_GRAPH))
    parser.add_argument("--vector-predictions", default=str(DEFAULT_VECTOR))
    parser.add_argument("--manual-review", default=str(DEFAULT_MANUAL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--no-generate-vector",
        action="store_true",
        help="Fail if the 28-row vector prediction file is missing instead of generating it.",
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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _tokens(text: object) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", _normalize_text(text))
        if len(tok) > 2 and tok not in STOPWORDS
    }


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


def _expected_signals(row: pd.Series) -> list[str]:
    signals: list[str] = []
    source_product = str(row.get("source_product", "") or "")
    for item in re.split(r"[|,;/]+", source_product):
        cleaned = item.strip()
        if cleaned:
            signals.append(cleaned)

    source_fields = _parse_json_cell(row.get("source_fields", ""), [])
    if isinstance(source_fields, list):
        for field in source_fields:
            cleaned = str(field).replace("_", " ").strip()
            if cleaned:
                signals.append(cleaned)

    seen: set[str] = set()
    out: list[str] = []
    for signal in signals:
        key = _normalize_text(signal)
        if key and key not in seen:
            seen.add(key)
            out.append(signal)
    return out


def _signal_found(signal: str, text: str) -> bool:
    normalized_signal = _normalize_text(signal).replace("-", " ")
    normalized_text = _normalize_text(text).replace("-", " ")
    if normalized_signal in normalized_text:
        return True
    signal_tokens = _tokens(normalized_signal)
    if not signal_tokens:
        return False
    text_tokens = _tokens(normalized_text)
    return len(signal_tokens & text_tokens) / len(signal_tokens) >= 0.8


def _entity_recall(expected: list[str], contexts: list[str]) -> tuple[float, list[str]]:
    if not expected:
        return 1.0, []
    joined_context = "\n".join(contexts)
    found = [signal for signal in expected if _signal_found(signal, joined_context)]
    return round(len(found) / len(expected), 4), found


def _context_precision(
    retrieved_contexts: list[str],
    gold_contexts: list[str],
    ground_truth: str,
    expected: list[str],
) -> float:
    if not retrieved_contexts:
        return 0.0
    gold_text = "\n".join(gold_contexts + [ground_truth])
    gold_tokens = _tokens(gold_text)
    useful = 0
    for context in retrieved_contexts:
        context_tokens = _tokens(context)
        overlap = len(context_tokens & gold_tokens) / max(1, len(gold_tokens))
        has_entity = any(_signal_found(signal, context) for signal in expected)
        if overlap >= 0.12 or has_entity:
            useful += 1
    return round(useful / len(retrieved_contexts), 4)


def _faithfulness(answer: str, retrieved_contexts: list[str]) -> float:
    normalized_answer = _normalize_text(answer)
    if not normalized_answer or "not found in provided context" in normalized_answer:
        return 0.0
    context_tokens = _tokens("\n".join(retrieved_contexts))
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    support = len(answer_tokens & context_tokens) / len(answer_tokens)
    return round(min(1.0, support), 4)


def _answer_relevancy(question: str, answer: str, ground_truth: str) -> float:
    normalized_answer = _normalize_text(answer)
    if not normalized_answer or "not found in provided context" in normalized_answer:
        return 0.0
    answer_tokens = _tokens(answer)
    intent_tokens = _tokens(question) | _tokens(ground_truth)
    if not intent_tokens:
        return 0.0
    overlap = len(answer_tokens & intent_tokens) / max(1, len(intent_tokens))
    return round(min(1.0, overlap * 1.5), 4)


def _answer_correctness(answer: str, ground_truth: str) -> float:
    normalized_answer = _normalize_text(answer)
    if not normalized_answer or "not found in provided context" in normalized_answer:
        return 0.0
    answer_tokens = _tokens(answer)
    truth_tokens = _tokens(ground_truth)
    if not truth_tokens:
        return 0.0
    return round(len(answer_tokens & truth_tokens) / len(truth_tokens), 4)


def _system_score(prefix: str, row: pd.Series) -> float:
    return round(
        0.35 * float(row[f"{prefix}_answer_correctness"])
        + 0.20 * float(row[f"{prefix}_faithfulness"])
        + 0.15 * float(row[f"{prefix}_answer_relevancy"])
        + 0.15 * float(row[f"{prefix}_entity_recall"])
        + 0.15 * float(row[f"{prefix}_context_precision"]),
        4,
    )


def _winner(row: pd.Series) -> str:
    graph_score = float(row["graph_score"])
    vector_score = float(row["vector_score"])
    if graph_score < 0.35 and vector_score < 0.35:
        return "neither"
    if abs(graph_score - vector_score) <= 0.05:
        return "tie"
    return "graph" if graph_score > vector_score else "vector"


def _failure_mode(row: pd.Series) -> str:
    if row["winner"] == "tie":
        return "both_correct_or_tie"
    winning_prefix = "graph" if row["winner"] == "graph" else "vector"
    losing_prefix = "vector" if winning_prefix == "graph" else "graph"
    if row["winner"] == "neither":
        losing_prefix = "graph" if row["graph_score"] <= row["vector_score"] else "vector"

    if float(row[f"{losing_prefix}_entity_recall"]) < 0.5:
        return "retrieval_missed_entities"
    if float(row[f"{losing_prefix}_context_precision"]) < 0.4:
        return "low_context_precision"
    if float(row[f"{losing_prefix}_faithfulness"]) < 0.5:
        return "unfaithful_answer"
    if float(row[f"{losing_prefix}_answer_relevancy"]) < 0.5:
        return "irrelevant_or_dodged_answer"
    return "incomplete_synthesis"


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _validate_id_column(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise ValueError(f"{label} must contain at least one row")
    if df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        raise ValueError(f"{label} contains blank id values")
    duplicate_ids = sorted(df.loc[df["id"].duplicated(), "id"].astype(str).unique())
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"{label} contains duplicate id values: {preview}")


def _validate_json_array_column(df: pd.DataFrame, column: str, label: str) -> None:
    bad_rows: list[str] = []
    for idx, value in df[column].items():
        parsed = _parse_json_cell(value, None)
        if not isinstance(parsed, list):
            row_id = df.at[idx, "id"] if "id" in df.columns else idx
            bad_rows.append(str(row_id))
    if bad_rows:
        preview = ", ".join(bad_rows[:5])
        raise ValueError(f"{label} column '{column}' must contain JSON arrays; bad ids: {preview}")


def _validate_prediction_ids(
    benchmark: pd.DataFrame, predictions: pd.DataFrame, label: str
) -> None:
    benchmark_ids = set(benchmark["id"].astype(str))
    prediction_ids = set(predictions["id"].astype(str))
    missing = sorted(benchmark_ids - prediction_ids)
    extra = sorted(prediction_ids - benchmark_ids)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{label} missing predictions for dataset ids: {preview}")
    if extra:
        preview = ", ".join(extra[:5])
        raise ValueError(f"{label} contains ids not present in dataset: {preview}")


def _validate_inputs(
    benchmark: pd.DataFrame, graph_df: pd.DataFrame, vector_df: pd.DataFrame
) -> None:
    _validate_columns(benchmark, REQUIRED_BENCHMARK_COLUMNS, "benchmark CSV")
    _validate_columns(graph_df, REQUIRED_PREDICTION_COLUMNS, "GraphRAG predictions CSV")
    _validate_columns(vector_df, REQUIRED_PREDICTION_COLUMNS, "vector RAG predictions CSV")

    _validate_id_column(benchmark, "benchmark CSV")
    _validate_id_column(graph_df, "GraphRAG predictions CSV")
    _validate_id_column(vector_df, "vector RAG predictions CSV")

    _validate_json_array_column(benchmark, "contexts", "benchmark CSV")
    _validate_json_array_column(graph_df, "retrieved_contexts", "GraphRAG predictions CSV")
    _validate_json_array_column(vector_df, "retrieved_contexts", "vector RAG predictions CSV")

    _validate_prediction_ids(benchmark, graph_df, "GraphRAG predictions CSV")
    _validate_prediction_ids(benchmark, vector_df, "vector RAG predictions CSV")


def _generate_vector_predictions(benchmark: pd.DataFrame, output_csv: Path) -> None:
    from src.week1_vector_rag.rag_pipeline import answer_question

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid4())
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(benchmark.iterrows(), total=len(benchmark), desc="Vector RAG baseline"):
        output = answer_question(str(row["question"]), metadata_filter={})
        record = {
            "id": row["id"],
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "contexts": _parse_json_cell(row["contexts"], []),
            "category": row.get("category", ""),
            "answer": output["answer"],
            "retrieved_contexts": output["retrieved_contexts"],
            "answer_mode": output.get("answer_mode", "llm"),
            "latency_ms": output["latency_ms"],
            "model_name": output["model_name"],
            "timestamp_utc": output["timestamp_utc"],
            "run_id": run_id,
            "query_class": row.get("query_class", ""),
            "reasoning_type": row.get("reasoning_type", ""),
            "source_product": row.get("source_product", ""),
            "source_fields": row.get("source_fields", ""),
            "source_split": row.get("source_split", ""),
        }
        rows.append(record)
    out_df = pd.DataFrame(rows)
    for col in ("contexts", "retrieved_contexts"):
        out_df[col] = out_df[col].apply(_json_dumps)
    out_df.to_csv(output_csv, index=False)


def _load_manual_review(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id", "manual_verdict", "manual_notes"])
    df = pd.read_csv(path)
    keep = [col for col in ["id", "manual_verdict", "notes", "manual_notes"] if col in df.columns]
    out = df[keep].copy()
    if "notes" in out.columns and "manual_notes" not in out.columns:
        out = out.rename(columns={"notes": "manual_notes"})
    return out


def _build_results(
    benchmark: pd.DataFrame,
    graph_df: pd.DataFrame,
    vector_df: pd.DataFrame,
    manual_df: pd.DataFrame,
) -> pd.DataFrame:
    graph = graph_df[["id", "answer", "retrieved_contexts"]].rename(
        columns={"answer": "graph_answer", "retrieved_contexts": "graph_retrieved_contexts"}
    )
    vector = vector_df[["id", "answer", "retrieved_contexts"]].rename(
        columns={"answer": "vector_answer", "retrieved_contexts": "vector_retrieved_contexts"}
    )
    merged = benchmark.merge(graph, on="id", how="left").merge(vector, on="id", how="left")
    if not manual_df.empty:
        merged = merged.merge(manual_df, on="id", how="left")

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        gold_contexts = _parse_json_cell(row.get("contexts", ""), [])
        if not isinstance(gold_contexts, list):
            gold_contexts = []
        gold_contexts = [str(item) for item in gold_contexts]
        expected = _expected_signals(row)
        graph_contexts = _context_texts(row.get("graph_retrieved_contexts", ""))
        vector_contexts = _context_texts(row.get("vector_retrieved_contexts", ""))

        out = row.to_dict()
        out["expected_entities"] = expected
        out["graph_retrieved_context_count"] = len(graph_contexts)
        out["vector_retrieved_context_count"] = len(vector_contexts)

        for prefix, answer, contexts in (
            ("graph", str(row.get("graph_answer", "") or ""), graph_contexts),
            ("vector", str(row.get("vector_answer", "") or ""), vector_contexts),
        ):
            entity_recall, found = _entity_recall(expected, contexts)
            out[f"{prefix}_entity_recall"] = entity_recall
            out[f"{prefix}_entities_found"] = found
            out[f"{prefix}_context_precision"] = _context_precision(
                contexts, gold_contexts, str(row["ground_truth"]), expected
            )
            out[f"{prefix}_faithfulness"] = _faithfulness(answer, contexts)
            out[f"{prefix}_answer_relevancy"] = _answer_relevancy(
                str(row["question"]), answer, str(row["ground_truth"])
            )
            out[f"{prefix}_answer_correctness"] = _answer_correctness(
                answer, str(row["ground_truth"])
            )
        out["graph_score"] = _system_score("graph", pd.Series(out))
        out["vector_score"] = _system_score("vector", pd.Series(out))
        out["winner"] = _winner(pd.Series(out))
        out["failure_mode"] = _failure_mode(pd.Series(out))
        rows.append(out)

    result = pd.DataFrame(rows)
    json_cols = [
        "contexts",
        "expected_entities",
        "graph_entities_found",
        "vector_entities_found",
        "graph_retrieved_contexts",
        "vector_retrieved_contexts",
    ]
    for col in json_cols:
        if col in result.columns:
            result[col] = result[col].apply(_json_dumps)
    return result


def _summarize_by_query_class(results: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for query_class, group in results.groupby("query_class", dropna=False):
        counts = Counter(group["winner"])
        grouped.append(
            {
                "query_class": query_class,
                "rows": len(group),
                "graph_wins": counts.get("graph", 0),
                "vector_wins": counts.get("vector", 0),
                "ties": counts.get("tie", 0),
                "neither": counts.get("neither", 0),
                "graph_win_rate": round(counts.get("graph", 0) / len(group), 4),
                "vector_win_rate": round(counts.get("vector", 0) / len(group), 4),
                "graph_entity_recall_mean": round(float(group["graph_entity_recall"].mean()), 4),
                "vector_entity_recall_mean": round(float(group["vector_entity_recall"].mean()), 4),
                "graph_context_precision_mean": round(
                    float(group["graph_context_precision"].mean()), 4
                ),
                "vector_context_precision_mean": round(
                    float(group["vector_context_precision"].mean()), 4
                ),
                "graph_faithfulness_mean": round(float(group["graph_faithfulness"].mean()), 4),
                "vector_faithfulness_mean": round(float(group["vector_faithfulness"].mean()), 4),
                "graph_answer_relevancy_mean": round(
                    float(group["graph_answer_relevancy"].mean()), 4
                ),
                "vector_answer_relevancy_mean": round(
                    float(group["vector_answer_relevancy"].mean()), 4
                ),
            }
        )
    return pd.DataFrame(grouped).sort_values("query_class")


def _metric_summary(results: pd.DataFrame, summary_by_class: pd.DataFrame) -> dict[str, Any]:
    counts = Counter(results["winner"])
    hard = results[results["query_class"].isin(HARD_QUERY_CLASSES)]
    hard_counts = Counter(hard["winner"])
    return {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "rows": int(len(results)),
        "winner_counts": dict(counts),
        "graph_win_rate": round(counts.get("graph", 0) / max(1, len(results)), 4),
        "vector_win_rate": round(counts.get("vector", 0) / max(1, len(results)), 4),
        "hard_query_classes": sorted(HARD_QUERY_CLASSES),
        "hard_query_rows": int(len(hard)),
        "hard_query_winner_counts": dict(hard_counts),
        "hard_query_graph_win_rate": round(hard_counts.get("graph", 0) / max(1, len(hard)), 4),
        "metric_means": {
            "graph_entity_recall": round(float(results["graph_entity_recall"].mean()), 4),
            "vector_entity_recall": round(float(results["vector_entity_recall"].mean()), 4),
            "graph_context_precision": round(float(results["graph_context_precision"].mean()), 4),
            "vector_context_precision": round(float(results["vector_context_precision"].mean()), 4),
            "graph_faithfulness": round(float(results["graph_faithfulness"].mean()), 4),
            "vector_faithfulness": round(float(results["vector_faithfulness"].mean()), 4),
            "graph_answer_relevancy": round(float(results["graph_answer_relevancy"].mean()), 4),
            "vector_answer_relevancy": round(float(results["vector_answer_relevancy"].mean()), 4),
        },
        "failure_mode_counts": dict(Counter(results["failure_mode"])),
        "summary_by_query_class": summary_by_class.to_dict(orient="records"),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rendered = df.copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].apply(lambda value: str(value))
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in rendered.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    report_path: Path,
    results: pd.DataFrame,
    summary_by_class: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    winner_counts = summary["winner_counts"]
    failure_counts = summary["failure_mode_counts"]
    artifact_dir = report_path.parent
    output_audit = _parse_json_cell(
        (artifact_dir / "output_quality_audit.json").read_text(encoding="utf-8")
        if (artifact_dir / "output_quality_audit.json").exists()
        else "{}",
        {},
    )
    graph_audit = _parse_json_cell(
        (artifact_dir / "graph_extraction_summary.json").read_text(encoding="utf-8")
        if (artifact_dir / "graph_extraction_summary.json").exists()
        else "{}",
        {},
    )
    metric_win_rates = (
        pd.read_csv(artifact_dir / "metric_win_rates_overall.csv")
        if (artifact_dir / "metric_win_rates_overall.csv").exists()
        else pd.DataFrame()
    )
    review_queue = (
        pd.read_csv(artifact_dir / "review_queue.csv")
        if (artifact_dir / "review_queue.csv").exists()
        else pd.DataFrame()
    )
    graph_coverage = (
        pd.read_csv(artifact_dir / "graph_coverage_summary_by_class.csv")
        if (artifact_dir / "graph_coverage_summary_by_class.csv").exists()
        else pd.DataFrame()
    )
    graph_coverage_vs_winner = (
        pd.read_csv(artifact_dir / "graph_coverage_vs_winner.csv")
        if (artifact_dir / "graph_coverage_vs_winner.csv").exists()
        else pd.DataFrame()
    )
    graph_coverage_interpretation = _parse_json_cell(
        (artifact_dir / "graph_coverage_interpretation.json").read_text(encoding="utf-8")
        if (artifact_dir / "graph_coverage_interpretation.json").exists()
        else "{}",
        {},
    )
    bootstrap = (
        pd.read_csv(artifact_dir / "bootstrap_confidence_intervals.csv")
        if (artifact_dir / "bootstrap_confidence_intervals.csv").exists()
        else pd.DataFrame()
    )
    ragas_comparison = (
        pd.read_csv(artifact_dir / "ragas_system_comparison.csv")
        if (artifact_dir / "ragas_system_comparison.csv").exists()
        else pd.DataFrame()
    )
    disagreement = (
        pd.read_csv(artifact_dir / "metric_disagreement_analysis.csv")
        if (artifact_dir / "metric_disagreement_analysis.csv").exists()
        else pd.DataFrame()
    )
    disagreement_summary = _parse_json_cell(
        (artifact_dir / "metric_disagreement_summary.json").read_text(encoding="utf-8")
        if (artifact_dir / "metric_disagreement_summary.json").exists()
        else "{}",
        {},
    )
    run_manifest = _parse_json_cell(
        (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
        if (artifact_dir / "run_manifest.json").exists()
        else "{}",
        {},
    )
    difficulty = (
        pd.read_csv(artifact_dir / "difficulty_vs_win_rate.csv")
        if (artifact_dir / "difficulty_vs_win_rate.csv").exists()
        else pd.DataFrame()
    )
    cost_latency = _parse_json_cell(
        (artifact_dir / "cost_latency_summary.json").read_text(encoding="utf-8")
        if (artifact_dir / "cost_latency_summary.json").exists()
        else "{}",
        {},
    )
    dataset_raw = str(results.attrs.get("benchmark_path", "")).strip()
    dataset_path = Path(dataset_raw) if dataset_raw else None
    dataset_file = dataset_path.as_posix() if dataset_path else "not recorded"
    dataset_hash = file_sha256(dataset_path) if dataset_path and dataset_path.is_file() else ""

    class_table = _markdown_table(
        summary_by_class[
            [
                "query_class",
                "rows",
                "graph_wins",
                "vector_wins",
                "ties",
                "neither",
                "graph_win_rate",
                "graph_entity_recall_mean",
                "graph_context_precision_mean",
            ]
        ]
    )

    failure_table = _markdown_table(
        pd.DataFrame(
            [{"failure_mode": key, "count": value} for key, value in failure_counts.items()]
        )
        .sort_values("count", ascending=False)
    )
    metric_win_table = _markdown_table(metric_win_rates)
    graph_coverage_table = _markdown_table(graph_coverage)
    graph_coverage_winner_table = _markdown_table(graph_coverage_vs_winner)
    bootstrap_table = _markdown_table(bootstrap)
    ragas_table = _markdown_table(
        ragas_comparison.head(10)[
            [
                col
                for col in [
                    "id",
                    "query_class",
                    "ragas_winner",
                    "faithfulness_winner",
                    "answer_relevancy_winner",
                    "context_precision_winner",
                ]
                if col in ragas_comparison.columns
            ]
        ]
        if not ragas_comparison.empty
        else ragas_comparison
    )
    disagreement_table = _markdown_table(
        disagreement.head(10)[
            [
                col
                for col in [
                    "id",
                    "query_class",
                    "deterministic_winner",
                    "ragas_winner",
                    "llm_judge_winner",
                    "disagreement_flags",
                ]
                if col in disagreement.columns
            ]
        ]
        if not disagreement.empty
        else disagreement
    )
    difficulty_table = _markdown_table(difficulty)
    review_queue_table = _markdown_table(
        review_queue.head(10)[
            [col for col in ["id", "query_class", "winner", "review_reasons", "question"] if col in review_queue.columns]
        ]
        if not review_queue.empty
        else review_queue
    )
    graph_quality = output_audit.get("systems", {}).get("graph", {})
    vector_quality = output_audit.get("systems", {}).get("vector", {})

    hard_rows = results[results["query_class"].isin(HARD_QUERY_CLASSES)]
    hard_table = _markdown_table(
        hard_rows.groupby("query_class")["winner"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    report = f"""# Week 3 Benchmarking and Comparative Analysis

## Project Goal

This project evaluates GraphRAG by query class and failure mode rather than by a generic average score alone. The main comparison is GraphRAG against a standard top-k vector RAG baseline on the 28-question Honeywell hard-label benchmark.

Main claim: GraphRAG is not universally better than vector RAG; its value is concentrated in thematic, multi-hop, comparison, and relationship-heavy queries.

## Benchmark and Systems

- Benchmark: `data/eval/honeywell_hard_labels_28.csv`
- Rows evaluated: `{len(results)}`
- GraphRAG predictions: `outputs/predictions/honeywell_hard_labels_28_graphrag.csv`
- Vector RAG predictions: `outputs/predictions/vector_rag_hard_labels_28_predictions.csv`

The benchmark contains labeled question classes including direct factual lookup, numeric specifications, list retrieval, comparison, multi-hop synthesis, theme summary, and relationship reasoning.

## Required Metrics

- Context Entities Recall: whether retrieved contexts contain expected products/entities from `source_product` and `source_fields`.
- Context Precision: share of retrieved contexts that overlap gold evidence or contain expected entities.
- Faithfulness: deterministic support proxy measuring whether answer tokens are grounded in retrieved context.
- Answer Relevancy: deterministic proxy measuring whether the answer addresses the question and reference answer.
- Winner and Failure Mode: row-level comparison of GraphRAG and vector RAG.

## Overall Results

| Outcome | Count |
|---|---:|
| GraphRAG wins | {winner_counts.get("graph", 0)} |
| Vector RAG wins | {winner_counts.get("vector", 0)} |
| Ties | {winner_counts.get("tie", 0)} |
| Neither | {winner_counts.get("neither", 0)} |

Overall GraphRAG win rate: `{summary["graph_win_rate"]}`

## Requirements Compliance

| Requirement | Status | Evidence |
|---|---|---|
| Context Entity Recall | Done | `eval_results.csv` |
| Context Precision | Done | `eval_results.csv` |
| Faithfulness | Done | deterministic full-set proxy; RAGAS smoke/full RAGAS when supplied |
| Answer Relevancy | Done | deterministic full-set proxy; RAGAS smoke/full RAGAS when supplied |
| Baseline RAG | Done | vector prediction CSV |
| Win rate by query class | Done | `summary_by_query_class.csv` |
| Failure dataframe | Done | `failure_analysis.csv` |
| Output quality audit | Done | `output_quality_audit.json` |
| Graph audit | Done when Neo4j is reachable | `graph_extraction_summary.json` |

## Metric Tiers

| Tier | Name | Purpose | Primary artifacts |
|---|---|---|---|
| Tier 1 | Deterministic reproducible metrics | Full-set reproducible scoring and win rates | `eval_results.csv`, `metric_summary.json`, `bootstrap_confidence_intervals.csv` |
| Tier 2 | RAGAS LLM metrics | LLM-based faithfulness/relevancy/context scoring when enabled | `ragas_system_comparison.csv` |
| Tier 3 | Pairwise LLM judge | Pairwise preference and confidence when enabled | `pairwise_llm_judge.csv` |
| Tier 4 | Derived graph coverage proxy metrics | Structural coverage over GraphRAG retrieved contexts | `graph_coverage_metrics.csv` |
| Tier 5 | Operational cost/latency estimates | Runtime and token-cost estimates | `cost_latency_summary.json` |

Headline sample size: `n={len(results)}`.

## Output Quality Audit

| System | Error Answers | Empty Answers | Not Found Answers | Empty Context Rows | Avg Context Count |
|---|---:|---:|---:|---:|---:|
| GraphRAG | {graph_quality.get("error_answers", "")} | {graph_quality.get("empty_answers", "")} | {graph_quality.get("not_found_answers", "")} | {graph_quality.get("empty_context_rows", "")} | {graph_quality.get("avg_context_count", "")} |
| Vector RAG | {vector_quality.get("error_answers", "")} | {vector_quality.get("empty_answers", "")} | {vector_quality.get("not_found_answers", "")} | {vector_quality.get("empty_context_rows", "")} | {vector_quality.get("avg_context_count", "")} |

## Graph Audit

- Domain: `{graph_audit.get("domain", "")}`
- Audit status: `{graph_audit.get("status", "missing")}`
- Node count: `{graph_audit.get("node_count", "")}`
- Relationship count: `{graph_audit.get("relationship_count", "")}`
- Claim extraction: `{graph_audit.get("claim_note", "not reported")}`

## Results by Query Class

{class_table}

## Per-Metric Win Rates

{metric_win_table}

## Full RAGAS System Comparison

{ragas_table}

## Metric Disagreement Analysis

{disagreement_table}

## Query Difficulty Analysis

{difficulty_table}

## Graph Coverage Metrics

These are graph coverage proxies, not full proof of graph-path correctness.

- `graph_expected_entity_coverage = expected_entities_found_in_graph_contexts / expected_entities`
- `relation_signal_coverage = source_field_or_relation_signals_found_in_graph_contexts / expected_relation_signals`
- `structural_density = relation_like_context_count / graph_context_count`
- `graph_coverage_score = 0.5 * graph_expected_entity_coverage + 0.3 * relation_signal_coverage + 0.2 * structural_density`

{graph_coverage_table}

Graph coverage by winner:

{graph_coverage_winner_table}

- Run median graph coverage score: `{graph_coverage_interpretation.get("median_graph_coverage_score", "")}`
- High-coverage GraphRAG losses: `{graph_coverage_interpretation.get("high_coverage_graph_losses", "")}`
- Low-coverage GraphRAG losses: `{graph_coverage_interpretation.get("low_coverage_graph_losses", "")}`

## Bootstrap Confidence Intervals

Class-level confidence intervals are reported only when the class has at least 5 rows; smaller classes are marked `insufficient_n_for_ci`.

{bootstrap_table}

## Cost and Latency Estimate

- Total estimated latency ms: `{cost_latency.get("total_latency_ms", "")}`
- Latency missing rows: `{cost_latency.get("latency_missing_rows", "")}`
- Total estimated input tokens: `{cost_latency.get("total_input_tokens_est", "")}`
- Total estimated output tokens: `{cost_latency.get("total_output_tokens_est", "")}`
- Total estimated cost USD: `{cost_latency.get("total_cost_est_usd", "")}`
- Cost warning: `{cost_latency.get("cost_warning", "")}`

## Metric Conflict Summary

- Rows with disagreement: `{disagreement_summary.get("rows_with_disagreement", "")}`
- Deterministic vs RAGAS conflicts: `{disagreement_summary.get("deterministic_vs_ragas_conflicts", "")}`
- Deterministic vs judge conflicts: `{disagreement_summary.get("deterministic_vs_judge_conflicts", "")}`
- RAGAS vs judge conflicts: `{disagreement_summary.get("ragas_vs_judge_conflicts", "")}`
- Low-confidence judge rows: `{disagreement_summary.get("low_judge_confidence_rows", "")}`

## Thematic and Multi-Hop Win Rate

Hard query classes used for this analysis: `{", ".join(sorted(HARD_QUERY_CLASSES))}`.

- Hard-query rows: `{summary["hard_query_rows"]}`
- Hard-query GraphRAG win rate: `{summary["hard_query_graph_win_rate"]}`

{hard_table}

## Failure Analysis

{failure_table}

## Review Queue Preview

{review_queue_table}

The failure taxonomy separates retrieval failures from context noise, unsupported answers, dodged answers, incomplete synthesis, and ties. This is useful because a single average score hides whether a system failed because it retrieved the wrong entities or because the generator failed to use good context.

## Limitations

The Week 3 layer uses deterministic proxy metrics for reproducibility and speed. These proxies are useful for ranking and debugging, but a final production evaluation should add human review or LLM-as-a-judge validation for borderline rows.

The vector RAG baseline is intentionally a standard retrieval baseline, while the structured baseline from Week 2 remains secondary evidence.

## Methods Appendix

- Metrics version: `{METRICS_VERSION}`
- Dataset file: `{dataset_file}`
- Dataset rows: `{len(results)}`
- Dataset SHA256: `{dataset_hash}`
- Pipeline git SHA: `{git_sha(ROOT)}`
- Run manifest: `run_manifest.json`
- Manifest git SHA: `{run_manifest.get("git_sha", "")}`
- Manifest combined_from: `{run_manifest.get("combined_from", "")}`
- Input contract: dataset CSV requires `id`, `question`, `ground_truth`, `contexts`, and `query_class`; prediction CSVs require `id`, `answer`, and `retrieved_contexts`.
- Failure taxonomy: retrieval missed entities, low context precision, unsupported answer, incomplete synthesis, tie, neither, and no clear failure.
- Deterministic metrics are reproducible proxies for retrieval and answer quality, not human correctness.
- RAGAS adds LLM-based faithfulness/relevancy/context scoring when enabled.
- LLM-as-a-judge adds pairwise preference validation when enabled.
- Graph coverage formulas are transparent structural proxies over retrieved GraphRAG contexts, not full graph-path precision.
- Row-level `low_judge_confidence` flags use the configurable disagreement threshold, default `0.7`; summary confidence buckets are fixed as high `>=0.8`, medium `0.5-0.8`, and low `<0.5`.

## Conclusion

GraphRAG should be evaluated by where it helps, not just by whether its global average is higher. The Week 3 results identify which query classes benefit from graph-style retrieval and which failure modes still need work.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    benchmark_path = Path(args.benchmark)
    graph_path = Path(args.graph_predictions)
    vector_path = Path(args.vector_predictions)
    manual_path = Path(args.manual_review)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)

    benchmark = _load_csv(benchmark_path, "benchmark CSV")
    graph_df = _load_csv(graph_path, "GraphRAG predictions CSV")

    if not vector_path.exists():
        if args.no_generate_vector:
            raise FileNotFoundError(f"Missing vector predictions CSV: {vector_path}")
        _generate_vector_predictions(benchmark, vector_path)
    vector_df = _load_csv(vector_path, "vector RAG predictions CSV")
    _validate_inputs(benchmark, graph_df, vector_df)

    manual_df = _load_manual_review(manual_path)

    results = _build_results(benchmark, graph_df, vector_df, manual_df)
    results.attrs["benchmark_path"] = str(benchmark_path)
    if len(results) != len(benchmark):
        raise RuntimeError(f"Expected {len(benchmark)} result rows, got {len(results)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "eval_results.csv"
    summary_path = out_dir / "summary_by_query_class.csv"
    failure_path = out_dir / "failure_analysis.csv"
    metric_path = out_dir / "metric_summary.json"

    results.to_csv(results_path, index=False)
    summary_by_class = _summarize_by_query_class(results)
    summary_by_class.to_csv(summary_path, index=False)
    failure_cols = [
        "id",
        "question",
        "query_class",
        "winner",
        "failure_mode",
        "graph_score",
        "vector_score",
        "graph_entity_recall",
        "vector_entity_recall",
        "graph_context_precision",
        "vector_context_precision",
        "graph_answer",
        "vector_answer",
    ]
    results[[col for col in failure_cols if col in results.columns]].to_csv(
        failure_path, index=False
    )
    summary = _metric_summary(results, summary_by_class)
    metric_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(report_path, results, summary_by_class, summary)

    print(f"Saved row-level results -> {results_path}")
    print(f"Saved query-class summary -> {summary_path}")
    print(f"Saved failure analysis -> {failure_path}")
    print(f"Saved metric summary -> {metric_path}")
    print(f"Saved report -> {report_path}")
    print(
        "Graph wins={graph} vector wins={vector} ties={tie} neither={neither}".format(
            graph=summary["winner_counts"].get("graph", 0),
            vector=summary["winner_counts"].get("vector", 0),
            tie=summary["winner_counts"].get("tie", 0),
            neither=summary["winner_counts"].get("neither", 0),
        )
    )


if __name__ == "__main__":
    main()
