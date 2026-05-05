from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.week3_benchmark.run_week3_eval import _context_texts, _normalize_text, _parse_json_cell


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"
RELATION_PATTERNS = re.compile(
    r"\b(related|compatible_with|connects_to|part_of|used_for|requires|supports|works with|compatible with|connects to|part of|used for)\b",
    re.IGNORECASE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute transparent GraphRAG coverage proxies.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def _as_list(value: Any) -> list[str]:
    parsed = _parse_json_cell(value, [])
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _contains_signal(signal: str, text: str) -> bool:
    normalized_signal = _normalize_text(signal).replace("_", " ").replace("-", " ")
    normalized_text = _normalize_text(text).replace("_", " ").replace("-", " ")
    if not normalized_signal:
        return False
    if normalized_signal in normalized_text:
        return True
    signal_tokens = {tok for tok in re.findall(r"[a-z0-9]+", normalized_signal) if len(tok) > 2}
    text_tokens = {tok for tok in re.findall(r"[a-z0-9]+", normalized_text) if len(tok) > 2}
    return bool(signal_tokens) and len(signal_tokens & text_tokens) / len(signal_tokens) >= 0.8


def _source_field_signals(row: pd.Series) -> list[str]:
    signals = _as_list(row.get("source_fields", "[]"))
    out: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        cleaned = signal.replace("_", " ").strip()
        key = _normalize_text(cleaned)
        if key and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def score_graph_coverage_row(row: pd.Series) -> dict[str, object]:
    contexts = _context_texts(row.get("graph_retrieved_contexts", "[]"))
    context_text = "\n".join(contexts)
    expected_entities = _as_list(row.get("expected_entities", "[]"))
    relation_signals = _source_field_signals(row)
    entity_hits = [signal for signal in expected_entities if _contains_signal(signal, context_text)]
    relation_hits = [signal for signal in relation_signals if _contains_signal(signal, context_text)]
    relation_like_context_count = sum(1 for context in contexts if RELATION_PATTERNS.search(context))
    entity_coverage = len(entity_hits) / len(expected_entities) if expected_entities else 1.0
    relation_coverage = len(relation_hits) / len(relation_signals) if relation_signals else 1.0
    structural_density = relation_like_context_count / len(contexts) if contexts else 0.0
    coverage_score = 0.5 * entity_coverage + 0.3 * relation_coverage + 0.2 * structural_density
    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "query_class": row.get("query_class", ""),
        "winner": row.get("winner", ""),
        "graph_expected_entity_coverage": round(entity_coverage, 4),
        "relation_signal_coverage": round(relation_coverage, 4),
        "structural_density": round(structural_density, 4),
        "graph_coverage_score": round(coverage_score, 4),
        "expected_entity_count": len(expected_entities),
        "expected_entities_found": json.dumps(entity_hits),
        "relation_signal_count": len(relation_signals),
        "relation_signals_found": json.dumps(relation_hits),
        "graph_context_count": len(contexts),
        "relation_like_context_count": relation_like_context_count,
    }


def build_graph_coverage(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([score_graph_coverage_row(row) for _, row in results.iterrows()])


def summarize_by_class(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    return (
        coverage.groupby("query_class", dropna=False)
        .agg(
            rows=("id", "count"),
            graph_coverage_score_mean=("graph_coverage_score", "mean"),
            graph_expected_entity_coverage_mean=("graph_expected_entity_coverage", "mean"),
            relation_signal_coverage_mean=("relation_signal_coverage", "mean"),
            structural_density_mean=("structural_density", "mean"),
        )
        .reset_index()
        .round(4)
    )


def coverage_vs_winner(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    return (
        coverage.groupby("winner", dropna=False)
        .agg(
            rows=("id", "count"),
            graph_coverage_score_mean=("graph_coverage_score", "mean"),
            graph_expected_entity_coverage_mean=("graph_expected_entity_coverage", "mean"),
            relation_signal_coverage_mean=("relation_signal_coverage", "mean"),
            structural_density_mean=("structural_density", "mean"),
        )
        .reset_index()
        .round(4)
    )


def interpret_graph_coverage(coverage: pd.DataFrame) -> dict[str, object]:
    if coverage.empty:
        return {
            "rows": 0,
            "median_graph_coverage_score": 0.0,
            "high_coverage_graph_losses": 0,
            "low_coverage_graph_losses": 0,
            "interpretation": {},
        }
    median = float(coverage["graph_coverage_score"].median())
    losses = coverage[coverage["winner"] != "graph"].copy()
    high_losses = losses[losses["graph_coverage_score"] >= median]
    low_losses = losses[losses["graph_coverage_score"] < median]
    by_winner = coverage_vs_winner(coverage)
    return {
        "rows": int(len(coverage)),
        "median_graph_coverage_score": round(median, 4),
        "high_coverage_threshold": "graph_coverage_score >= run median",
        "low_coverage_threshold": "graph_coverage_score < run median",
        "high_coverage_graph_losses": int(len(high_losses)),
        "low_coverage_graph_losses": int(len(low_losses)),
        "coverage_by_winner": by_winner.to_dict(orient="records"),
        "interpretation": {
            "high_coverage_loss": "Graph evidence was retrieved, so the likely issue is generation or synthesis.",
            "low_coverage_loss": "Graph evidence was weak or missing, so the likely issue is retrieval or coverage.",
        },
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage = build_graph_coverage(pd.read_csv(args.results))
    coverage.to_csv(out_dir / "graph_coverage_metrics.csv", index=False)
    summarize_by_class(coverage).to_csv(out_dir / "graph_coverage_summary_by_class.csv", index=False)
    coverage_vs_winner(coverage).to_csv(out_dir / "graph_coverage_vs_winner.csv", index=False)
    (out_dir / "graph_coverage_interpretation.json").write_text(
        json.dumps(interpret_graph_coverage(coverage), indent=2), encoding="utf-8"
    )
    print(f"Saved graph coverage metrics -> {out_dir}")


if __name__ == "__main__":
    main()
