from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample


@dataclass(frozen=True)
class PackagedRagasDataset:
    samples: list[SingleTurnSample]
    audit_rows: list[dict[str, Any]]
    dataset: EvaluationDataset


def parse_json_cell(value: object, fallback: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None or pd.isna(value):
        return fallback
    raw = str(value).strip()
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def extract_context_texts(value: object) -> list[str]:
    parsed = parse_json_cell(value, [])
    if not isinstance(parsed, list):
        return []
    contexts: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            contexts.append(text)
    return contexts


def build_single_turn_sample(row: pd.Series) -> SingleTurnSample:
    return SingleTurnSample(
        user_input=str(row["question"]),
        response=str(row["answer"]),
        retrieved_contexts=extract_context_texts(row["retrieved_contexts"]),
        reference=str(row["ground_truth"]),
    )


def build_packaged_dataset(predictions_df: pd.DataFrame, system_name: str) -> PackagedRagasDataset:
    required = {"question", "answer", "retrieved_contexts", "ground_truth"}
    missing = sorted(required - set(predictions_df.columns))
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {missing}")

    samples: list[SingleTurnSample] = []
    audit_rows: list[dict[str, Any]] = []
    optional_cols = [
        "id",
        "category",
        "query_class",
        "reasoning_type",
        "source_product",
        "source_fields",
        "source_split",
        "answer_mode",
        "retrieval_mode",
        "matched_field_name",
        "model_name",
        "domain_name",
        "run_id",
    ]
    for _, row in predictions_df.iterrows():
        sample = build_single_turn_sample(row)
        samples.append(sample)
        audit = {
            "system_name": system_name,
            "user_input": sample.user_input,
            "response": sample.response,
            "retrieved_contexts": sample.retrieved_contexts,
            "reference": sample.reference,
            "retrieved_context_count": len(sample.retrieved_contexts or []),
        }
        for col in optional_cols:
            if col in predictions_df.columns and pd.notna(row.get(col)):
                audit[col] = row.get(col)
        audit_rows.append(audit)

    dataset = EvaluationDataset(samples=samples)
    return PackagedRagasDataset(samples=samples, audit_rows=audit_rows, dataset=dataset)

