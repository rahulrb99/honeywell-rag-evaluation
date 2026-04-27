from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset


def _parse_json_list_cell(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        return json.loads(raw)
    return []


def main() -> None:
    input_csv = Path(os.getenv("PREDICTIONS_CSV", "outputs/debug/week1_rag_outputs.csv"))
    output_json = Path(os.getenv("RAGAS_DATASET_JSON", "outputs/week2/ragas_dataset.json"))
    output_csv = Path(os.getenv("RAGAS_SAMPLES_CSV", "outputs/week2/ragas_samples.csv"))

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"question", "ground_truth", "answer", "retrieved_contexts"}
    if not required.issubset(df.columns):
        raise ValueError(f"Predictions CSV must contain columns: {sorted(required)}")

    samples = []
    for _, row in df.iterrows():
        retrieved_contexts = [
            item["text"] if isinstance(item, dict) else str(item)
            for item in _parse_json_list_cell(row["retrieved_contexts"])
        ]
        sample = {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": retrieved_contexts,
            "reference": row["ground_truth"],
        }
        for optional in (
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
        ):
            if optional in row and pd.notna(row[optional]):
                sample[optional] = row[optional]
        samples.append(sample)

    ragas_dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": sample["user_input"],
                "response": sample["response"],
                "retrieved_contexts": sample["retrieved_contexts"],
                "reference": sample["reference"],
            }
            for sample in samples
        ]
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    pd.DataFrame(samples).to_csv(output_csv, index=False)
    print(f"Loaded predictions -> {input_csv}")
    print(f"Built EvaluationDataset with {len(ragas_dataset)} samples")
    print(f"Saved sample JSON -> {output_json}")
    print(f"Saved sample CSV -> {output_csv}")


if __name__ == "__main__":
    main()
