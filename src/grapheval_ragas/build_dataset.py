from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.grapheval_ragas.samples import build_packaged_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package GraphRAG predictions as RAGAS SingleTurnSample objects."
    )
    parser.add_argument("--predictions", required=True, help="Input predictions CSV.")
    parser.add_argument("--system-name", required=True, help="System label for audit exports.")
    parser.add_argument("--output-json", required=True, help="Output JSON sample audit file.")
    parser.add_argument("--output-csv", required=True, help="Output CSV sample audit file.")
    return parser.parse_args()


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> None:
    args = _parse_args()
    predictions_path = Path(args.predictions)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions_path}")

    df = pd.read_csv(predictions_path)
    packaged = build_packaged_dataset(df, system_name=args.system_name)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(packaged.audit_rows, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    audit_df = pd.DataFrame(packaged.audit_rows)
    if "retrieved_contexts" in audit_df.columns:
        audit_df["retrieved_contexts"] = audit_df["retrieved_contexts"].apply(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    audit_df.to_csv(output_csv, index=False)

    print(f"Loaded predictions -> {predictions_path}")
    print(f"Built SingleTurnSample count -> {len(packaged.samples)}")
    print(f"Built EvaluationDataset sample count -> {len(packaged.dataset.samples)}")
    print(f"Saved sample JSON -> {output_json}")
    print(f"Saved sample CSV -> {output_csv}")


if __name__ == "__main__":
    main()
