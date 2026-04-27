from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MANUAL_CSV = ROOT / "data" / "eval" / "honeywell_graph_subset_24.csv"
AUTOQ_CSV = ROOT / "data" / "eval" / "autoq_honeywell_structured.csv"
OUTPUT_CSV = ROOT / "data" / "eval" / "week2_honeywell_eval.csv"


def _infer_reasoning_type(category: str) -> str:
    mapping = {
        "product_description": "single_hop",
        "usage": "single_hop",
        "capacity": "single_hop",
        "function": "single_hop",
        "io": "single_hop",
        "specification": "single_hop",
        "fault_monitoring": "list_retrieval",
        "ui": "single_hop",
        "configuration": "list_retrieval",
        "capability": "single_hop",
        "installation": "single_hop",
        "usage_environment": "list_retrieval",
    }
    return mapping.get(category, "single_hop")


def main() -> None:
    if not MANUAL_CSV.exists():
        raise FileNotFoundError(f"Missing manual eval set: {MANUAL_CSV}")
    if not AUTOQ_CSV.exists():
        raise FileNotFoundError(f"Missing AutoQ eval set: {AUTOQ_CSV}")

    manual_df = pd.read_csv(MANUAL_CSV).copy()
    autoq_df = pd.read_csv(AUTOQ_CSV).copy()

    manual_df["query_class"] = "manual_curated"
    manual_df["reasoning_type"] = manual_df["category"].apply(_infer_reasoning_type)
    manual_df["source_product"] = manual_df["question"].apply(
        lambda q: "RK-ZONE8" if "RK-ZONE8" in q else ("L-PCM06B" if "L-PCM06B" in q else ("L-Series" if "L-Series" in q else ""))
    )
    manual_df["source_fields"] = manual_df["category"].apply(lambda value: json.dumps([value]))
    manual_df["source_split"] = "manual"

    max_manual_id = int(manual_df["id"].max())
    autoq_df["id"] = range(max_manual_id + 1, max_manual_id + 1 + len(autoq_df))

    merged = pd.concat([manual_df, autoq_df], ignore_index=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Loaded manual set -> {MANUAL_CSV}")
    print(f"Loaded AutoQ set -> {AUTOQ_CSV}")
    print(f"Built merged Week 2 eval set with {len(merged)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
