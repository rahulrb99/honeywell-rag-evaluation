from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import BENCHMARK_PATH, SOURCE_EVAL_CSV_PATH, seed_everything
from .io_utils import write_json
from .normalize import normalize_entities


def _parse_json_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


def convert_eval_csv_to_benchmark(csv_path: Path = SOURCE_EVAL_CSV_PATH) -> list[dict[str, Any]]:
    seed_everything()
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in sorted(reader, key=lambda item: int(item.get("id", "0") or 0)):
            source_product = str(row.get("source_product", "")).strip()
            source_fields = _parse_json_list(str(row.get("source_fields", "")))
            question_entities = normalize_entities([source_product, *source_fields])
            rows.append(
                {
                    "id": f"q{int(row.get('id', len(rows) + 1)):03d}",
                    "source_eval_id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("ground_truth", ""),
                    "answer_entity": source_product or row.get("ground_truth", ""),
                    "question_entities": question_entities,
                    "gold_reasoning_path": [],
                    "contexts": _parse_json_list(str(row.get("contexts", ""))),
                    "category": row.get("category", ""),
                    "query_class": row.get("query_class", ""),
                    "reasoning_type": row.get("reasoning_type", ""),
                    "source_product": source_product,
                    "source_fields": source_fields,
                    "source_split": row.get("source_split", ""),
                }
            )
    return rows


def write_benchmark_from_eval(csv_path: Path = SOURCE_EVAL_CSV_PATH, out_path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    benchmark = convert_eval_csv_to_benchmark(csv_path)
    write_json(out_path, benchmark)
    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an existing repo eval CSV into newevals benchmark JSON.")
    parser.add_argument("--csv", default=str(SOURCE_EVAL_CSV_PATH))
    parser.add_argument("--out", default=str(BENCHMARK_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_benchmark_from_eval(Path(args.csv), Path(args.out))
    print(f"Wrote {len(rows)} benchmark rows to {args.out}")


if __name__ == "__main__":
    main()
