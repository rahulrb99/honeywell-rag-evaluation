"""Compute Retrieval Recall @ K for week1_predictions.csv.

For each question, checks whether any retrieved chunk contains enough of the
ground-truth answer tokens to count as a "hit". Uses dual-threshold overlap
to avoid length-bias errors:

  >= 0.4  → recall_hit = True
  <= 0.2  → recall_hit = False
  between → recall_hit = "uncertain"

Usage:
    python -m src.run_retrieval_recall
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import get_settings

_HIT_THRESHOLD = 0.4
_MISS_THRESHOLD = 0.2

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "with", "from", "by", "as", "up", "but",
    "not", "no", "so", "if", "than", "then", "also", "which", "who",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def _token_overlap(ground_truth: str, chunk_text: str) -> float:
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    ch_tokens = _tokenize(chunk_text)
    return len(gt_tokens & ch_tokens) / len(gt_tokens)


def _recall_hit(overlap: float) -> bool | str:
    if overlap >= _HIT_THRESHOLD:
        return True
    if overlap <= _MISS_THRESHOLD:
        return False
    return "uncertain"


def _parse_json_list_cell(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    settings = get_settings()

    pred_path = Path(settings.predictions_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    required = {"id", "question", "ground_truth", "retrieved_contexts"}
    if not required.issubset(pred_df.columns):
        raise ValueError(f"Predictions CSV must contain: {sorted(required)}")

    rows = []
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Retrieval Recall"):
        ground_truth = str(row["ground_truth"])
        retrieved_contexts = _parse_json_list_cell(row["retrieved_contexts"])

        if not retrieved_contexts:
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "recall_hit": False,
                    "best_overlap_score": 0.0,
                    "best_chunk_id": "",
                    "top_chunk_text": "",
                }
            )
            continue

        overlaps = [
            _token_overlap(ground_truth, str(ctx.get("text", "")))
            for ctx in retrieved_contexts
        ]
        best_idx = int(max(range(len(overlaps)), key=lambda i: overlaps[i]))
        best_overlap = overlaps[best_idx]
        best_ctx = retrieved_contexts[best_idx]

        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "recall_hit": _recall_hit(best_overlap),
                "best_overlap_score": round(best_overlap, 4),
                "best_chunk_id": best_ctx.get("chunk_id", ""),
                "top_chunk_text": str(best_ctx.get("text", ""))[:200].replace("\n", " "),
            }
        )

    result_df = pd.DataFrame(rows)
    _ensure_parent(settings.retrieval_recall_csv)
    result_df.to_csv(settings.retrieval_recall_csv, index=False)

    # Aggregate — exclude uncertain rows
    confident = result_df[result_df["recall_hit"] != "uncertain"]
    uncertain_count = int((result_df["recall_hit"] == "uncertain").sum())
    recall_at_k = float(confident["recall_hit"].mean()) if len(confident) > 0 else 0.0

    summary = {
        "recall_at_k": round(recall_at_k, 4),
        "total_rows": int(len(result_df)),
        "hits": int((result_df["recall_hit"] == True).sum()),  # noqa: E712
        "misses": int((result_df["recall_hit"] == False).sum()),  # noqa: E712
        "uncertain": uncertain_count,
    }

    recall_json_path = Path(settings.retrieval_recall_csv).with_suffix(".json")
    with open(recall_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved recall results -> {settings.retrieval_recall_csv}")
    print(f"Recall@K: {recall_at_k:.3f}  (hits={summary['hits']}, misses={summary['misses']}, uncertain={uncertain_count})")


if __name__ == "__main__":
    main()
