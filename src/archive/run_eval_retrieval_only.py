"""Run merged retrieval coverage evaluation.

Consumes the configured raw RAG outputs CSV and writes both retrieval-only
diagnostics and recall@k summary artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import get_settings


HIT_THRESHOLD = 0.4
MISS_THRESHOLD = 0.2

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "with", "from", "by", "as", "up", "but",
    "not", "no", "so", "if", "than", "then", "also", "which", "who",
}


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_json_list_cell(value: object) -> list[dict[str, Any]]:
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


def token_overlap(ground_truth: str, chunk_text: str) -> float:
    gt_tokens = set(re.findall(r"[a-z0-9]+", ground_truth.lower())) - STOPWORDS
    if not gt_tokens:
        return 0.0
    chunk_tokens = set(re.findall(r"[a-z0-9]+", chunk_text.lower())) - STOPWORDS
    return len(gt_tokens & chunk_tokens) / len(gt_tokens)


def hit_label(overlap: float) -> bool | str:
    if overlap >= HIT_THRESHOLD:
        return True
    if overlap <= MISS_THRESHOLD:
        return False
    return "uncertain"


def failure_reason(hit: bool | str, has_contexts: bool) -> str:
    if not has_contexts:
        return "miss_no_context"
    if hit is True:
        return "ok"
    if hit is False:
        return "miss_no_overlap"
    return "uncertain_overlap"


def evaluate_predictions(pred_df: pd.DataFrame, desc: str = "Retrieval coverage") -> pd.DataFrame:
    required = {"id", "question", "ground_truth", "retrieved_contexts"}
    if not required.issubset(pred_df.columns):
        raise ValueError(f"Predictions CSV must contain: {sorted(required)}")

    rows = []
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc=desc):
        contexts = parse_json_list_cell(row["retrieved_contexts"])
        category = row["category"] if "category" in pred_df.columns else ""

        if not contexts:
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "category": category,
                    "retrieval_hit": False,
                    "recall_hit": False,
                    "best_overlap_score": 0.0,
                    "best_chunk_id": "",
                    "top_chunk_text": "",
                    "failure_reason": failure_reason(False, has_contexts=False),
                }
            )
            continue

        overlaps = [
            token_overlap(str(row["ground_truth"]), str(ctx.get("text", "")))
            for ctx in contexts
        ]
        best_idx = int(max(range(len(overlaps)), key=lambda i: overlaps[i]))
        best_overlap = overlaps[best_idx]
        best_ctx = contexts[best_idx]
        hit = hit_label(best_overlap)

        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "category": category,
                "retrieval_hit": hit,
                "recall_hit": hit,
                "best_overlap_score": round(best_overlap, 4),
                "best_chunk_id": best_ctx.get("chunk_id", ""),
                "top_chunk_text": str(best_ctx.get("text", ""))[:200].replace("\n", " "),
                "failure_reason": failure_reason(hit, has_contexts=True),
            }
        )

    return pd.DataFrame(rows)


def build_retrieval_only_summary(result_df: pd.DataFrame, pred_path: Path, output_csv: str) -> dict[str, Any]:
    confident = result_df[result_df["retrieval_hit"] != "uncertain"]
    hit_rate = float(confident["retrieval_hit"].mean()) if len(confident) else 0.0
    return {
        "input_predictions_csv": str(pred_path),
        "retrieval_only_csv": output_csv,
        "total_rows": int(len(result_df)),
        "hits": int((result_df["retrieval_hit"] == True).sum()),  # noqa: E712
        "misses": int((result_df["retrieval_hit"] == False).sum()),  # noqa: E712
        "uncertain": int((result_df["retrieval_hit"] == "uncertain").sum()),
        "retrieval_hit_rate_excluding_uncertain": round(hit_rate, 4),
    }


def build_recall_summary(result_df: pd.DataFrame) -> dict[str, Any]:
    total = len(result_df)
    strict_hits = int((result_df["recall_hit"] == True).sum())  # noqa: E712
    misses = int((result_df["recall_hit"] == False).sum())  # noqa: E712
    uncertain = int((result_df["recall_hit"] == "uncertain").sum())

    strict_recall = strict_hits / total if total > 0 else 0.0
    filtered_den = total - uncertain
    filtered_recall = strict_hits / filtered_den if filtered_den > 0 else 0.0

    return {
        "recall_at_k": round(filtered_recall, 4),
        "recall_at_k_strict": round(strict_recall, 4),
        "recall_at_k_filtered": round(filtered_recall, 4),
        "uncertain_rate": round((uncertain / total) if total > 0 else 0.0, 4),
        "total_rows": int(total),
        "hits": strict_hits,
        "misses": misses,
        "uncertain": uncertain,
        "uncertain_count": uncertain,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    settings = get_settings()
    pred_path = Path(settings.predictions_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing RAG outputs: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    result_df = evaluate_predictions(pred_df, desc="Retrieval coverage eval")

    retrieval_columns = [
        "id", "question", "category", "retrieval_hit", "recall_hit",
        "best_overlap_score", "best_chunk_id", "top_chunk_text", "failure_reason",
    ]
    ensure_parent(settings.retrieval_only_csv)
    result_df[retrieval_columns].to_csv(settings.retrieval_only_csv, index=False)

    retrieval_payload = build_retrieval_only_summary(
        result_df, pred_path, settings.retrieval_only_csv
    )
    recall_payload = build_recall_summary(result_df)
    recall_json_path = Path(settings.retrieval_recall_csv).with_suffix(".json")
    write_json(settings.retrieval_only_json, retrieval_payload)
    write_json(recall_json_path, recall_payload)

    print(f"Saved retrieval-only rows -> {settings.retrieval_only_csv}")
    print(f"Saved retrieval-only summary -> {settings.retrieval_only_json}")
    print(f"Saved recall summary -> {recall_json_path}")
    print("Retrieval-only summary:", retrieval_payload)
    print("Recall summary:", recall_payload)


if __name__ == "__main__":
    main()
