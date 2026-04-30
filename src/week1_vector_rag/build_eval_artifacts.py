"""Build consolidated eval artifacts from individual evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_settings


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _select_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]].copy()


def _rename_existing(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _merge_optional(base: pd.DataFrame, other: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if other is None or "id" not in other.columns:
        return base
    other = _select_existing(other, columns)
    other["id"] = other["id"].astype(str)
    return base.merge(other, on="id", how="left")


def build_rows(settings) -> pd.DataFrame:
    rag_outputs = _read_csv(settings.predictions_csv)
    if rag_outputs is None:
        raise FileNotFoundError(f"Missing RAG outputs: {settings.predictions_csv}")

    rows = _select_existing(
        rag_outputs,
        [
            "id", "question", "category", "ground_truth", "answer",
            "metadata_filter_used", "latency_ms", "model_name",
        ],
    )
    rows["id"] = rows["id"].astype(str)

    retrieval = _read_csv(settings.retrieval_only_csv)
    rows = _merge_optional(
        rows,
        retrieval,
        [
            "id", "retrieval_hit", "recall_hit", "best_overlap_score",
            "best_chunk_id", "failure_reason",
        ],
    )
    rows = _rename_existing(rows, {"best_overlap_score": "retrieval_best_overlap_score"})

    generation = _read_csv(settings.generation_only_csv)
    if generation is not None:
        generation = _rename_existing(
            generation,
            {
                "answer": "generation_only_answer",
                "judge_mean_1_to_5": "generation_only_judge_mean_1_to_5",
                "judge_mean_0_to_1": "generation_only_judge_mean_0_to_1",
                "correctness_score": "generation_only_correctness_score",
                "completeness_score": "generation_only_completeness_score",
            },
        )
    rows = _merge_optional(
        rows,
        generation,
        [
            "id", "generation_only_answer", "generation_only_judge_mean_1_to_5",
            "generation_only_judge_mean_0_to_1", "generation_only_correctness_score",
            "generation_only_completeness_score",
        ],
    )

    judge = _read_csv(settings.judge_scores_csv)
    if judge is not None:
        judge = _rename_existing(
            judge,
            {
                "judge_mean_1_to_5": "end_to_end_judge_mean_1_to_5",
                "judge_mean_0_to_1": "end_to_end_judge_mean_0_to_1",
                "faithfulness_score": "end_to_end_faithfulness_score",
                "answer_relevance_score": "end_to_end_answer_relevance_score",
                "completeness_score": "end_to_end_completeness_score",
                "correctness_score": "end_to_end_correctness_score",
                "conciseness_score": "end_to_end_conciseness_score",
            },
        )
    rows = _merge_optional(
        rows,
        judge,
        [
            "id", "end_to_end_judge_mean_1_to_5", "end_to_end_judge_mean_0_to_1",
            "end_to_end_faithfulness_score", "end_to_end_answer_relevance_score",
            "end_to_end_completeness_score", "end_to_end_correctness_score",
            "end_to_end_conciseness_score",
        ],
    )

    grounded = _read_csv(settings.groundedness_csv)
    rows = _merge_optional(
        rows,
        grounded,
        ["id", "failure_tag", "max_chunk_sim", "used_status"],
    )
    rows = _rename_existing(rows, {"used_status": "grounded_used"})
    return rows


def build_summary(settings, rows: pd.DataFrame) -> dict[str, Any]:
    ragas = _read_json(settings.ragas_scores_json)
    judge = _read_json(settings.judge_scores_json)
    retrieval = _read_json(settings.retrieval_only_json)
    recall = _read_json(str(Path(settings.retrieval_recall_csv).with_suffix(".json")))
    generation = _read_json(settings.generation_only_json)

    return {
        "total_rows": int(len(rows)),
        "rag_outputs_csv": settings.predictions_csv,
        "eval_rows_csv": settings.eval_rows_csv,
        "ragas": ragas.get("scores", ragas),
        "retrieval": retrieval,
        "recall": recall,
        "generation_only": generation,
        "end_to_end_judge": judge.get("metric_means", judge),
        "groundedness_failure_counts": (
            rows["failure_tag"].value_counts(dropna=False).to_dict()
            if "failure_tag" in rows.columns
            else {}
        ),
    }


def main() -> None:
    settings = get_settings()
    rows = build_rows(settings)
    _ensure_parent(settings.eval_rows_csv)
    rows.to_csv(settings.eval_rows_csv, index=False)

    summary = build_summary(settings, rows)
    _ensure_parent(settings.eval_summary_json)
    with open(settings.eval_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved consolidated eval rows -> {settings.eval_rows_csv}")
    print(f"Saved consolidated eval summary -> {settings.eval_summary_json}")


if __name__ == "__main__":
    main()
