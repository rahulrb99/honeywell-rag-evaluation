"""Run full RAG pipeline from raw docs + gold dataset to dashboard.

Usage:
    python -m src.run_all
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from src.config import get_settings


def _run(module: str) -> None:
    print(f"\n=== Running: python -m {module} ===")
    runpy.run_module(module, run_name="__main__")


def _ensure_inputs() -> None:
    settings = get_settings()
    raw_dir = Path(settings.raw_data_dir)
    eval_csv = Path(settings.eval_csv)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw data folder: {raw_dir}")

    raw_files = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}
    ]
    if not raw_files:
        raise ValueError(f"No source docs found in {raw_dir}. Add .pdf/.txt/.md files.")

    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing eval CSV: {eval_csv}")

    print(f"Input check OK: {len(raw_files)} source docs, eval={eval_csv}")


def _print_outputs() -> None:
    settings = get_settings()
    print("\n=== Pipeline Complete ===")
    print(f"RAG outputs:      {settings.predictions_csv}")
    print(f"RAGAS scores:     {settings.ragas_scores_json}")
    print(f"Retrieval metrics: {settings.retrieval_only_csv}")
    print(f"Judge scores:     {settings.judge_scores_csv}")
    print(f"Generation-only:  {settings.generation_only_csv}")
    print(f"Groundedness:     {settings.groundedness_csv}")
    print(f"Eval rows:        {settings.eval_rows_csv}")
    print(f"Eval summary:     {settings.eval_summary_json}")
    print(f"Dashboard:        {settings.failure_dashboard_html}")

    recall_json = Path(settings.retrieval_recall_csv).with_suffix(".json")
    if recall_json.exists():
        data = json.loads(recall_json.read_text(encoding="utf-8"))
        print(
            "Recall strict={} filtered={} uncertain_rate={}".format(
                data.get("recall_at_k_strict"),
                data.get("recall_at_k_filtered"),
                data.get("uncertain_rate"),
            )
        )


def main() -> None:
    _ensure_inputs()
    _run("src.ingest")
    _run("src.week1_vector_rag.run_rag_outputs")
    _run("src.week1_vector_rag.run_eval_week1")
    _run("src.week1_vector_rag.run_eval_retrieval_only")
    _run("src.week1_vector_rag.run_eval_llm_judge")
    _run("src.week1_vector_rag.run_groundedness")
    _run("src.week1_vector_rag.run_eval_generation_only")
    _run("src.week1_vector_rag.build_eval_artifacts")
    _run("src.week1_vector_rag.build_failure_dashboard")
    _print_outputs()


if __name__ == "__main__":
    main()
