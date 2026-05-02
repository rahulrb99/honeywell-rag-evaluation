"""Run RAGAS evaluation on week 2 GraphRAG predictions.

Reads:  outputs/predictions/week2_graphrag_predictions.csv  (or GRAPH_PREDICTIONS_CSV)
Writes: outputs/eval_outputs/week2_ragas_scores.json   – mean metric scores
        outputs/eval_outputs/week2_ragas_rows.csv       – per-row metric scores

Usage:
    python -m src.week2_graph_rag.run_eval_week2
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from src.config import get_settings, require_groq_api_key
from src.week1_vector_rag.run_eval_week1 import load_rag_outputs_df


_DEFAULT_PREDICTIONS = "outputs/predictions/week2_graphrag_predictions.csv"
_DEFAULT_SCORES_JSON = "outputs/eval_outputs/week2_ragas_scores.json"
_DEFAULT_ROWS_CSV    = "outputs/eval_outputs/week2_ragas_rows.csv"


def run_ragas_week2(pred_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Run RAGAS on a predictions dataframe.

    Returns (mean_scores_dict, per_row_dataframe).
    """
    settings = get_settings()
    api_key = require_groq_api_key(settings)

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": [
                    item["text"] if isinstance(item, dict) else str(item)
                    for item in row["retrieved_contexts"]
                ],
                "reference": row["ground_truth"],
            }
            for _, row in pred_df.iterrows()
        ]
    )

    groq_retries = int(os.getenv("GROQ_MAX_RETRIES", "8"))
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model=settings.groq_model,
            api_key=api_key,
            temperature=settings.temperature,
            max_retries=groq_retries,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    answer_relevancy = AnswerRelevancy(strictness=1)
    run_config = RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT_SEC", "600")),
        max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "1")),
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    rows_df = result.to_pandas()
    mean_scores = rows_df.mean(numeric_only=True).to_dict()
    return mean_scores, rows_df


def main() -> None:
    predictions_csv = Path(os.getenv("GRAPH_PREDICTIONS_CSV", _DEFAULT_PREDICTIONS))
    scores_json     = Path(os.getenv("GRAPH_RAGAS_SCORES_JSON", _DEFAULT_SCORES_JSON))
    rows_csv        = Path(os.getenv("GRAPH_RAGAS_ROWS_CSV", _DEFAULT_ROWS_CSV))

    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"Missing predictions CSV: {predictions_csv}\n"
            "Run src.week2_graph_rag.run_graphrag_predictions first."
        )

    print(f"Loading predictions -> {predictions_csv}", flush=True)
    pred_df = load_rag_outputs_df(predictions_csv)
    print(f"  {len(pred_df)} rows loaded", flush=True)

    print("Running RAGAS evaluation ...", flush=True)
    mean_scores, rows_df = run_ragas_week2(pred_df)

    scores_json.parent.mkdir(parents=True, exist_ok=True)
    scores_json.write_text(json.dumps(mean_scores, indent=2), encoding="utf-8")

    rows_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(rows_csv, index=False)

    print(f"\nSaved mean scores  -> {scores_json}")
    print(f"Saved per-row CSV  -> {rows_csv}")
    print("\nWeek 2 GraphRAG RAGAS mean scores:")
    for metric, score in mean_scores.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()
