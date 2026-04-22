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


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_json_list_cell(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        return json.loads(raw)
    return []


def load_rag_outputs_df(rag_outputs_path: Path) -> pd.DataFrame:
    df = pd.read_csv(rag_outputs_path)
    required = {"question", "ground_truth", "answer", "retrieved_contexts"}
    if not required.issubset(df.columns):
        raise ValueError(f"Predictions CSV must contain columns: {sorted(required)}")
    out = df.copy()
    out["retrieved_contexts"] = out["retrieved_contexts"].apply(_parse_json_list_cell)
    return out


def run_ragas(pred_df: pd.DataFrame) -> dict:
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

    # Groq default max_retries=2 is thin for TPM bursts; RAGAS sends many LLM calls.
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

    # Groq chat completions return n=1; AnswerRelevancy(strictness=3) triggers
    # "LLM returned 1 generations instead of requested 3" and weaker scores.
    answer_relevancy = AnswerRelevancy(strictness=1)
    # Groq on-demand TPM is low; parallel metric jobs easily hit 429. Default to 1 worker;
    # raise RAGAS_MAX_WORKERS only if your org tier allows higher concurrency.
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
    return result.to_pandas().mean(numeric_only=True).to_dict()


def main() -> None:
    settings = get_settings()
    rag_outputs_path = Path(os.getenv("PREDICTIONS_CSV", settings.predictions_csv))
    if not rag_outputs_path.exists():
        raise FileNotFoundError(
            f"Missing RAG outputs file: {rag_outputs_path}. "
            "Generate raw RAG outputs first, then run this script."
        )

    pred_df = load_rag_outputs_df(rag_outputs_path)
    scores = run_ragas(pred_df)
    _ensure_parent(settings.ragas_scores_json)
    with open(settings.ragas_scores_json, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Loaded RAG outputs from -> {rag_outputs_path}")
    print(f"Saved scores -> {settings.ragas_scores_json}")
    print("Baseline RAGAS mean scores:", scores)


if __name__ == "__main__":
    main()
