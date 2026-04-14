from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from src.config import get_settings, require_groq_api_key
from src.rag_pipeline import answer_question


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_contexts(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            return [raw]
    return []


def run_predictions(eval_df: pd.DataFrame) -> pd.DataFrame:
    run_id = str(uuid4())
    rows = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Running baseline RAG"):
        output = answer_question(str(row["question"]))
        rows.append(
            {
                "id": row.get("id", output["query_id"]),
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "contexts": _parse_contexts(row["contexts"]),
                "category": row.get("category", ""),
                "answer": output["answer"],
                "retrieved_contexts": output["retrieved_contexts"],
                "latency_ms": output["latency_ms"],
                "model_name": output["model_name"],
                "timestamp_utc": output["timestamp_utc"],
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows)


def _serialize_prediction_export(pred_df: pd.DataFrame) -> pd.DataFrame:
    export_df = pred_df.copy()
    for column in ("contexts", "retrieved_contexts"):
        export_df[column] = export_df[column].apply(json.dumps)
    return export_df


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

    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model=settings.groq_model,
            api_key=api_key,
            temperature=settings.temperature,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()


def main() -> None:
    settings = get_settings()
    eval_path = Path(settings.eval_csv)
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {settings.eval_csv}")

    eval_df = pd.read_csv(eval_path)
    required_cols = {"id", "question", "ground_truth", "contexts", "category"}
    if not required_cols.issubset(eval_df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")

    pred_df = run_predictions(eval_df)
    _ensure_parent(settings.predictions_csv)
    export_df = _serialize_prediction_export(pred_df)
    export_df.to_csv(settings.predictions_csv, index=False)

    scores = run_ragas(pred_df)
    _ensure_parent(settings.ragas_scores_json)
    with open(settings.ragas_scores_json, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Saved predictions -> {settings.predictions_csv}")
    print(f"Saved scores -> {settings.ragas_scores_json}")
    print("Baseline RAGAS mean scores:", scores)


if __name__ == "__main__":
    main()
