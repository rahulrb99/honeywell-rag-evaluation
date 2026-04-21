from __future__ import annotations

import json
import os
import re
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

GRAPH_EVAL_CSV = os.getenv("GRAPH_EVAL_CSV", "data/eval/graph_rag_gold_triplets.csv")
GRAPH_PREDICTIONS_CSV = os.getenv("GRAPH_PREDICTIONS_CSV", "outputs/graph_rag_predictions.csv")
GRAPH_SCORES_JSON = os.getenv("GRAPH_SCORES_JSON", "outputs/graph_rag_scores.json")


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_contexts(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
            parsed_text = str(parsed).strip()
            return [parsed_text] if parsed_text else []
        except json.JSONDecodeError:
            return [raw]
    return []


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    normalized = re.sub(r"\s+", " ", lowered)
    return normalized.strip()


def _context_hit(row: pd.Series) -> bool:
    gold_contexts = [_normalize_text(c) for c in row["contexts"] if c]
    retrieved_contexts = [
        _normalize_text(item.get("text", ""))
        for item in row["retrieved_contexts"]
        if isinstance(item, dict)
    ]
    if not gold_contexts or not retrieved_contexts:
        return False

    for gold in gold_contexts:
        if not gold:
            continue
        for retrieved in retrieved_contexts:
            if gold in retrieved or retrieved in gold:
                return True
    return False


def run_predictions(eval_df: pd.DataFrame) -> pd.DataFrame:
    run_id = str(uuid4())
    rows = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Running Graph RAG baseline"):
        output = answer_question(str(row["question"]))
        rows.append(
            {
                "id": row.get("id", output["query_id"]),
                "domain": row.get("domain", ""),
                "reasoning_type": row.get("reasoning_type", ""),
                "question": row["question"],
                "ground_truth": row.get("ground_truth", ""),
                "contexts": _parse_contexts(row.get("contexts", "")),
                "answer": output["answer"],
                "retrieved_contexts": output["retrieved_contexts"],
                "context_hit": False,
                "latency_ms": output["latency_ms"],
                "model_name": output["model_name"],
                "timestamp_utc": output["timestamp_utc"],
                "run_id": run_id,
            }
        )

    pred_df = pd.DataFrame(rows)
    if not pred_df.empty:
        pred_df["context_hit"] = pred_df.apply(_context_hit, axis=1)
    return pred_df


def _rows_with_ground_truth(pred_df: pd.DataFrame) -> pd.DataFrame:
    keep = pred_df[
        pred_df["ground_truth"].astype(str).str.strip().ne("")
        & pred_df["contexts"].apply(lambda values: len(values) > 0)
    ].copy()
    return keep


def _run_ragas_for_rows(pred_df: pd.DataFrame) -> dict:
    settings = get_settings()
    api_key = require_groq_api_key(settings)

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": [
                    item.get("text", "") if isinstance(item, dict) else str(item)
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


def _serialize_prediction_export(pred_df: pd.DataFrame) -> pd.DataFrame:
    export_df = pred_df.copy()
    for column in ("contexts", "retrieved_contexts"):
        export_df[column] = export_df[column].apply(json.dumps)
    return export_df


def _compute_domain_context_hit(pred_df: pd.DataFrame) -> dict:
    by_domain = {}
    if pred_df.empty:
        return by_domain

    for domain, rows in pred_df.groupby("domain"):
        label = str(domain).strip() or "unlabeled"
        by_domain[label] = float(rows["context_hit"].mean())
    return by_domain


def main() -> None:
    eval_path = Path(GRAPH_EVAL_CSV)
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {GRAPH_EVAL_CSV}")

    eval_df = pd.read_csv(eval_path)
    required_cols = {"id", "domain", "reasoning_type", "question", "ground_truth", "contexts"}
    if not required_cols.issubset(eval_df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")

    eval_df = eval_df[eval_df["question"].astype(str).str.strip().ne("")].copy()
    if eval_df.empty:
        raise ValueError("Eval dataset has no usable questions.")

    pred_df = run_predictions(eval_df)

    _ensure_parent(GRAPH_PREDICTIONS_CSV)
    export_df = _serialize_prediction_export(pred_df)
    export_df.to_csv(GRAPH_PREDICTIONS_CSV, index=False)

    scored_rows = _rows_with_ground_truth(pred_df)
    overall_ragas = {}
    by_domain_ragas = {}
    if not scored_rows.empty:
        overall_ragas = _run_ragas_for_rows(scored_rows)
        for domain, rows in scored_rows.groupby("domain"):
            if len(rows) < 2:
                continue
            by_domain_ragas[str(domain)] = _run_ragas_for_rows(rows)

    scores = {
        "num_questions_answered": int(len(pred_df)),
        "num_questions_scored_with_ragas": int(len(scored_rows)),
        "context_hit_rate_overall": float(pred_df["context_hit"].mean()) if len(pred_df) else 0.0,
        "context_hit_rate_by_domain": _compute_domain_context_hit(pred_df),
        "overall_ragas": overall_ragas,
        "by_domain_ragas": by_domain_ragas,
    }

    _ensure_parent(GRAPH_SCORES_JSON)
    with open(GRAPH_SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Saved predictions -> {GRAPH_PREDICTIONS_CSV}")
    print(f"Saved scores -> {GRAPH_SCORES_JSON}")
    print("Graph-RAG baseline metrics:", scores)


if __name__ == "__main__":
    main()
