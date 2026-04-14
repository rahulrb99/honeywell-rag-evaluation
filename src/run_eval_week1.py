from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from tqdm import tqdm

from src.config import get_settings
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
    rows = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Running baseline RAG"):
        output = answer_question(str(row["question"]))
        rows.append(
            {
                "id": row.get("id", ""),
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "contexts": _parse_contexts(row["contexts"]),
                "category": row.get("category", ""),
                "answer": output["answer"],
                "retrieved_contexts": output["retrieved_contexts"],
                "retrieved_sources": output["retrieved_sources"],
            }
        )
    return pd.DataFrame(rows)


def run_ragas(pred_df: pd.DataFrame) -> dict:
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference": row["ground_truth"],
            }
            for _, row in pred_df.iterrows()
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
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
    pred_df.to_csv(settings.predictions_csv, index=False)

    scores = run_ragas(pred_df)
    _ensure_parent(settings.ragas_scores_json)
    with open(settings.ragas_scores_json, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Saved predictions -> {settings.predictions_csv}")
    print(f"Saved scores -> {settings.ragas_scores_json}")
    print("Baseline RAGAS mean scores:", scores)


if __name__ == "__main__":
    main()
