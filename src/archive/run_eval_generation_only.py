"""Run generation-only evaluation using gold contexts (no retrieval).

Uses contexts from `settings.eval_csv` as the only context source,
generates answers, then scores with the existing LLM judge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from langchain_groq import ChatGroq
from tqdm import tqdm

from src.config import get_settings, require_groq_api_key
from src.llm_judge import METRIC_KEYS, score_answer
from src.rag_pipeline import PROMPT


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


def _generate_from_gold_contexts(question: str, contexts: list[str]) -> tuple[str, int, str]:
    settings = get_settings()
    api_key = require_groq_api_key(settings)
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=api_key,
        temperature=settings.temperature,
    )
    joined_context = "\n\n".join(contexts)
    prompt_value = PROMPT.format_prompt(question=question, context=joined_context)
    started_at = time.perf_counter()
    answer = str(llm.invoke(prompt_value.to_messages()).content)
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return answer, latency_ms, settings.groq_model


def main() -> None:
    settings = get_settings()
    eval_path = Path(settings.eval_csv)
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    eval_df = pd.read_csv(eval_path)
    required_cols = {"id", "question", "ground_truth", "contexts", "category"}
    if not required_cols.issubset(eval_df.columns):
        raise ValueError(f"Eval CSV must contain: {sorted(required_cols)}")

    run_id = str(uuid4())
    rows = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Generation-only eval"):
        contexts = _parse_contexts(row["contexts"])
        answer, latency_ms, model_name = _generate_from_gold_contexts(
            question=str(row["question"]),
            contexts=contexts,
        )
        retrieved_contexts = [
            {
                "doc_id": "gold_context",
                "chunk_id": f"gold_{idx}",
                "text": text,
                "source_type": "gold",
                "source_path": str(eval_path),
            }
            for idx, text in enumerate(contexts, start=1)
        ]
        judged = score_answer(
            question=str(row["question"]),
            answer=answer,
            ground_truth=str(row["ground_truth"]),
            retrieved_contexts=retrieved_contexts,
        )

        out: dict[str, Any] = {
            "id": row["id"],
            "question": row["question"],
            "category": row.get("category", ""),
            "ground_truth": row["ground_truth"],
            "gold_contexts": json.dumps(contexts),
            "answer": answer,
            "latency_ms": latency_ms,
            "model_name": model_name,
            "judge_model": judged["judge_model"],
            "judge_mean_1_to_5": judged["mean_score_1_to_5"],
            "judge_mean_0_to_1": judged["mean_score_0_to_1"],
            "run_id": run_id,
        }
        for metric in METRIC_KEYS:
            out[f"{metric}_score"] = judged["metrics"][metric]["score"]
            out[f"{metric}_rationale"] = judged["metrics"][metric]["rationale"]
        rows.append(out)

    result_df = pd.DataFrame(rows)
    _ensure_parent(settings.generation_only_csv)
    result_df.to_csv(settings.generation_only_csv, index=False)

    summary = {
        "input_eval_csv": str(eval_path),
        "generation_only_csv": settings.generation_only_csv,
        "total_rows": int(len(result_df)),
        "judge_mean_1_to_5": round(float(result_df["judge_mean_1_to_5"].mean()), 4),
        "judge_mean_0_to_1": round(float(result_df["judge_mean_0_to_1"].mean()), 4),
        "correctness_mean_1_to_5": round(float(result_df["correctness_score"].mean()), 4),
    }
    _ensure_parent(settings.generation_only_json)
    with open(settings.generation_only_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved generation-only rows -> {settings.generation_only_csv}")
    print(f"Saved generation-only summary -> {settings.generation_only_json}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
