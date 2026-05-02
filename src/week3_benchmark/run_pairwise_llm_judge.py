from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tqdm import tqdm

from src.config import get_settings, require_groq_api_key
from src.week3_benchmark.run_week3_eval import _context_texts


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_OUT_CSV = ROOT / "outputs" / "eval_outputs" / "pairwise_llm_judge.csv"
DEFAULT_OUT_JSON = ROOT / "outputs" / "eval_outputs" / "pairwise_llm_judge_summary.json"

PROMPT = ChatPromptTemplate.from_template(
    """You are a strict evaluator comparing two RAG answers.
Choose the better answer using only the question, reference answer, and retrieved contexts.

Return ONLY valid JSON with this exact structure:
{{
  "winner": "graph" | "vector" | "tie" | "neither",
  "confidence": <float from 0 to 1>,
  "rationale": "<one concise sentence>"
}}

Rubric:
- Prefer answers that match the reference answer.
- Prefer answers supported by their retrieved contexts.
- Penalize hallucinations, unsupported claims, omissions, and dodged answers.
- Use "tie" if both are similarly correct.
- Use "neither" if both are materially wrong or unsupported.

Question:
{question}

Reference answer:
{ground_truth}

GraphRAG answer:
{graph_answer}

GraphRAG retrieved contexts:
{graph_contexts}

Vector RAG answer:
{vector_answer}

Vector RAG retrieved contexts:
{vector_contexts}
"""
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise LLM judge for Week 3 outputs.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    return parser.parse_args()


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_winner(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"graph", "graphrag", "graph_rag"}:
        return "graph"
    if raw in {"vector", "vector_rag", "baseline"}:
        return "vector"
    if raw in {"tie", "both"}:
        return "tie"
    if raw in {"neither", "none", "no_winner"}:
        return "neither"
    return "neither"


def _invoke_with_retry(llm: ChatGroq, messages: list[Any], attempts: int = 6) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return str(llm.invoke(messages).content)
        except Exception as exc:
            message = str(exc).lower()
            if "rate_limit" not in message and "429" not in message:
                raise
            if attempt == attempts:
                raise
            time.sleep(min(5.0, 0.5 * attempt))
    raise RuntimeError("Judge invocation failed after retries.")


def _format_contexts(value: object) -> str:
    contexts = _context_texts(value)
    return "\n\n".join(f"[{idx}] {text}" for idx, text in enumerate(contexts, start=1))


def _judge_row(llm: ChatGroq, row: pd.Series) -> dict[str, Any]:
    prompt_value = PROMPT.format_prompt(
        question=str(row.get("question", "")),
        ground_truth=str(row.get("ground_truth", "")),
        graph_answer=str(row.get("graph_answer", "")),
        graph_contexts=_format_contexts(row.get("graph_retrieved_contexts", "")),
        vector_answer=str(row.get("vector_answer", "")),
        vector_contexts=_format_contexts(row.get("vector_retrieved_contexts", "")),
    )
    parsed = _extract_json(_invoke_with_retry(llm, prompt_value.to_messages()))
    winner = _normalize_winner(parsed.get("winner"))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "id": row["id"],
        "question": row.get("question", ""),
        "query_class": row.get("query_class", ""),
        "winner": winner,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "metric_winner": row.get("winner", ""),
        "judge_metric_agree": winner == row.get("winner", ""),
        "rationale": str(parsed.get("rationale", "")).strip()[:500],
    }


def main() -> None:
    args = _parse_args()
    results_path = Path(args.results)
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    if not results_path.exists():
        raise FileNotFoundError(f"Missing Week 3 results CSV: {results_path}")

    settings = get_settings()
    api_key = require_groq_api_key(settings)
    llm = ChatGroq(
        model=settings.judge_model,
        api_key=api_key,
        temperature=settings.judge_temperature,
    )

    results = pd.read_csv(results_path)
    rows = [_judge_row(llm, row) for _, row in tqdm(results.iterrows(), total=len(results))]
    judged = pd.DataFrame(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    judged.to_csv(out_csv, index=False)

    counts = judged["winner"].value_counts().to_dict()
    summary = {
        "rows": int(len(judged)),
        "judge_model": settings.judge_model,
        "winner_counts": counts,
        "judge_metric_agreement_rate": round(float(judged["judge_metric_agree"].mean()), 4),
        "mean_confidence": round(float(judged["confidence"].mean()), 4),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved pairwise LLM judge rows -> {out_csv}")
    print(f"Saved pairwise LLM judge summary -> {out_json}")
    print(f"LLM judge winner counts: {counts}")
    print(f"Judge vs metric agreement: {summary['judge_metric_agreement_rate']}")


if __name__ == "__main__":
    main()
