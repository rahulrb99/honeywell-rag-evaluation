from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import get_settings, require_groq_api_key


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict evaluator for RAG responses.
Score each metric from 1-5 where 1 is very poor and 5 is excellent.
Return ONLY valid JSON with this exact structure:
{{
  "faithfulness": {{"score": <int 1-5>, "rationale": "<string>"}},
  "answer_relevance": {{"score": <int 1-5>, "rationale": "<string>"}},
  "completeness": {{"score": <int 1-5>, "rationale": "<string>"}},
  "correctness": {{"score": <int 1-5>, "rationale": "<string>"}},
  "conciseness": {{"score": <int 1-5>, "rationale": "<string>"}}
}}

Rubric:
- faithfulness: Answer is supported by retrieved contexts without hallucinations.
- answer_relevance: Answer addresses the user question directly and stays on-topic.
- completeness: Answer covers the key information expected from available evidence.
- correctness: Answer aligns with ground-truth reference when one is provided.
- conciseness: Answer is clear and compact without unnecessary text.

Question:
{question}

Answer:
{answer}

Ground truth reference:
{ground_truth}

Retrieved contexts:
{retrieved_contexts}
"""
)

METRIC_KEYS = (
    "faithfulness",
    "answer_relevance",
    "completeness",
    "correctness",
    "conciseness",
)


@dataclass(frozen=True)
class MetricScore:
    score: int
    rationale: str


def _extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        recovered = _recover_from_near_json(raw)
        if recovered:
            return recovered
        raise


def _recover_from_near_json(text: str) -> dict[str, Any]:
    recovered: dict[str, Any] = {}
    for key in METRIC_KEYS:
        pattern = rf'"{re.escape(key)}"\s*:\s*\{{(.*?)\}}'
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        block = match.group(1)
        score_match = re.search(r'"score"\s*:\s*([1-5])', block)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', block, flags=re.DOTALL)
        score = int(score_match.group(1)) if score_match else 1
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        recovered[key] = {"score": score, "rationale": rationale}
    return recovered


def _sanitize_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 1
    return min(5, max(1, score))


def _normalize_result(parsed: dict[str, Any]) -> dict[str, MetricScore]:
    out: dict[str, MetricScore] = {}
    for key in METRIC_KEYS:
        section = parsed.get(key, {})
        if not isinstance(section, dict):
            section = {}
        out[key] = MetricScore(
            score=_sanitize_score(section.get("score", 1)),
            rationale=str(section.get("rationale", "")).strip()[:400],
        )
    return out


def _contexts_as_text(retrieved_contexts: list[dict[str, Any]]) -> str:
    rows = []
    for idx, item in enumerate(retrieved_contexts, start=1):
        doc_id = item.get("doc_id", "")
        chunk_id = item.get("chunk_id", "")
        text = str(item.get("text", "")).strip()
        rows.append(f"[{idx}] doc_id={doc_id} chunk_id={chunk_id}\n{text}")
    return "\n\n".join(rows)


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
            wait_seconds = min(5.0, 0.4 * attempt)
            time.sleep(wait_seconds)
    raise RuntimeError("Judge invocation failed after retries.")


def score_answer(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    retrieved_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = get_settings()
    api_key = require_groq_api_key(settings)
    prompt_value = JUDGE_PROMPT.format_prompt(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        retrieved_contexts=_contexts_as_text(retrieved_contexts),
    )
    model_used = settings.judge_model
    try:
        llm = ChatGroq(
            model=settings.judge_model,
            api_key=api_key,
            temperature=settings.judge_temperature,
        )
        raw = _invoke_with_retry(llm, prompt_value.to_messages())
    except Exception as exc:
        message = str(exc).lower()
        if "model_decommissioned" not in message and "decommissioned" not in message:
            raise
        model_used = settings.groq_model
        fallback_llm = ChatGroq(
            model=settings.groq_model,
            api_key=api_key,
            temperature=settings.judge_temperature,
        )
        raw = _invoke_with_retry(fallback_llm, prompt_value.to_messages())
    parsed = _extract_json(raw)
    normalized = _normalize_result(parsed)
    mean_score = sum(metric.score for metric in normalized.values()) / len(METRIC_KEYS)
    return {
        "metrics": {
            key: {"score": value.score, "rationale": value.rationale}
            for key, value in normalized.items()
        },
        "mean_score_1_to_5": round(mean_score, 4),
        "mean_score_0_to_1": round((mean_score - 1) / 4, 4),
        "judge_model": model_used,
    }

