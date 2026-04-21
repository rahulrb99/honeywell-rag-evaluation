from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tqdm import tqdm

from src.config import get_settings, require_groq_api_key
from src.llm_judge import METRIC_KEYS, score_answer
from src.rag_pipeline import answer_question


PARAPHRASE_PROMPT = ChatPromptTemplate.from_template(
    """Generate {count} paraphrases of the question below.
Keep meaning unchanged and keep domain-specific product names or IDs exactly.
Return only valid JSON array of strings.

Question:
{question}
"""
)


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_json_array(text: str) -> list[str]:
    raw = text.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
        else:
            parsed = _parse_list_fallback(raw)
    if not isinstance(parsed, list):
        return _parse_list_fallback(raw)
    return [str(item).strip() for item in parsed if str(item).strip()]


def _parse_list_fallback(raw: str) -> list[str]:
    lines = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-").strip()
        cleaned = cleaned.lstrip("0123456789. )(").strip()
        cleaned = cleaned.strip('"').strip("'").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _generate_paraphrases(question: str, count: int) -> list[str]:
    settings = get_settings()
    api_key = require_groq_api_key(settings)
    prompt_value = PARAPHRASE_PROMPT.format_prompt(question=question, count=count)
    messages = prompt_value.to_messages()
    try:
        llm = ChatGroq(
            model=settings.judge_model,
            api_key=api_key,
            temperature=settings.judge_temperature,
        )
        raw = _invoke_with_retry(llm, messages)
    except Exception as exc:
        if "decommissioned" not in str(exc).lower():
            raise
        fallback_llm = ChatGroq(
            model=settings.groq_model,
            api_key=api_key,
            temperature=settings.judge_temperature,
        )
        raw = _invoke_with_retry(fallback_llm, messages)
    paraphrases = _parse_json_array(raw)
    # ensure consistent count and uniqueness while preserving order
    seen = set()
    out = []
    for item in paraphrases:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) == count:
            break
    return out


def _invoke_with_retry(llm: ChatGroq, messages: list[Any], attempts: int = 6) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return str(llm.invoke(messages).content)
        except Exception as exc:
            text = str(exc).lower()
            if "rate_limit" not in text and "429" not in text:
                raise
            if attempt == attempts:
                raise
            time.sleep(min(5.0, 0.4 * attempt))
    raise RuntimeError("Paraphrase generation failed after retries.")


def _normalize_judge_means(metric_means_1_to_5: dict[str, float]) -> dict[str, float]:
    return {
        metric.replace("_mean_1_to_5", "_mean_0_to_1"): round((value - 1.0) / 4.0, 4)
        for metric, value in metric_means_1_to_5.items()
        if metric.endswith("_mean_1_to_5")
    }


def main() -> None:
    settings = get_settings()
    eval_df = pd.read_csv(settings.eval_csv)
    paraphrase_count = int(os.getenv("PARAPHRASE_VARIANTS", "5"))
    max_questions = int(os.getenv("CONSISTENCY_QUESTIONS", "5"))
    selected = eval_df.head(max_questions).copy()

    rows: list[dict[str, Any]] = []
    for _, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="Consistency paraphrase run",
    ):
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])
        variants = [question] + _generate_paraphrases(question, paraphrase_count)
        for idx, variant in enumerate(variants):
            rag_output = answer_question(variant)
            judged = score_answer(
                question=variant,
                answer=rag_output["answer"],
                ground_truth=ground_truth,
                retrieved_contexts=rag_output["retrieved_contexts"],
            )
            out: dict[str, Any] = {
                "base_id": row.get("id", ""),
                "base_question": question,
                "variant_index": idx,
                "variant_question": variant,
                "is_original_question": idx == 0,
                "answer": rag_output["answer"],
                "judge_mean_1_to_5": judged["mean_score_1_to_5"],
                "judge_mean_0_to_1": judged["mean_score_0_to_1"],
            }
            for metric in METRIC_KEYS:
                out[f"{metric}_score"] = judged["metrics"][metric]["score"]
            rows.append(out)

    consistency_df = pd.DataFrame(rows)
    metric_columns = [f"{metric}_score" for metric in METRIC_KEYS] + ["judge_mean_1_to_5"]
    variance = (
        consistency_df.groupby("base_id")[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    # flatten columns for export readability
    variance.columns = [
        "base_id" if col[0] == "base_id" else f"{col[0]}_{col[1]}" for col in variance.columns
    ]

    judge_means = {
        f"{metric}_mean_1_to_5": round(float(consistency_df[f"{metric}_score"].mean()), 4)
        for metric in METRIC_KEYS
    }
    judge_means["judge_mean_1_to_5"] = round(float(consistency_df["judge_mean_1_to_5"].mean()), 4)
    judge_means_0_to_1 = _normalize_judge_means(judge_means)
    judge_means_0_to_1["judge_mean_0_to_1"] = round(
        float(consistency_df["judge_mean_0_to_1"].mean()), 4
    )

    ragas_scores: dict[str, float] = {}
    ragas_path = Path(settings.ragas_scores_json)
    if ragas_path.exists():
        ragas_scores = json.loads(ragas_path.read_text(encoding="utf-8"))

    mapping = {
        "faithfulness_mean_0_to_1": "faithfulness",
        "answer_relevance_mean_0_to_1": "answer_relevancy",
    }
    agreement = {}
    for judge_key, ragas_key in mapping.items():
        if judge_key in judge_means_0_to_1 and ragas_key in ragas_scores:
            delta = round(judge_means_0_to_1[judge_key] - float(ragas_scores[ragas_key]), 4)
            agreement[judge_key] = {
                "judge": judge_means_0_to_1[judge_key],
                "ragas": round(float(ragas_scores[ragas_key]), 4),
                "delta": delta,
                "agrees": abs(delta) <= 0.1,
            }

    # context/coverage proxy mapping for comparison commentary
    if "completeness_mean_0_to_1" in judge_means_0_to_1 and "context_recall" in ragas_scores:
        delta = round(
            judge_means_0_to_1["completeness_mean_0_to_1"] - float(ragas_scores["context_recall"]),
            4,
        )
        agreement["completeness_vs_context_recall"] = {
            "judge": judge_means_0_to_1["completeness_mean_0_to_1"],
            "ragas": round(float(ragas_scores["context_recall"]), 4),
            "delta": delta,
            "agrees": abs(delta) <= 0.1,
        }

    _ensure_parent(settings.judge_consistency_json)
    consistency_csv = str(Path(settings.judge_consistency_json).with_suffix(".csv"))
    consistency_df.to_csv(consistency_csv, index=False)
    variance_csv = str(Path(settings.judge_consistency_json).with_name("week1_judge_consistency_variance.csv"))
    variance.to_csv(variance_csv, index=False)

    payload = {
        "questions_tested": int(len(selected)),
        "variants_per_question_including_original": int(paraphrase_count + 1),
        "consistency_rows_csv": consistency_csv,
        "variance_csv": variance_csv,
        "judge_metric_means_1_to_5": judge_means,
        "judge_metric_means_0_to_1": judge_means_0_to_1,
        "ragas_scores": ragas_scores,
        "judge_vs_ragas_agreement": agreement,
    }
    with open(settings.judge_consistency_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _ensure_parent(settings.judge_compare_json)
    with open(settings.judge_compare_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_metric_means_0_to_1": judge_means_0_to_1,
                "ragas_scores": ragas_scores,
                "agreement": agreement,
            },
            f,
            indent=2,
        )

    print(f"Saved consistency JSON -> {settings.judge_consistency_json}")
    print(f"Saved comparison JSON -> {settings.judge_compare_json}")
    print("Agreement summary:", agreement)


if __name__ == "__main__":
    main()

