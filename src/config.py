from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    groq_api_key: Optional[str]
    openai_api_key: Optional[str]
    llm_provider: str
    groq_model: str
    openai_chat_model: str
    judge_model: str
    temperature: float
    judge_temperature: float
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    fetch_k: int
    lambda_mult: float
    log_retrieved_contexts: bool
    vectorstore_dir: str
    product_records_path: str
    raw_data_dir: str = "data/raw"
    eval_csv: str = "data/eval/archive/week1_gold_triplets_20.csv"
    predictions_csv: str = "outputs/debug/week1_rag_outputs.csv"
    ragas_scores_json: str = "outputs/debug/week1_ragas_scores.json"
    judge_scores_json: str = "outputs/debug/week1_judge_scores.json"
    judge_scores_csv: str = "outputs/debug/week1_judge_scores.csv"
    judge_consistency_json: str = "outputs/debug/week1_judge_consistency.json"
    judge_compare_json: str = "outputs/debug/week1_judge_vs_ragas.json"
    groundedness_csv: str = "outputs/debug/week1_groundedness.csv"
    retrieval_recall_csv: str = "outputs/debug/week1_retrieval_metrics.csv"
    retrieval_only_csv: str = "outputs/debug/week1_retrieval_metrics.csv"
    retrieval_only_json: str = "outputs/debug/week1_retrieval_only.json"
    generation_only_csv: str = "outputs/debug/week1_generation_only.csv"
    generation_only_json: str = "outputs/debug/week1_generation_only.json"
    eval_rows_csv: str = "outputs/week1_eval_rows.csv"
    eval_summary_json: str = "outputs/week1_eval_summary.json"
    failure_dashboard_html: str = "outputs/dashboard/failure_dashboard.html"
    experiments_dir: str = "outputs/experiments"


def get_settings() -> Settings:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    return Settings(
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        llm_provider=os.getenv(
            "PIPELINE_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "groq")
        ).strip().lower(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        openai_chat_model=os.getenv(
            "PIPELINE_OPENAI_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        ),
        judge_model=os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("TEMPERATURE", "0")),
        judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "0")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "5")),
        fetch_k=int(os.getenv("FETCH_K", "20")),
        lambda_mult=float(os.getenv("LAMBDA_MULT", "0.7")),
        log_retrieved_contexts=os.getenv("LOG_RETRIEVED_CONTEXTS", "true").lower()
        in {"1", "true", "yes", "on"},
        vectorstore_dir=os.getenv("VECTORSTORE_DIR", "data/vectorstore"),
        product_records_path=os.getenv("PRODUCT_RECORDS_PATH", "outputs/structured/product_records.json"),
        eval_csv=os.getenv("EVAL_CSV", "data/eval/archive/week1_gold_triplets_20.csv"),
        predictions_csv=os.getenv("PREDICTIONS_CSV", "outputs/debug/week1_rag_outputs.csv"),
        judge_scores_json=os.getenv("JUDGE_SCORES_JSON", "outputs/debug/week1_judge_scores.json"),
        judge_scores_csv=os.getenv("JUDGE_SCORES_CSV", "outputs/debug/week1_judge_scores.csv"),
        judge_consistency_json=os.getenv(
            "JUDGE_CONSISTENCY_JSON", "outputs/debug/week1_judge_consistency.json"
        ),
        judge_compare_json=os.getenv("JUDGE_COMPARE_JSON", "outputs/debug/week1_judge_vs_ragas.json"),
        groundedness_csv=os.getenv("GROUNDEDNESS_CSV", "outputs/debug/week1_groundedness.csv"),
        retrieval_recall_csv=os.getenv(
            "RETRIEVAL_RECALL_CSV", "outputs/debug/week1_retrieval_metrics.csv"
        ),
        retrieval_only_csv=os.getenv("RETRIEVAL_ONLY_CSV", "outputs/debug/week1_retrieval_metrics.csv"),
        retrieval_only_json=os.getenv(
            "RETRIEVAL_ONLY_JSON", "outputs/debug/week1_retrieval_only.json"
        ),
        generation_only_csv=os.getenv(
            "GENERATION_ONLY_CSV", "outputs/debug/week1_generation_only.csv"
        ),
        generation_only_json=os.getenv(
            "GENERATION_ONLY_JSON", "outputs/debug/week1_generation_only.json"
        ),
        eval_rows_csv=os.getenv("EVAL_ROWS_CSV", "outputs/week1_eval_rows.csv"),
        eval_summary_json=os.getenv(
            "EVAL_SUMMARY_JSON", "outputs/week1_eval_summary.json"
        ),
        failure_dashboard_html=os.getenv(
            "FAILURE_DASHBOARD_HTML", "outputs/dashboard/failure_dashboard.html"
        ),
        experiments_dir=os.getenv("EXPERIMENTS_DIR", "outputs/experiments"),
    )


def require_groq_api_key(settings: Settings) -> str:
    if settings.groq_api_key:
        return settings.groq_api_key
    raise ValueError(
        "GROQ_API_KEY is missing. Copy .env.example to .env and set your key."
    )


def require_openai_api_key(settings: Settings) -> str:
    if settings.openai_api_key:
        return settings.openai_api_key
    raise ValueError(
        "OPENAI_API_KEY is missing. Set it in .env or use LLM_PROVIDER=groq."
    )


def selected_chat_model(settings: Settings, *, judge: bool = False) -> str:
    if settings.llm_provider == "openai":
        return settings.openai_chat_model
    return settings.judge_model if judge else settings.groq_model


def make_chat_llm(settings: Settings, *, judge: bool = False) -> Any:
    """Create the configured chat model without leaking provider details to callers."""
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=require_openai_api_key(settings),
            temperature=settings.judge_temperature if judge else settings.temperature,
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.judge_model if judge else settings.groq_model,
        api_key=require_groq_api_key(settings),
        temperature=settings.judge_temperature if judge else settings.temperature,
        max_retries=int(os.getenv("GROQ_MAX_RETRIES", "8")),
    )
