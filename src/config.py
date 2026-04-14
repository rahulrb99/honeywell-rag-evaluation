from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: Optional[str]
    groq_model: str
    temperature: float
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    fetch_k: int
    lambda_mult: float
    log_retrieved_contexts: bool
    vectorstore_dir: str
    raw_data_dir: str = "data/raw"
    eval_csv: str = "data/eval/week1_gold_triplets.csv"
    predictions_csv: str = "outputs/week1_predictions.csv"
    ragas_scores_json: str = "outputs/week1_ragas_scores.json"


def get_settings() -> Settings:
    api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    return Settings(
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=float(os.getenv("TEMPERATURE", "0")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "5")),
        fetch_k=int(os.getenv("FETCH_K", "20")),
        lambda_mult=float(os.getenv("LAMBDA_MULT", "0.7")),
        log_retrieved_contexts=os.getenv("LOG_RETRIEVED_CONTEXTS", "true").lower()
        in {"1", "true", "yes", "on"},
        vectorstore_dir=os.getenv("VECTORSTORE_DIR", "data/vectorstore"),
    )


def require_groq_api_key(settings: Settings) -> str:
    if settings.groq_api_key:
        return settings.groq_api_key
    raise ValueError(
        "GROQ_API_KEY is missing. Copy .env.example to .env and set your key."
    )
