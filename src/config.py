from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    vectorstore_dir: str
    raw_data_dir: str = "data/raw"
    eval_csv: str = "data/eval/week1_gold_triplets.csv"
    predictions_csv: str = "outputs/week1_predictions.csv"
    ragas_scores_json: str = "outputs/week1_ragas_scores.json"


def get_settings() -> Settings:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and set your key."
        )

    return Settings(
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "4")),
        vectorstore_dir=os.getenv("VECTORSTORE_DIR", "data/vectorstore"),
    )
