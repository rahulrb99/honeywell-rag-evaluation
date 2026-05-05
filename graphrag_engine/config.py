import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def _get_secret(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key, "")
        except Exception:
            pass
    return val


OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
NEO4J_URI = _get_secret("NEO4J_URI")
NEO4J_USERNAME = _get_secret("NEO4J_USERNAME")
NEO4J_PASSWORD = _get_secret("NEO4J_PASSWORD")
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

PROJECT_DIR = Path(__file__).parent
PROMPTS_DIR = PROJECT_DIR / "prompts"

KNOWN_MODEL_LIMITS = {
    "gpt-4o":       {"context_window": 128_000,   "doc_budget": 90_000},
    "gpt-4o-mini":  {"context_window": 128_000,   "doc_budget": 90_000},
    "gpt-4.1":      {"context_window": 1_000_000, "doc_budget": 900_000},
    "gpt-4.1-mini": {"context_window": 1_000_000, "doc_budget": 900_000},
    "gpt-4.1-nano": {"context_window": 1_000_000, "doc_budget": 900_000},
}

TIKTOKEN_ENCODING = "cl100k_base"
BATCH_TOKEN_BUDGET = 30_000
CHUNK_TOKEN_LIMIT = 2_000
ACCEPTED_FILE_TYPES = ["txt", "pdf", "docx"]

WEB_MAX_ARTICLES = 5
WEB_SIMILARITY_THRESHOLD = 0.5


def make_llm_client() -> OpenAI:
    """Return an OpenAI-compatible client.

    Prefers OPENAI_API_KEY when set; falls back to GROQ_API_KEY routed
    through Groq's OpenAI-compatible endpoint.
    """
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)
    if GROQ_API_KEY:
        return OpenAI(api_key=GROQ_API_KEY, base_url=_GROQ_BASE_URL)
    raise ValueError(
        "Neither OPENAI_API_KEY nor GROQ_API_KEY is set. "
        "Add one to the project root .env."
    )
