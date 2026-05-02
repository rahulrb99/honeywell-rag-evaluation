from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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
        "Add one to graphrag_engine/.env or the project root .env."
    )
