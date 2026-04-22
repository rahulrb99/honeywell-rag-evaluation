"""Groundedness analysis: attribute retrieved chunks to generated answers.

Uses dual-threshold cosine similarity to avoid length-bias failures:
  >= 0.65  → chunk is "used" by the answer
  <= 0.35  → chunk is "not used"
  between  → "uncertain" (not tagged as failure)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import get_settings

_HIGH_THRESHOLD = 0.65
_LOW_THRESHOLD = 0.35

_EMBEDDER: HuggingFaceEmbeddings | None = None


def _get_embedder() -> HuggingFaceEmbeddings:
    global _EMBEDDER
    if _EMBEDDER is None:
        settings = get_settings()
        _EMBEDDER = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _EMBEDDER


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _used_status(max_sim: float) -> bool | str:
    if max_sim >= _HIGH_THRESHOLD:
        return True
    if max_sim <= _LOW_THRESHOLD:
        return False
    return "uncertain"


def analyse_row(
    answer: str,
    retrieved_contexts: list[dict[str, Any]],
    correctness_score: int | None = None,
) -> dict[str, Any]:
    """Return groundedness attributes for a single prediction row.

    Args:
        answer: The RAG-generated answer text.
        retrieved_contexts: List of context dicts with at least a "text" key.
        correctness_score: Judge correctness score (1-5), or None if unavailable.

    Returns:
        Dict with keys: max_chunk_sim, used_status, used_chunk_ids,
        top_chunk_text, failure_tag.
    """
    embedder = _get_embedder()

    if not retrieved_contexts:
        return {
            "max_chunk_sim": 0.0,
            "used_status": False,
            "used_chunk_ids": [],
            "top_chunk_text": "",
            "failure_tag": "retrieval_miss" if (correctness_score is not None and correctness_score < 3) else "correct",
        }

    answer_emb = embedder.embed_query(answer)

    sims: list[float] = []
    for ctx in retrieved_contexts:
        chunk_text = str(ctx.get("text", "")).strip()
        if not chunk_text:
            sims.append(0.0)
            continue
        chunk_emb = embedder.embed_query(chunk_text)
        sims.append(_cosine_sim(answer_emb, chunk_emb))

    max_sim = max(sims)
    best_idx = int(np.argmax(sims))
    used = _used_status(max_sim)

    used_chunk_ids = [
        retrieved_contexts[i].get("chunk_id", str(i))
        for i, s in enumerate(sims)
        if _used_status(s) is True
    ]

    best_ctx = retrieved_contexts[best_idx]
    top_chunk_text = str(best_ctx.get("text", ""))[:200].replace("\n", " ")

    failure_tag = _classify_failure(used, correctness_score)

    return {
        "max_chunk_sim": round(max_sim, 4),
        "used_status": used,
        "used_chunk_ids": used_chunk_ids,
        "top_chunk_text": top_chunk_text,
        "failure_tag": failure_tag,
    }


def _classify_failure(
    used: bool | str,
    correctness_score: int | None,
) -> str:
    if correctness_score is None or correctness_score >= 4:
        return "correct"
    # correctness_score < 3 is a confirmed failure
    if correctness_score >= 3:
        return "correct"
    if used is True:
        return "generation_ignore"
    if used is False:
        return "retrieval_miss"
    return "ambiguous"
