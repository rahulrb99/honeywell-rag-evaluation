from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from openai import OpenAI

from .config import DEFAULT_MODEL, USE_OPENAI_QA


QA_PROMPT = """Answer the question using only the supplied graph evidence and chunks.
Return JSON with keys "answer" and "citations".

Question:
{question}

Graph evidence:
{evidence}

Supporting chunks:
{chunks}
"""


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return {"answer": text.strip(), "citations": []}


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 1 and term not in _STOPWORDS]


def _entity_terms(retrieval_result: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for entity in retrieval_result.get("question_entities", []):
        terms.extend(_terms(str(entity)))
    for entity in retrieval_result.get("shortest_path", []):
        terms.extend(_terms(str(entity)))
    return terms


def _edge_chunk_counts(retrieval_result: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, _, data in retrieval_result.get("retrieved_edges", []):
        raw = str(data.get("supporting_chunk_ids", "") or "")
        for chunk_id in raw.split("|"):
            if chunk_id:
                counts[chunk_id] += 1
    return counts


def _index_like_penalty(text: str) -> float:
    lower = text[:350].lower()
    numeric_tokens = len(re.findall(r"\b\d{1,3}\b", text[:1200]))
    if " index " in f" {lower} " or lower.startswith("index ") or numeric_tokens >= 35:
        return 0.25
    return 1.0


def _rank_supporting_chunks(
    question: str,
    retrieval_result: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, str]]:
    query_terms = Counter(_terms(question) + _entity_terms(retrieval_result))
    edge_counts = _edge_chunk_counts(retrieval_result)
    shortest_path_terms = set(_terms(" ".join(str(node) for node in retrieval_result.get("shortest_path", []))))
    ranked: list[tuple[float, int, str, str]] = []

    for index, chunk_id in enumerate(retrieval_result.get("supporting_chunk_ids", [])):
        text = str(chunks_by_id.get(chunk_id, {}).get("text", ""))
        if not text:
            continue
        chunk_terms = Counter(_terms(text))
        lexical_score = sum(min(count, chunk_terms.get(term, 0)) for term, count in query_terms.items())
        sentence_scores = [
            sum(min(count, Counter(_terms(sentence)).get(term, 0)) for term, count in query_terms.items())
            for sentence in re.split(r"(?<=[.!?])\s+", text)
        ]
        best_sentence_score = max(sentence_scores or [0])
        path_score = sum(1 for term in shortest_path_terms if chunk_terms.get(term, 0))
        graph_score = min(edge_counts.get(chunk_id, 0), 3)
        score = (lexical_score + (2 * best_sentence_score) + (2 * path_score) + graph_score) * _index_like_penalty(text)
        ranked.append((float(score), -index, str(chunk_id), text))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [{"chunk_id": chunk_id, "text": text} for score, _, chunk_id, text in ranked[:limit] if score > 0] or [
        {"chunk_id": chunk_id, "text": text} for _, _, chunk_id, text in ranked[:limit]
    ]


def _evidence(retrieval_result: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"source": str(s), "relation": str(data.get("relation", "")), "target": str(t)}
        for s, t, data in retrieval_result.get("retrieved_edges", [])
    ][:100]


def _answer_from_retrieval(
    question: str,
    retrieval_result: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    evidence = _evidence(retrieval_result)
    chunks = _rank_supporting_chunks(question, retrieval_result, chunks_by_id, limit=5)
    if not USE_OPENAI_QA or not os.getenv("OPENAI_API_KEY"):
        return {"answer": _heuristic_answer(question, chunks), "citations": [c["chunk_id"] for c in chunks]}
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": QA_PROMPT.format(
                    question=question,
                    evidence=json.dumps(evidence, ensure_ascii=True),
                    chunks=json.dumps(chunks, ensure_ascii=True),
                ),
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return _extract_json(response.choices[0].message.content or "{}")


def answer_graph_a(
    question: str,
    retrieval_result: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    return _answer_from_retrieval(question, retrieval_result, chunks_by_id, model=model)


def answer_graph_b(
    question: str,
    retrieval_result: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    return _answer_from_retrieval(question, retrieval_result, chunks_by_id, model=model)


def is_correct(predicted: str, gold: str) -> bool:
    pred = re.sub(r"[^a-z0-9]+", " ", predicted.lower()).strip()
    expected = re.sub(r"[^a-z0-9]+", " ", gold.lower()).strip()
    return bool(expected and (expected in pred or pred in expected))


def _heuristic_answer(question: str, chunks: list[dict[str, str]]) -> str:
    query_terms = Counter(_terms(question))
    snippets: list[tuple[float, str]] = []
    for chunk in chunks[:5]:
        text = str(chunk.get("text", ""))
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        for position, sentence in enumerate(sentences or [text]):
            sentence_terms = Counter(_terms(sentence))
            overlap = sum(min(count, sentence_terms.get(term, 0)) for term, count in query_terms.items())
            score = overlap + (1 / (position + 1))
            snippets.append((score, sentence))
    selected = [sentence for score, sentence in sorted(snippets, key=lambda item: item[0], reverse=True)[:3] if sentence]
    return " ".join(selected)[:700]
