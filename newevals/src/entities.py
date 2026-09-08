from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from .config import DEFAULT_MODEL, ENTITY_CACHE_PATH, QUESTION_ENTITY_CACHE_PATH, USE_OPENAI_ENTITY_EXTRACTION
from .io_utils import read_json, read_jsonl, write_json
from .normalize import normalize_entities


ENTITY_KEYS = ("products", "protocols", "voltages", "components", "error_codes")

ENTITY_PROMPT = """Extract Honeywell technical entities from the text.
Return only JSON with keys: products, protocols, voltages, components, error_codes.
Each value must be an array of short strings.

Text:
{text}
"""


def _extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def heuristic_extract_entities(text: str) -> dict[str, list[str]]:
    protocols = re.findall(r"\b(?:BACnet|Modbus|LonWorks|SLC|NFC|Wi-?Fi|Ethernet)\b", text, flags=re.I)
    voltages = re.findall(r"\b\d+(?:\.\d+)?\s*(?:V|VAC|VDC|volts?)\b", text, flags=re.I)
    error_codes = re.findall(r"\b(?:error|fault|code)\s*[A-Z]?\d{2,5}\b|\bE\d{2,5}\b", text, flags=re.I)
    components = re.findall(
        r"\b(?:controller|sensor|module|amplifier|speaker|strobe|microphone|telephone|expander|panel|circuit|loop)\b",
        text,
        flags=re.I,
    )
    products = re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", text)
    return {
        "products": normalize_entities(products),
        "protocols": normalize_entities(protocols),
        "voltages": normalize_entities(voltages),
        "components": normalize_entities(components),
        "error_codes": normalize_entities(error_codes),
    }


def _normalize_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    return {
        key: normalize_entities([str(item) for item in payload.get(key, []) if str(item).strip()])
        for key in ENTITY_KEYS
    }


def extract_entities_openai(text: str, model: str = DEFAULT_MODEL, client: OpenAI | None = None) -> dict[str, list[str]]:
    if (not USE_OPENAI_ENTITY_EXTRACTION or not os.getenv("OPENAI_API_KEY")) and client is None:
        return heuristic_extract_entities(text)
    client = client or OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": ENTITY_PROMPT.format(text=text[:6000])}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return _normalize_payload(_extract_json(content))


def flatten_entities(entity_payload: dict[str, list[str]]) -> list[str]:
    return normalize_entities([item for key in ENTITY_KEYS for item in entity_payload.get(key, [])])


def build_entity_cache(chunks: list[dict[str, Any]], model: str = DEFAULT_MODEL) -> dict[str, dict[str, list[str]]]:
    existing = read_json(ENTITY_CACHE_PATH, default={}) or {}
    cache: dict[str, dict[str, list[str]]] = dict(existing)
    for chunk in sorted(chunks, key=lambda c: str(c.get("chunk_id", ""))):
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in cache:
            continue
        cache[chunk_id] = extract_entities_openai(str(chunk.get("text", "")), model=model)
    write_json(ENTITY_CACHE_PATH, cache)
    return cache


def extract_question_entities(question: str, question_id: str, model: str = DEFAULT_MODEL) -> dict[str, list[str]]:
    cache = read_json(QUESTION_ENTITY_CACHE_PATH, default={}) or {}
    if question_id not in cache:
        cache[question_id] = extract_entities_openai(question, model=model)
        write_json(QUESTION_ENTITY_CACHE_PATH, cache)
    return cache[question_id]


def main() -> None:
    from .config import CHUNKS_PATH

    chunks = read_jsonl(CHUNKS_PATH)
    cache = build_entity_cache(chunks)
    print(f"Wrote entities for {len(cache)} chunks to {ENTITY_CACHE_PATH}")


if __name__ == "__main__":
    main()
