from __future__ import annotations

import argparse
import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
from openai import OpenAI

from .config import (
    CHUNKS_PATH,
    DEFAULT_MODEL,
    ENTITY_CACHE_PATH,
    GRAPH_B_EXPORT_PATH,
    GRAPH_B_MAX_LLM_CHUNKS,
    GRAPH_B_RELATION_CACHE_PATH,
    GRAPH_B_RELATION_MODE,
)
from .entities import ENTITY_KEYS, build_entity_cache, flatten_entities
from .io_utils import read_json, read_jsonl, write_json


SUPPORTS_RE = re.compile(
    r"\b(supports?|supporting|protocol|communicates?|communication|network|networked|ethernet|bacnet|modbus|slc)\b",
    flags=re.I,
)
VOLTAGE_RE = re.compile(
    r"\b(requires?|powered|power|supply|voltage|operat(?:es|ing)|input|output|vdc|vac|volts?)\b",
    flags=re.I,
)
COMPONENT_RE = re.compile(
    r"\b(includes?|contains?|has|with|features?|equipped|module|sensor|controller|panel|amplifier|speaker|strobe|microphone|circuit|loop)\b",
    flags=re.I,
)
COMPATIBLE_RE = re.compile(
    r"\b(compatible with|works with|used with|connects? to|interfaces? with|integrates? with|supports?)\b",
    flags=re.I,
)
INSTALL_RE = re.compile(r"\b(install(?:ed|ation)?|mount(?:ed|ing)?|wire(?:d|s|ing)?|connect(?:ed|s|ion)?)\b", flags=re.I)
ERROR_RE = re.compile(r"\b(error|fault|trouble|indicates?|means|code|condition|alarm)\b", flags=re.I)
ALLOWED_RELATIONS = {
    "supports_protocol",
    "requires_voltage",
    "has_component",
    "compatible_with",
    "installed_with",
    "communicates_with",
    "indicates_fault",
    "configured_by",
    "monitors",
    "controls",
}
RELATION_ALIASES = {
    "supports": "supports_protocol",
    "uses_protocol": "supports_protocol",
    "requires": "requires_voltage",
    "requires_power": "requires_voltage",
    "operates_at_voltage": "requires_voltage",
    "contains": "has_component",
    "includes": "has_component",
    "has": "has_component",
    "compatible": "compatible_with",
    "connects_to": "installed_with",
    "wired_to": "installed_with",
    "indicates": "indicates_fault",
}
LLM_RELATION_PROMPT = """Extract typed technical relations from this Honeywell document chunk.

Use only entities from the provided entity list when possible. Keep subjects and objects short.
Allowed relations:
- supports_protocol
- requires_voltage
- has_component
- compatible_with
- installed_with
- communicates_with
- indicates_fault
- configured_by
- monitors
- controls

Return JSON only:
{{"relations":[{{"subject":"...", "relation":"...", "object":"...", "evidence":"short quote"}}]}}

Entity list:
{entities}

Chunk text:
{text}
"""


def _append_unique(raw: str, value: str) -> str:
    values = [part for part in str(raw or "").split("|") if part]
    if value not in values:
        values.append(value)
    return "|".join(values)


def _clean_sentence(text: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _normalize_sentence(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    normalized = re.sub(r"\bbacnet\s*(?:/ip|ip|mstp|ms/tp|protocol)?\b", " bacnet ", normalized)
    normalized = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:vdc|vac|v|volt|volts)(?:\s*(?:dc|ac))?\b", r" \1v ", normalized)
    normalized = re.sub(r"[^a-z0-9./#+-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_entity(sentence: str, entity: str) -> bool:
    escaped = re.escape(entity.lower())
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", sentence))


def _entities_in_sentence(entity_payload: dict[str, list[str]], sentence: str) -> dict[str, list[str]]:
    normalized_sentence = _normalize_sentence(sentence)
    present: dict[str, list[str]] = {key: [] for key in ENTITY_KEYS}
    for key in ENTITY_KEYS:
        for entity in entity_payload.get(key, []):
            if entity and _contains_entity(normalized_sentence, entity):
                present[key].append(entity)
    return present


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _node_kind(entity: str, present: dict[str, list[str]]) -> str:
    kinds = [key[:-1] if key.endswith("s") else key for key, values in present.items() if entity in values]
    return "|".join(_unique(kinds)) or "entity"


def _add_node(graph: nx.DiGraph, entity: str, kind: str) -> None:
    if graph.has_node(entity):
        graph.nodes[entity]["node_type"] = _append_unique(str(graph.nodes[entity].get("node_type", "")), kind)
    else:
        graph.add_node(entity, name=entity, node_type=kind)


def _add_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    relation: str,
    chunk: dict[str, Any],
    sentence: str,
    present: dict[str, list[str]],
    confidence: float,
) -> bool:
    if not source or not target or source == target:
        return False
    _add_node(graph, source, _node_kind(source, present))
    _add_node(graph, target, _node_kind(target, present))
    chunk_id = str(chunk.get("chunk_id", ""))
    doc_id = str(chunk.get("doc_id", ""))
    page = str(chunk.get("page_start", chunk.get("page", "")))
    if graph.has_edge(source, target):
        data = graph[source][target]
        data["relation"] = _append_unique(str(data.get("relation", "")), relation)
        data["weight"] = int(data.get("weight", 1)) + 1
        data["supporting_chunk_ids"] = _append_unique(str(data.get("supporting_chunk_ids", "")), chunk_id)
        data["doc_ids"] = _append_unique(str(data.get("doc_ids", "")), doc_id)
        data["pages"] = _append_unique(str(data.get("pages", "")), page)
        data["confidence"] = max(float(data.get("confidence", 0.0) or 0.0), confidence)
        if not data.get("evidence"):
            data["evidence"] = _clean_sentence(sentence)
    else:
        graph.add_edge(
            source,
            target,
            relation=relation,
            weight=1,
            supporting_chunk_ids=chunk_id,
            doc_ids=doc_id,
            pages=page,
            confidence=confidence,
            evidence=_clean_sentence(sentence),
        )
    return True


def _entity_sources(present: dict[str, list[str]]) -> list[str]:
    return _unique(present.get("products", []) + present.get("components", []))


def _add_pair_edges(
    graph: nx.DiGraph,
    entities: list[str],
    relation: str,
    chunk: dict[str, Any],
    sentence: str,
    present: dict[str, list[str]],
    confidence: float,
) -> int:
    count = 0
    for source, target in combinations(_unique(entities), 2):
        if _add_edge(graph, source, target, relation, chunk, sentence, present, confidence):
            count += 1
    return count


def _add_sentence_relations(
    graph: nx.DiGraph,
    chunk: dict[str, Any],
    sentence: str,
    present: dict[str, list[str]],
) -> int:
    count = 0
    sources = _entity_sources(present)

    if sources and present.get("protocols"):
        confidence = 0.92 if SUPPORTS_RE.search(sentence) else 0.78
        for source in sources:
            for protocol in present["protocols"]:
                count += int(_add_edge(graph, source, protocol, "supports_protocol", chunk, sentence, present, confidence))

    if sources and present.get("voltages"):
        confidence = 0.94 if VOLTAGE_RE.search(sentence) else 0.8
        for source in sources:
            for voltage in present["voltages"]:
                count += int(_add_edge(graph, source, voltage, "requires_voltage", chunk, sentence, present, confidence))

    products = present.get("products", [])
    components = present.get("components", [])
    if products and components:
        confidence = 0.9 if COMPONENT_RE.search(sentence) else 0.72
        for product in products:
            for component in components:
                count += int(_add_edge(graph, product, component, "has_component", chunk, sentence, present, confidence))

    if products and components and COMPATIBLE_RE.search(sentence):
        for product in products:
            for component in components[:8]:
                count += int(_add_edge(graph, product, component, "compatible_with", chunk, sentence, present, 0.86))

    if products and components and INSTALL_RE.search(sentence):
        for product in products:
            for component in components[:8]:
                count += int(_add_edge(graph, product, component, "installed_with", chunk, sentence, present, 0.84))

    if present.get("error_codes") and ERROR_RE.search(sentence):
        targets = sources or present.get("products", []) or present.get("components", [])
        for error_code in present["error_codes"]:
            for target in targets:
                count += int(_add_edge(graph, error_code, target, "indicates_fault", chunk, sentence, present, 0.9))

    return count


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        return {"relations": []}


def _relation_name(value: str) -> str:
    relation = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    relation = RELATION_ALIASES.get(relation, relation)
    return relation if relation in ALLOWED_RELATIONS else ""


def _chunk_signal_score(chunk: dict[str, Any], entity_payload: dict[str, list[str]]) -> tuple[int, str]:
    score = (
        3 * len(entity_payload.get("products", []))
        + 3 * len(entity_payload.get("protocols", []))
        + 3 * len(entity_payload.get("voltages", []))
        + 2 * len(entity_payload.get("error_codes", []))
        + len(entity_payload.get("components", []))
    )
    return (-score, str(chunk.get("chunk_id", "")))


def _selected_llm_chunks(
    chunks: list[dict[str, Any]],
    entity_cache: dict[str, dict[str, list[str]]],
    max_chunks: int,
) -> list[dict[str, Any]]:
    candidates = [
        chunk
        for chunk in chunks
        if len(flatten_entities(entity_cache.get(str(chunk.get("chunk_id", "")), {}))) >= 2
    ]
    candidates = sorted(candidates, key=lambda chunk: _chunk_signal_score(chunk, entity_cache.get(str(chunk.get("chunk_id", "")), {})))
    if max_chunks > 0:
        return candidates[:max_chunks]
    return candidates


def extract_relations_openai(
    chunk: dict[str, Any],
    entity_payload: dict[str, list[str]],
    *,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> list[dict[str, str]]:
    entities = flatten_entities(entity_payload)
    if len(entities) < 2:
        return []
    client = client or OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": LLM_RELATION_PROMPT.format(
                    entities=json.dumps(entities[:80], ensure_ascii=True),
                    text=str(chunk.get("text", ""))[:5000],
                ),
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    payload = _extract_json_object(response.choices[0].message.content or "{}")
    relations: list[dict[str, str]] = []
    for item in payload.get("relations", []):
        relation = _relation_name(str(item.get("relation", "")))
        subject = str(item.get("subject", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subject and obj and relation:
            relations.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "evidence": _clean_sentence(str(item.get("evidence", ""))),
                }
            )
    return relations[:16]


def _add_llm_relations(
    graph: nx.DiGraph,
    chunks: list[dict[str, Any]],
    entity_cache: dict[str, dict[str, list[str]]],
    *,
    model: str,
    max_chunks: int,
    relation_cache_path: Path,
) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping LLM relation extraction: OPENAI_API_KEY is not set.")
        return 0

    relation_cache = read_json(relation_cache_path, default={}) or {}
    selected = _selected_llm_chunks(chunks, entity_cache, max_chunks)
    client = OpenAI()
    added = 0
    for index, chunk in enumerate(selected, start=1):
        chunk_id = str(chunk.get("chunk_id", ""))
        entity_payload = entity_cache.get(chunk_id, {})
        if chunk_id not in relation_cache:
            print(f"Extracting Graph B LLM relations {index}/{len(selected)}: {chunk_id}")
            try:
                relation_cache[chunk_id] = extract_relations_openai(chunk, entity_payload, model=model, client=client)
            except Exception as exc:
                print(f"Skipping LLM relations for {chunk_id}: {exc}")
                relation_cache[chunk_id] = []
            write_json(relation_cache_path, relation_cache)
        present = {key: list(entity_payload.get(key, [])) for key in ENTITY_KEYS}
        for relation in relation_cache.get(chunk_id, []):
            subject = _best_entity_match(str(relation.get("subject", "")), entity_payload)
            obj = _best_entity_match(str(relation.get("object", "")), entity_payload)
            relation_name = _relation_name(str(relation.get("relation", "")))
            if subject and obj and relation_name:
                added += int(
                    _add_edge(
                        graph,
                        subject,
                        obj,
                        relation_name,
                        chunk,
                        str(relation.get("evidence", "")) or str(chunk.get("text", ""))[:240],
                        present,
                        0.97,
                    )
                )
    return added


def _best_entity_match(value: str, entity_payload: dict[str, list[str]]) -> str:
    normalized = _normalize_sentence(value)
    entities = flatten_entities(entity_payload)
    for entity in entities:
        if entity == normalized or _contains_entity(normalized, entity):
            return entity
    for entity in entities:
        if len(entity) >= 3 and (entity in normalized or normalized in entity):
            return entity
    return normalized


def build_semantic_relation_graph(
    chunks: list[dict[str, Any]],
    entity_cache: dict[str, dict[str, list[str]]],
    *,
    relation_mode: str = GRAPH_B_RELATION_MODE,
    model: str = DEFAULT_MODEL,
    max_llm_chunks: int = GRAPH_B_MAX_LLM_CHUNKS,
    relation_cache_path: Path = GRAPH_B_RELATION_CACHE_PATH,
) -> nx.DiGraph:
    relation_mode = relation_mode.lower().strip()
    graph = nx.DiGraph(graph_type="semantic_relation", graph_b_builder=relation_mode)
    for chunk in sorted(chunks, key=lambda c: str(c.get("chunk_id", ""))):
        chunk_id = str(chunk.get("chunk_id", ""))
        entity_payload = entity_cache.get(chunk_id, {})
        for sentence in _split_sentences(str(chunk.get("text", ""))):
            present = _entities_in_sentence(entity_payload, sentence)
            if relation_mode in {"rules", "hybrid"}:
                _add_sentence_relations(graph, chunk, sentence, present)
    if relation_mode in {"llm", "hybrid"}:
        _add_llm_relations(
            graph,
            chunks,
            entity_cache,
            model=model,
            max_chunks=max_llm_chunks,
            relation_cache_path=relation_cache_path,
        )
    return graph


def save_graph(graph: nx.DiGraph, path: Path = GRAPH_B_EXPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic newevals Graph B semantic relation graph.")
    parser.add_argument("--chunks", default=str(CHUNKS_PATH))
    parser.add_argument("--entities", default=str(ENTITY_CACHE_PATH))
    parser.add_argument("--output", default=str(GRAPH_B_EXPORT_PATH))
    parser.add_argument("--relation-mode", choices=["rules", "llm", "hybrid"], default=GRAPH_B_RELATION_MODE)
    parser.add_argument("--max-llm-chunks", type=int, default=GRAPH_B_MAX_LLM_CHUNKS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(Path(args.chunks))
    cache = read_json(Path(args.entities), default={}) or {}
    if not cache:
        cache = build_entity_cache(chunks)
    graph = build_semantic_relation_graph(
        chunks,
        cache,
        relation_mode=args.relation_mode,
        model=args.model,
        max_llm_chunks=args.max_llm_chunks,
    )
    save_graph(graph, Path(args.output))
    print(
        f"Wrote Graph B semantic relation graph with "
        f"{graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {Path(args.output)}"
    )


if __name__ == "__main__":
    main()
