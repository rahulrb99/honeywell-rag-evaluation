from __future__ import annotations

import time
from collections import deque
from typing import Any
import re

import networkx as nx

from .config import DEFAULT_MODEL, K_HOPS
from .entities import extract_question_entities, flatten_entities
from .normalize import normalize_entity, normalize_entities


def _edge_chunk_ids(data: dict[str, Any]) -> list[str]:
    raw = str(data.get("supporting_chunk_ids", "") or "")
    return [part for part in raw.split("|") if part]


def _bfs_edges(graph: nx.Graph, starts: list[str], k: int) -> tuple[set[str], list[tuple[str, str, dict[str, Any]]]]:
    visited = set(starts)
    queue = deque((start, 0) for start in starts if start in graph)
    edges: list[tuple[str, str, dict[str, Any]]] = []
    seen_edges: set[tuple[str, str]] = set()
    while queue:
        node, depth = queue.popleft()
        if depth >= k:
            continue
        for neighbor in sorted(graph.neighbors(node)):
            key = tuple(sorted((node, neighbor))) if not graph.is_directed() else (node, neighbor)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append((node, neighbor, dict(graph.get_edge_data(node, neighbor, default={}))))
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return visited, edges


def _shortest_hops(graph: nx.Graph, starts: list[str], answer_entity: str | None) -> tuple[bool, int | None, list[str]]:
    answers = _match_graph_nodes(graph, _split_entity_targets(answer_entity))
    if not answers:
        return False, None, []
    best_path: list[str] = []
    best_hops: int | None = None
    undirected_view = graph.to_undirected() if graph.is_directed() else graph
    for start in starts:
        if start not in undirected_view:
            continue
        for answer in answers:
            if answer not in undirected_view:
                continue
            try:
                path = nx.shortest_path(undirected_view, start, answer)
            except nx.NetworkXNoPath:
                continue
            hops = len(path) - 1
            if best_hops is None or hops < best_hops:
                best_hops = hops
                best_path = path
    return best_hops is not None, best_hops, best_path


def _split_entity_targets(value: str | None) -> list[str]:
    raw = str(value or "")
    parts = re.split(r"\||,|\band\b|/", raw, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        normalized = normalize_entity(part)
        if normalized:
            out.append(normalized)
    whole = normalize_entity(raw)
    if whole:
        out.append(whole)
    return list(dict.fromkeys(out))


def _match_graph_nodes(graph: nx.Graph, values: list[str]) -> list[str]:
    nodes = [str(node) for node in graph.nodes()]
    matches: list[str] = []
    for value in values:
        if value in graph:
            matches.append(value)
            continue
        if len(value) < 3:
            continue
        for node in nodes:
            if node == value or value in node or node in value:
                matches.append(node)
    return sorted(set(matches))


def _path_precision(retrieved_edges: list[tuple[str, str, dict[str, Any]]], gold_path: list[list[str]] | None) -> float | None:
    if not gold_path:
        return None
    gold = set()
    for item in gold_path:
        if len(item) < 3:
            continue
        source = normalize_entity(item[0])
        target = normalize_entity(item[2])
        if source and target:
            gold.add(tuple(sorted((source, target))))
    if not retrieved_edges:
        return 0.0
    hits = 0
    for source, target, _ in retrieved_edges:
        if tuple(sorted((normalize_entity(source), normalize_entity(target)))) in gold:
            hits += 1
    return round(hits / len(retrieved_edges), 4)


def retrieve_graph_a(
    graph: nx.Graph,
    question: str,
    question_id: str,
    answer_entity: str | None = None,
    gold_path: list[list[str]] | None = None,
    k: int = K_HOPS,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    started = time.perf_counter()
    question_payload = extract_question_entities(question, question_id, model=model)
    starts = _match_graph_nodes(graph, flatten_entities(question_payload))
    nodes, edges = _bfs_edges(graph, starts, k)
    reachable, hops, shortest_path = _shortest_hops(graph, starts, answer_entity)
    supporting_chunk_ids = sorted({chunk_id for _, _, data in edges for chunk_id in _edge_chunk_ids(data)})
    return {
        "question_entities": starts,
        "retrieved_nodes": sorted(nodes),
        "retrieved_edges": edges,
        "supporting_chunk_ids": supporting_chunk_ids,
        "reachable": bool(reachable and hops is not None and hops <= k),
        "hops": hops,
        "shortest_path": shortest_path,
        "path_precision": _path_precision(edges, gold_path),
        "latency": time.perf_counter() - started,
    }


def retrieve_graph_b(
    graph: nx.Graph,
    question: str,
    question_id: str,
    answer_entity: str | None = None,
    gold_path: list[list[str]] | None = None,
    k: int = K_HOPS,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    started = time.perf_counter()
    question_payload = extract_question_entities(question, question_id, model=model)
    starts = _match_graph_nodes(graph, normalize_entities(flatten_entities(question_payload)))
    nodes, edges = _bfs_edges(graph.to_undirected() if graph.is_directed() else graph, starts, k)
    reachable, hops, shortest_path = _shortest_hops(graph, starts, answer_entity)
    supporting_chunk_ids = sorted({chunk_id for _, _, data in edges for chunk_id in _edge_chunk_ids(data)})
    return {
        "question_entities": starts,
        "retrieved_nodes": sorted(nodes),
        "retrieved_edges": edges,
        "supporting_chunk_ids": supporting_chunk_ids,
        "reachable": bool(reachable and hops is not None and hops <= k),
        "hops": hops,
        "shortest_path": shortest_path,
        "path_precision": _path_precision(edges, gold_path),
        "latency": time.perf_counter() - started,
    }
