from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx

from .config import CHUNKS_PATH, ENTITY_CACHE_PATH, GRAPH_A_PATH
from .entities import flatten_entities
from .io_utils import read_json, read_jsonl


def _append_unique(values: list[str], value: str) -> list[str]:
    if value not in values:
        values.append(value)
    return values


def build_cooccurrence_graph(
    chunks: list[dict[str, Any]],
    entity_cache: dict[str, dict[str, list[str]]],
) -> nx.Graph:
    graph = nx.Graph(graph_type="co_occurrence")
    for chunk in sorted(chunks, key=lambda c: str(c.get("chunk_id", ""))):
        chunk_id = str(chunk["chunk_id"])
        entities = flatten_entities(entity_cache.get(chunk_id, {}))
        for entity in entities:
            graph.add_node(entity, name=entity)
        for source, target in combinations(entities, 2):
            if graph.has_edge(source, target):
                data = graph[source][target]
                data["weight"] = int(data.get("weight", 1)) + 1
                data["supporting_chunk_ids"] = "|".join(_append_unique(str(data.get("supporting_chunk_ids", "")).split("|") if data.get("supporting_chunk_ids") else [], chunk_id))
            else:
                graph.add_edge(
                    source,
                    target,
                    relation="co_occurs_in_chunk",
                    weight=1,
                    supporting_chunk_ids=chunk_id,
                    doc_ids=str(chunk.get("doc_id", "")),
                    pages=str(chunk.get("page_start", "")),
                )
    return graph


def save_graph(graph: nx.Graph, path=GRAPH_A_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def main() -> None:
    chunks = read_jsonl(CHUNKS_PATH)
    cache = read_json(ENTITY_CACHE_PATH, default={}) or {}
    graph = build_cooccurrence_graph(chunks, cache)
    save_graph(graph)
    print(f"Wrote Graph A with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {GRAPH_A_PATH}")


if __name__ == "__main__":
    main()

