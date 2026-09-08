from __future__ import annotations

from typing import Any

import networkx as nx

from .config import GRAPH_METADATA_PATH
from .io_utils import write_json


def compute_graph_stats(graph: nx.Graph) -> dict[str, Any]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    avg_degree = round((sum(dict(graph.degree()).values()) / nodes) if nodes else 0.0, 4)
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    return {
        "nodes": nodes,
        "edges": edges,
        "avg_degree": avg_degree,
        "connected_components": len(components),
        "largest_component_size": max((len(component) for component in components), default=0),
    }


def write_graph_metadata(graph_a: nx.Graph, graph_b: nx.Graph, path=GRAPH_METADATA_PATH) -> dict[str, Any]:
    metadata = {
        "graph_a": compute_graph_stats(graph_a),
        "graph_b": compute_graph_stats(graph_b),
    }
    write_json(path, metadata)
    return metadata

