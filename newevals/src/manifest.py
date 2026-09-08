from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import BENCHMARK_PATH, DEFAULT_MODEL, GRAPH_B_MAX_LLM_CHUNKS, GRAPH_B_RELATION_MODE, K_HOPS, RUN_MANIFEST_PATH, SEED
from .io_utils import write_json


def build_run_manifest(
    *,
    num_docs: int,
    num_chunks: int,
    model: str = DEFAULT_MODEL,
    benchmark_file: str = str(BENCHMARK_PATH),
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_id = f"{timestamp}_{BENCHMARK_PATH.stem}"
    return {
        "run_id": run_id,
        "benchmark_file": benchmark_file,
        "seed": SEED,
        "model": model,
        "num_docs": num_docs,
        "num_chunks": num_chunks,
        "graph_a_type": "co_occurrence",
        "graph_b_type": "semantic_relation",
        "graph_b_builder": GRAPH_B_RELATION_MODE,
        "graph_b_max_llm_chunks": GRAPH_B_MAX_LLM_CHUNKS,
        "k_hops": K_HOPS,
    }


def write_run_manifest(manifest: dict[str, Any], path=RUN_MANIFEST_PATH) -> dict[str, Any]:
    write_json(path, manifest)
    return manifest
