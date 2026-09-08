from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx

from .build_graph_a import build_cooccurrence_graph, save_graph
from .config import (
    AGGREGATE_JSON_PATH,
    AGGREGATE_RESULTS_PATH,
    BENCHMARK_PATH,
    CHUNKS_PATH,
    DEFAULT_MODEL,
    ENTITY_CACHE_PATH,
    GRAPH_A_PATH,
    GRAPH_B_EXPORT_PATH,
    GRAPH_METADATA_PATH,
    K_HOPS,
    QUESTION_RESULTS_PATH,
    ensure_directories,
    seed_everything,
)
from .dashboard import generate_dashboard
from .entities import build_entity_cache
from .graph_stats import write_graph_metadata
from .io_utils import read_json, read_jsonl, write_json
from .manifest import build_run_manifest, write_run_manifest
from .qa_pipeline import answer_graph_a, answer_graph_b, is_correct
from .retrieval import retrieve_graph_a, retrieve_graph_b


def _load_graph(path: Path, directed: bool = False) -> nx.Graph:
    if path.exists():
        graph = nx.read_graphml(path)
        return nx.DiGraph(graph) if directed else nx.Graph(graph)
    return nx.DiGraph() if directed else nx.Graph()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _numeric_values(values: list[Any]) -> list[float]:
        numbers: list[float] = []
        for value in values:
            if value in (None, ""):
                continue
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                continue
        return numbers

    out = []
    for graph_name in ("graph_a", "graph_b"):
        subset = [row for row in rows if row["graph"] == graph_name]
        if not subset:
            continue
        reachable = [1 if row["reachable"] else 0 for row in subset]
        hops = _numeric_values([row["hops"] for row in subset])
        precisions = _numeric_values([row["path_precision"] for row in subset])
        correct = [1 if row["correct"] else 0 for row in subset]
        latencies = _numeric_values([row["latency"] for row in subset])
        out.append(
            {
                "graph": graph_name,
                "reachability_at_3": round(sum(reachable) / len(reachable), 4) if reachable else 0.0,
                "avg_hops": round(sum(hops) / len(hops), 4) if hops else "",
                "path_precision": round(sum(precisions) / len(precisions), 4) if precisions else "",
                "qa_accuracy": round(sum(correct) / len(correct), 4) if correct else 0.0,
                "avg_latency": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            }
        )
    return out


def run_evaluation(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return run_evaluation_for_paths(model=model)


def _prefixed(path: Path, output_prefix: str) -> Path:
    if not output_prefix:
        return path
    return path.with_name(f"{path.stem}_{output_prefix}{path.suffix}")


def run_evaluation_for_paths(
    *,
    model: str = DEFAULT_MODEL,
    benchmark_path: Path = BENCHMARK_PATH,
    output_prefix: str = "",
) -> dict[str, Any]:
    seed_everything()
    ensure_directories()
    question_results_path = _prefixed(QUESTION_RESULTS_PATH, output_prefix)
    aggregate_results_path = _prefixed(AGGREGATE_RESULTS_PATH, output_prefix)
    aggregate_json_path = _prefixed(AGGREGATE_JSON_PATH, output_prefix)
    graph_metadata_path = _prefixed(GRAPH_METADATA_PATH, output_prefix)
    run_manifest_path = _prefixed(Path("newevals/results/run_manifest.json"), output_prefix)
    dashboard_path = _prefixed(Path("newevals/results/dashboard.html"), output_prefix)

    chunks = read_jsonl(CHUNKS_PATH)
    benchmark = sorted(read_json(benchmark_path, default=[]) or [], key=lambda q: str(q.get("id", "")))
    if not chunks:
        raise ValueError(f"No chunks found at {CHUNKS_PATH}. Run ingest.py and chunking.py first.")
    if not benchmark:
        raise ValueError(f"No benchmark questions found at {benchmark_path}.")

    entity_cache = build_entity_cache(chunks, model=model)
    graph_a = build_cooccurrence_graph(chunks, entity_cache)
    save_graph(graph_a, GRAPH_A_PATH)
    graph_b = _load_graph(GRAPH_B_EXPORT_PATH, directed=True)
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}

    rows: list[dict[str, Any]] = []
    for item in benchmark:
        qid = str(item["id"])
        question = str(item["question"])
        gold_answer = str(item.get("answer", ""))
        answer_entity = str(item.get("answer_entity", gold_answer))
        gold_path = item.get("gold_reasoning_path") or None

        retrieval_a = retrieve_graph_a(graph_a, question, qid, answer_entity, gold_path, k=K_HOPS, model=model)
        answer_a = answer_graph_a(question, retrieval_a, chunks_by_id, model=model)
        rows.append(
            {
                "question_id": qid,
                "question": question,
                "graph": "graph_a",
                "reachable": retrieval_a["reachable"],
                "hops": retrieval_a["hops"] if retrieval_a["hops"] is not None else "",
                "path_precision": retrieval_a["path_precision"] if retrieval_a["path_precision"] is not None else "",
                "correct": is_correct(str(answer_a.get("answer", "")), gold_answer),
                "latency": round(float(retrieval_a["latency"]), 4),
                "supporting_chunk_ids": json.dumps(retrieval_a["supporting_chunk_ids"]),
                "answer": str(answer_a.get("answer", "")),
            }
        )

        retrieval_b = retrieve_graph_b(graph_b, question, qid, answer_entity, gold_path, k=K_HOPS, model=model)
        answer_b = answer_graph_b(question, retrieval_b, chunks_by_id, model=model)
        rows.append(
            {
                "question_id": qid,
                "question": question,
                "graph": "graph_b",
                "reachable": retrieval_b["reachable"],
                "hops": retrieval_b["hops"] if retrieval_b["hops"] is not None else "",
                "path_precision": retrieval_b["path_precision"] if retrieval_b["path_precision"] is not None else "",
                "correct": is_correct(str(answer_b.get("answer", "")), gold_answer),
                "latency": round(float(retrieval_b["latency"]), 4),
                "supporting_chunk_ids": json.dumps(retrieval_b["supporting_chunk_ids"]),
                "answer": str(answer_b.get("answer", "")),
            }
        )

    aggregate = _aggregate(rows)
    _write_csv(
        question_results_path,
        rows,
        ["question_id", "question", "graph", "reachable", "hops", "path_precision", "correct", "latency", "supporting_chunk_ids", "answer"],
    )
    _write_csv(
        aggregate_results_path,
        aggregate,
        ["graph", "reachability_at_3", "avg_hops", "path_precision", "qa_accuracy", "avg_latency"],
    )
    write_json(aggregate_json_path, aggregate)
    metadata = write_graph_metadata(graph_a, graph_b, graph_metadata_path)
    manifest = write_run_manifest(
        build_run_manifest(
            num_docs=len({str(chunk.get("doc_id", "")) for chunk in chunks}),
            num_chunks=len(chunks),
            model=model,
            benchmark_file=str(benchmark_path),
        ),
        path=run_manifest_path,
    )
    dashboard = generate_dashboard(
        aggregate_path=aggregate_results_path,
        question_path=question_results_path,
        metadata_path=graph_metadata_path,
        manifest_path=run_manifest_path,
        out_path=dashboard_path,
    )
    return {"rows": rows, "aggregate": aggregate, "metadata": metadata, "manifest": manifest, "dashboard": str(dashboard)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Graph A vs Graph B for newevals.")
    parser.add_argument("--benchmark", default=str(BENCHMARK_PATH))
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation_for_paths(
        model=args.model,
        benchmark_path=Path(args.benchmark),
        output_prefix=args.output_prefix,
    )
    print(f"Wrote {len(result['rows'])} question rows")
    print(f"Wrote dashboard to {result['dashboard']}")


if __name__ == "__main__":
    main()
