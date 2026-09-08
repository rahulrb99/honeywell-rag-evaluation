from __future__ import annotations

import argparse

from .build_graph_a import build_cooccurrence_graph, save_graph
from .build_graph_b import main as build_graph_b_main
from .chunking import chunk_page_records
from .config import CHUNKS_PATH, ENTITY_CACHE_PATH, GRAPH_A_PATH, PAGE_RECORDS_PATH, ensure_directories, seed_everything
from .entities import build_entity_cache
from .evaluate import run_evaluation
from .ingest import load_pdf_folder
from .io_utils import read_json, write_jsonl
from .prepare_benchmark import write_benchmark_from_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone newevals Graph_RAG benchmark pipeline.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse existing page records and chunks.")
    parser.add_argument("--skip-benchmark-prepare", action="store_true", help="Reuse existing benchmark_v1.json.")
    parser.add_argument("--skip-graph-b", action="store_true", help="Do not build the local semantic relation Graph B.")
    parser.add_argument("--skip-eval", action="store_true", help="Stop after graph construction.")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Use only the first N deterministic chunks for a smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    ensure_directories()

    if not args.skip_benchmark_prepare:
        rows = write_benchmark_from_eval()
        print(f"Prepared benchmark_v1.json from existing eval CSV with {len(rows)} rows.")

    if not args.skip_ingest:
        pages = load_pdf_folder()
        write_jsonl(PAGE_RECORDS_PATH, pages)
        chunks = chunk_page_records(pages)
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
        write_jsonl(CHUNKS_PATH, chunks)
        print(f"Wrote {len(pages)} page records and {len(chunks)} chunks.")
    else:
        from .io_utils import read_jsonl

        chunks = read_jsonl(CHUNKS_PATH)
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
            write_jsonl(CHUNKS_PATH, chunks)
        print(f"Reusing {len(chunks)} chunks from {CHUNKS_PATH}.")

    cache = build_entity_cache(chunks)
    graph_a = build_cooccurrence_graph(chunks, cache)
    save_graph(graph_a, GRAPH_A_PATH)
    print(f"Wrote Graph A to {GRAPH_A_PATH}.")

    if not args.skip_graph_b:
        build_graph_b_main()
    else:
        print("Skipped Graph B build. Evaluation will use an existing Graph B export if present.")

    if not args.skip_eval:
        result = run_evaluation()
        print(f"Wrote dashboard to {result['dashboard']}.")


if __name__ == "__main__":
    main()
