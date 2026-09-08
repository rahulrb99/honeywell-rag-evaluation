from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for this subproject
    np = None


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

load_dotenv(REPO_ROOT / ".env", override=True)

SEED = 42
K_HOPS = 3
BENCHMARK_VERSION = "v1"
DEFAULT_MODEL = os.getenv("NEWEVALS_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
USE_OPENAI_ENTITY_EXTRACTION = os.getenv("NEWEVALS_USE_OPENAI_ENTITY_EXTRACTION", "").lower() in {"1", "true", "yes", "on"}
USE_OPENAI_QA = os.getenv("NEWEVALS_USE_OPENAI_QA", "").lower() in {"1", "true", "yes", "on"}
CHUNK_TOKENS = int(os.getenv("NEWEVALS_CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("NEWEVALS_CHUNK_OVERLAP", "100"))

RAW_PDF_DIR = ROOT / "data" / "raw_pdfs"
LEGACY_RAW_PDF_DIR = REPO_ROOT / "data" / "raw"
SOURCE_EVAL_CSV_PATH = REPO_ROOT / "data" / "eval" / "honeywell_hard_labels_28.csv"
PAGE_RECORDS_PATH = ROOT / "data" / "chunks" / "pages.jsonl"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
ENTITY_CACHE_PATH = ROOT / "data" / "chunks" / "entities.json"
QUESTION_ENTITY_CACHE_PATH = ROOT / "data" / "chunks" / "question_entities.json"
BENCHMARK_PATH = ROOT / "qa" / "benchmark_v1.json"
ONTOLOGY_TTL_PATH = ROOT / "graphs" / "graph_b" / "ontology.ttl"

GRAPH_A_PATH = ROOT / "graphs" / "graph_a" / "graph.graphml"
GRAPH_B_EXPORT_PATH = ROOT / "graphs" / "graph_b" / "graph_export.graphml"
GRAPH_B_RELATION_CACHE_PATH = ROOT / "data" / "chunks" / "graph_b_relations.json"
QUESTION_RESULTS_PATH = ROOT / "results" / "question_results.csv"
AGGREGATE_RESULTS_PATH = ROOT / "results" / "aggregate_results.csv"
AGGREGATE_JSON_PATH = ROOT / "results" / "aggregate_results.json"
GRAPH_METADATA_PATH = ROOT / "results" / "graph_metadata.json"
RUN_MANIFEST_PATH = ROOT / "results" / "run_manifest.json"
DASHBOARD_PATH = ROOT / "results" / "dashboard.html"
GRAPH_B_RELATION_MODE = os.getenv("NEWEVALS_GRAPH_B_RELATION_MODE", "hybrid").strip().lower()
GRAPH_B_MAX_LLM_CHUNKS = int(os.getenv("NEWEVALS_GRAPH_B_MAX_LLM_CHUNKS", "48"))


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str | None = os.getenv("NEO4J_URI") or None
    username: str | None = os.getenv("NEO4J_USERNAME") or None
    password: str | None = os.getenv("NEO4J_PASSWORD") or None


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)


def ensure_directories() -> None:
    for path in (
        RAW_PDF_DIR,
        PAGE_RECORDS_PATH.parent,
        ENTITY_CACHE_PATH.parent,
        GRAPH_A_PATH.parent,
        GRAPH_B_EXPORT_PATH.parent,
        BENCHMARK_PATH.parent,
        QUESTION_RESULTS_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
