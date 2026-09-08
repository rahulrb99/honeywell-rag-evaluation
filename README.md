# GraphEval-Ragas

[![CI](https://github.com/rahulrb99/honeywell-rag-evaluation/actions/workflows/tests.yml/badge.svg)](https://github.com/rahulrb99/honeywell-rag-evaluation/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphRAG-green)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**End-to-end evaluation framework comparing GraphRAG against a vector RAG baseline on Honeywell technical documentation.** Produces a reproducible static dashboard with system win rates, query-class breakdowns, failure analysis, RAGAS scores, and bootstrap confidence intervals.

---

## Key Results

Evaluated on 28 expert-labelled Honeywell technical questions across 5 query classes:

| Metric | Value |
|--------|-------|
| GraphRAG win rate | **46.4%** |
| Vector RAG win rate | 32.1% |
| Ties | 17.9% |
| Hard-query GraphRAG win rate | 37.5% |

GraphRAG showed the clearest advantage on **multi-hop** and **relationship reasoning** questions, where context from linked graph entities is required. Vector RAG was competitive on single-hop factual lookups.

---

## What This Project Demonstrates

| Skill | How |
|-------|-----|
| RAG pipeline design | FAISS vector store ingestion, chunking, retrieval, answer generation |
| Knowledge graph construction | Neo4j-backed GraphRAG domain with ontology generation and entity resolution |
| LLM evaluation | RAGAS metrics, pairwise LLM-as-a-judge, inter-rater kappa |
| Reproducible benchmarking | SHA256-verified manifests, bootstrap CIs, deterministic proxy metrics |
| Data engineering | Two eval datasets (28 hard-labelled + 35 auto-generated), query-class stratification |
| Software engineering | Modular pipeline, pytest suite (38 tests), GitHub Actions CI |

---

## Tech Stack

**Core:** Python 3.11 · Neo4j · FAISS · NetworkX  
**LLMs:** OpenAI · Groq (LLaMA 3)  
**Eval:** RAGAS · LLM-as-a-judge · bootstrap resampling  
**Viz:** Streamlit · Plotly · static HTML dashboard  
**Infra:** GitHub Actions CI · pytest · pandas

---

## System Architecture

```
Honeywell PDFs
    │
    ├─► src/ingest.py ──────────────► FAISS vectorstore
    │                                        │
    └─► graphrag_engine/ ──────────► Neo4j graph domain
              │                              │
              ▼                              ▼
    week2_graph_rag/              week1_vector_rag/
    run_graphrag_predictions      rag_pipeline
              │                              │
              └──────────┬───────────────────┘
                         ▼
              src/week3_benchmark/
              ├── run_week3_eval.py       (metrics)
              ├── run_full_ragas_comparison.py
              ├── run_pairwise_llm_judge.py
              ├── bootstrap_confidence_intervals.py
              ├── graph_coverage_metrics.py
              └── build_week3_dashboard.py
                         │
                         ▼
              outputs/eval_outputs/
              ├── dashboard.html          ← open this
              └── benchmarking_comparative_analysis.md
```

---

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# Copy and fill in credentials
cp .env.example .env

# Build the FAISS vectorstore
python -m src.ingest

# Run the full pipeline
python -m src.run_week3_pipeline
```

Open `outputs/eval_outputs/dashboard.html` in a browser.

**Fast mode** (skip generation, use supplied outputs):

```powershell
python -m src.run_week3_pipeline `
  --dataset data\eval\honeywell_hard_labels_28.csv `
  --graph-output outputs\predictions\honeywell_hard_labels_28_graphrag.csv `
  --vector-output outputs\predictions\vector_rag_hard_labels_28_predictions.csv `
  --skip-graph-output-generation `
  --skip-vector-output-generation `
  --skip-live-ragas `
  --skip-llm-judge
```

See [quickstart.md](quickstart.md) for environment variables and the full CLI reference.

---

## Repository Structure

```
data/eval/              Benchmark CSVs (28 hard-labelled + 35 auto-generated)
data/raw/               Honeywell source PDFs
graphrag_engine/        Neo4j graph construction engine with entity resolution
newevals/               Graph A vs Graph B standalone evaluation sub-project
src/ingest.py           FAISS vectorstore builder
src/week1_vector_rag/   Vector RAG baseline pipeline
src/week2_graph_rag/    GraphRAG domain builder and prediction generator
src/week3_benchmark/    Benchmark metrics, RAGAS, judge, dashboard builder
src/run_week3_pipeline.py  Main orchestration entry point
tests/                  pytest suite (38 tests, CI-verified)
```

---

## Milestones

| Milestone | What was built |
|-----------|----------------|
| Week 1 | Vector RAG baseline — ingestion, FAISS retrieval, answer generation, initial evals |
| Week 2 | GraphRAG integration — Neo4j domain builder, ontology generation, prediction export |
| Week 3 | Comparative benchmark — RAGAS, LLM judge, bootstrap CIs, failure taxonomy, dashboard |
| newevals | Standalone Graph A (co-occurrence) vs Graph B (semantic) evaluation framework |

---

## Tests

```powershell
python -m pytest -p no:cacheprovider
```

38 tests covering metric helpers, failure-mode logic, CSV schema validation, bootstrap CIs, disagreement analysis, manifest generation, and dashboard rendering.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | GraphRAG domain generation and RAGAS |
| `GROQ_API_KEY` | Answer generation and LLM judge |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Graph database |

---

## Security

`.env`, API keys, vectorstore, and generated `outputs/` are gitignored and must never be committed.
