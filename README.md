# GraphEval-Ragas

[![CI](https://github.com/rahulrb99/honeywell-rag-evaluation/actions/workflows/tests.yml/badge.svg)](https://github.com/rahulrb99/honeywell-rag-evaluation/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**End-to-end evaluation framework comparing a Neo4j GraphRAG system against a vector RAG baseline on Honeywell technical documentation.**

This project builds two retrieval systems from the same PDF corpus, runs them against a 28-question expert-labelled benchmark, and produces a reproducible static dashboard with system win rates, query-class breakdowns, failure taxonomy, RAGAS scores, bootstrap confidence intervals, and a pairwise LLM judge. A second evaluation tier (`newevals/`) independently compares two local graph construction strategies (co-occurrence vs. semantic) without requiring Neo4j.

---

## Key Results

Evaluated on **28 expert-labelled Honeywell technical questions** across 5 query classes:

| Outcome | Count | Share of 28 questions |
|---------|------:|----------------------:|
| GraphRAG wins | 13 | 46.4% |
| Vector RAG wins | 9 | 32.1% |
| Ties | 5 | 17.9% |
| Neither system answered well | 1 | 3.6% |

All rates are computed over all 28 rows. The 1 "neither" row (3.6%) is a case where both systems scored below the minimum-quality threshold (composite score < 0.35); it is not excluded from the denominator.

**Hard-query GraphRAG win rate (multi-hop, comparison, theme summary, relationship reasoning): 37.5%** (6/16 hard rows)

> **Nuance:** GraphRAG leads overall (46.4% vs 32.1%), but on the hard 16-row subset vector RAG actually wins more often (8 vs 6). GraphRAG's overall advantage is concentrated in `list_settings` (4/4) and `numeric_spec` (3/4) classes; it is weaker on `theme_summary` (0/4) and `relationship_reasoning` (1/4). The `retrieval_missed_entities` failure mode accounts for 13 of 28 rows, indicating the dominant limitation is retrieval quality rather than generation.

---

## Why GraphRAG Helps (and Where It Doesn't)

**GraphRAG advantage:**
- `list_settings`: 4/4 wins — structured graph traversal retrieves enumerated settings reliably
- `numeric_spec`: 3/4 wins — entity-linked lookups outperform chunk-boundary splits
- `comparison`: 3/4 wins — multi-entity context from graph neighbours

**Vector RAG competitive or superior:**
- `theme_summary`: 0/4 GraphRAG wins — vector RAG's broad passage retrieval handles abstractive synthesis better
- `relationship_reasoning`: 1/4 GraphRAG wins — vector context precision (0.96 mean) vs. GraphRAG (0.59 mean) shows vector chunking is better calibrated for these questions
- `single_hop_fact`: largely ties (3/4) — both systems retrieve single-hop facts equally well
- `multi_hop`: 2/4 each — no clear winner

---

## Evaluation Methodology

### Benchmark datasets
- **28 expert-labelled questions** (`data/eval/honeywell_hard_labels_28.csv`) — manually curated across 5 query classes (`single_hop_fact`, `numeric_spec`, `list_settings`, `comparison`, `multi_hop`, `theme_summary`, `relationship_reasoning`) with ground-truth answers, gold contexts, source product labels, and source field annotations.
- **35 auto-generated questions** (`data/eval/honeywell_autoq_35.csv`) — machine-generated from the same PDFs via the pipeline's `autoq` mode. These expand benchmark coverage but lack expert-verified ground truth; results are presented separately and not mixed with the 28-question headline numbers.

### Metric tiers

| Tier | Component | Role |
|------|-----------|------|
| 1 | **Deterministic proxy metrics** (`run_week3_eval.py`) | Reproducible, no-API scoring: entity recall, context precision, faithfulness, answer relevancy, composite score, winner assignment. Winner logic: "neither" if both scores < 0.35; "tie" if scores within 0.05; else the higher scorer wins. |
| 2 | **RAGAS** (`run_full_ragas_comparison.py`) | LLM-based faithfulness, answer relevancy, and context precision/recall on a 5-row smoke subset (default) or full set. Provides a second, independent signal. |
| 3 | **Pairwise LLM-as-a-judge** (`run_pairwise_llm_judge.py`) | A Groq-hosted LLM (Llama 3.3 70B) scores each row head-to-head using question, ground truth, both answers, and both retrieved contexts. Returns winner, confidence, and rationale. |
| 4 | **Bootstrap confidence intervals** (`bootstrap_confidence_intervals.py`) | 1,000-resample percentile CIs (seed 5010) on win rates by query class (classes with n ≥ 5 only). Quantifies how much the small-n class-level numbers should be trusted. |
| 5 | **Graph coverage proxies** (`graph_coverage_metrics.py`) | Structural coverage metrics over GraphRAG contexts: entity coverage, relation signal coverage, structural density, composite graph coverage score. |

### Additional analyses
- **Failure taxonomy**: Each losing row is classified as `retrieval_missed_entities`, `low_context_precision`, `unfaithful_answer`, `irrelevant_or_dodged_answer`, `incomplete_synthesis`, `both_correct_or_tie`, using deterministic rules on per-metric scores.
- **Inter-rater kappa** (`run_kappa_analysis.py`): Cohen's Kappa between human manual review, LLM judge verdicts, and deterministic metric winners — available when `--manual-review` and `--llm-judge` CSVs are supplied.
- **Run manifest with SHA256** (`run_manifest.py`): Each pipeline run writes `run_manifest.json` containing dataset paths, SHA256 hashes of input files, git SHA, CLI arguments, and timestamps. Ensures results are traceable to specific input snapshots.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SOURCE DATA                          │
│  data/raw/ (Honeywell PDFs)                             │
│  data/eval/honeywell_hard_labels_28.csv  (28 labelled)  │
│  data/eval/honeywell_autoq_35.csv        (35 auto-gen)  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌────────────────────────────┐
│  src/ingest.py│         │  graphrag_engine/           │
│  PDFPlumber   │         │  app.py + graph.py          │
│  HuggingFace  │         │  OpenAI (text-embedding-    │
│  all-MiniLM   │         │  3-large + GPT-4 for        │
│  doc-type     │         │  ontology/KG extraction)    │
│  aware split  │         │  Entity resolution          │
│  noise filter │         │  (score_exact, score_embed, │
│  + FAISS save │         │  score_llm_batch)           │
└───────┬───────┘         └────────────┬───────────────┘
        │                              │
        ▼                              ▼
  data/vectorstore/              Neo4j (AuraDB or local)
  (FAISS index)                  Knowledge graph domain
        │                              │
        ▼                              ▼
┌────────────────────┐   ┌────────────────────────────────┐
│ week1_vector_rag/  │   │ week2_graph_rag/                │
│ rag_pipeline.py    │   │ run_graphrag_predictions.py     │
│ Groq (Llama 3.1    │   │ Groq (Llama 3.3 70B)           │
│ 8B) answer gen     │   │ + VectorCypherRetriever         │
│ FAISS sim+MMR+     │   │ answer generation               │
│ cross-encoder      │   │                                 │
│ rerank             │   │                                 │
└────────┬───────────┘   └──────────────┬─────────────────┘
         │                              │
         │  outputs/predictions/        │
         │  vector_rag_hard_labels_28   │  honeywell_hard_labels_28
         │  _predictions.csv ◄──────────┘  _graphrag.csv
         │
         └───────────────────┬──────────────────────────────
                             ▼
               ┌─────────────────────────────────────┐
               │   src/week3_benchmark/              │
               │   run_week3_eval.py  (Tier 1)       │
               │   run_full_ragas_comparison.py (T2) │ ← OpenAI / Groq
               │   run_pairwise_llm_judge.py    (T3) │ ← Groq (judge)
               │   bootstrap_confidence_intervals.py │
               │   graph_coverage_metrics.py         │
               │   build_week3_dashboard.py          │
               └───────────────────┬─────────────────┘
                                   │
                                   ▼
               ┌─────────────────────────────────────┐
               │   outputs/eval_outputs/             │
               │   ├── dashboard.html         ← open │
               │   ├── eval_results.csv              │
               │   ├── metric_summary.json           │
               │   ├── bootstrap_*.csv               │
               │   ├── run_manifest.json             │
               │   └── benchmarking_comparative_     │
               │       analysis.md                   │
               └─────────────────────────────────────┘

Side: GitHub Actions CI
  → installs requirements-dashboard.txt
  → python -m pytest -p no:cacheprovider (38 tests)
  → py_compile check on 4 core modules
```

**Where LLMs are called:**
- `graphrag_engine/graph.py`: OpenAI `text-embedding-3-large` (embeddings) + `OpenAILLM` (KG extraction / ontology generation during graph build)
- `src/week1_vector_rag/rag_pipeline.py`: Groq Llama 3.1 8B (answer generation, fallback path)
- `src/week2_graph_rag/run_graphrag_predictions.py`: Groq Llama 3.3 70B (answer generation via GraphRAG)
- `src/week3_benchmark/run_pairwise_llm_judge.py`: Groq Llama 3.3 70B (judge)
- `src/week3_benchmark/run_full_ragas_comparison.py`: OpenAI or Groq (RAGAS LLM metrics)

**Where no LLM is called:**
- All Tier 1 deterministic metrics in `run_week3_eval.py` — pure token overlap and string matching, fully reproducible without API keys.

---

## What This Project Demonstrates

| Skill | Evidence |
|-------|----------|
| RAG pipeline design | `src/ingest.py` — doc-type-aware chunking, noise filtering, section header detection, metadata tagging, FAISS build; `rag_pipeline.py` — FAISS sim + MMR + lexical hybrid + cross-encoder rerank |
| Knowledge graph construction | `graphrag_engine/` — Neo4j domain build, OWL/RDF ontology generation, OpenAI embedding + LLM extraction, entity resolution (exact/embedding/LLM scoring), transitive cluster merging |
| Multi-tier LLM evaluation | Deterministic proxies (Tier 1) → RAGAS (Tier 2) → pairwise LLM judge (Tier 3); metric disagreement analysis across all three |
| Reproducible benchmarking | SHA256-verified run manifests; frozen prediction CSVs; bootstrap CIs with fixed seed; deterministic proxy metrics requiring no API calls |
| Data engineering | Two benchmark datasets (28 expert-labelled + 35 auto-generated); 5 query classes; failure taxonomy; output quality audit; review queue generation |
| Software engineering | Modular orchestration (`run_week3_pipeline.py`); pytest suite (38 tests covering metric helpers, schema validation, bootstrap, manifest, dashboard rendering); GitHub Actions CI |

---

## Tech Stack

**Core:** Python 3.11 · Neo4j (AuraDB or local) · FAISS (`faiss-cpu`) · NetworkX  
**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (vector RAG) · OpenAI `text-embedding-3-large` (GraphRAG)  
**LLMs:** Groq (Llama 3.1 8B for generation, Llama 3.3 70B for judge/GraphRAG) · OpenAI (graph build + RAGAS)  
**Evaluation:** RAGAS · pairwise LLM-as-a-judge · bootstrap resampling · Cohen's Kappa  
**Document processing:** pdfplumber · LangChain `RecursiveCharacterTextSplitter` · `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker)  
**Dashboard:** static HTML/CSS (no Streamlit dependency for dashboard) · Streamlit (graphrag_engine UI only)  
**Infra:** GitHub Actions CI · pytest · pandas

---

## Quickstart

### Prerequisites

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

```powershell
Copy-Item .env.example .env
# Edit .env: set GROQ_API_KEY, OPENAI_API_KEY, NEO4J_URI/USERNAME/PASSWORD
```

### Full pipeline (builds everything from scratch)

```powershell
# Build the FAISS vectorstore from PDFs in data/raw/
python -m src.ingest

# Run the full end-to-end pipeline
python -m src.run_week3_pipeline
```

Open `outputs/eval_outputs/dashboard.html` in a browser.

### Fast/reproducible mode (skip generation, use frozen outputs)

This skips GraphRAG domain build, prediction generation, live RAGAS, and LLM judge. Uses the committed prediction CSVs and produces deterministic metrics only.

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

**What fast mode skips:** GraphRAG domain rebuild (Neo4j), new prediction generation, live RAGAS API calls, pairwise LLM judge API calls. All Tier 1 deterministic metrics still run fully.

### Both datasets

```powershell
python -m src.run_week3_pipeline --dataset-mode both
```

---

## Repository Structure

```
data/eval/
  honeywell_hard_labels_28.csv   28 expert-labelled benchmark questions
  honeywell_autoq_35.csv         35 auto-generated questions
data/raw/                        Honeywell source PDFs (committed)
graphrag_engine/                 Neo4j GraphRAG engine — Streamlit app, KG builder,
                                 entity resolution, evidence layer
newevals/                        Standalone experimental sub-project:
                                 Graph A (co-occurrence) vs Graph B (semantic),
                                 no Neo4j required; separate pipeline and dashboard
src/ingest.py                    FAISS vectorstore builder
src/week1_vector_rag/            Vector RAG baseline — retrieval, reranking, generation
src/week2_graph_rag/             GraphRAG domain builder and prediction generator
src/week3_benchmark/             Benchmark orchestration:
                                   run_week3_eval.py       deterministic metrics
                                   run_full_ragas_comparison.py  RAGAS
                                   run_pairwise_llm_judge.py     LLM judge
                                   bootstrap_confidence_intervals.py
                                   graph_coverage_metrics.py
                                   build_week3_dashboard.py
src/run_week3_pipeline.py        Main entry point — orchestrates full pipeline
outputs/predictions/             Frozen prediction CSVs (committed, SHA256-tracked)
outputs/eval_outputs/            Generated eval artifacts (gitignored except .gitkeep)
tests/                           pytest suite (38 tests)
.github/workflows/tests.yml      CI: install, pytest, py_compile
```

---

## Reproducibility

- **Run manifest:** Every pipeline run writes `outputs/eval_outputs/run_manifest.json` containing input file paths, SHA256 hashes of dataset and prediction CSVs, git SHA, CLI arguments, and UTC timestamp.
- **Frozen predictions:** The prediction CSVs in `outputs/predictions/` are committed and SHA256-tracked so the deterministic evaluation tier can be re-run without API calls.
- **Deterministic metrics:** Tier 1 metrics in `run_week3_eval.py` use only token overlap and string matching — no LLM, no randomness. Results are bit-for-bit reproducible given the same CSV inputs.
- **Bootstrap seed:** Fixed at 5010; results are reproducible across runs.
- **Fast mode flag:** `--skip-graph-output-generation --skip-vector-output-generation --skip-live-ragas --skip-llm-judge` reproduces the headline dashboard numbers without any external API calls.

---

## Testing

```powershell
python -m pytest -p no:cacheprovider
```

**38 tests** covering:
- Deterministic metric helpers (entity recall, faithfulness, context precision, answer relevancy, winner logic, failure modes)
- CSV schema validation and ID contract enforcement
- Bootstrap confidence interval correctness
- Run manifest generation and SHA256 hashing
- Dashboard rendering from fixture files
- Metric disagreement analysis
- Output quality audit
- Graph coverage metrics
- Cost/latency audit helpers
- Pipeline orchestration (leaderboard mode, dataset-mode flags)

**CI (`.github/workflows/tests.yml`):** Runs on every push and pull request. Installs `requirements-dashboard.txt` (pandas, pytest, tqdm, streamlit, plotly, networkx, openai, python-dotenv), runs the full pytest suite, and `py_compile`-checks four core modules. Does **not** require Neo4j, Groq, or OpenAI keys in CI — all tests that need LLM calls use fixtures or stubs.

---

## Limitations

- **Small benchmark (n=28):** Per-class sample sizes are 4 rows each. Bootstrap CIs are wide; class-level win rates have high variance. Hard-query results (n=16) should not be over-interpreted.
- **Domain-specific:** All PDFs and questions are Honeywell fire-alarm and HVAC documentation. Findings may not generalise to other domains or document types.
- **Deterministic proxy metrics are proxies:** Tier 1 metrics use token overlap rather than semantic similarity. They are reproducible and debug-friendly, but may disagree with human judgment on paraphrase-heavy answers.
- **LLM judge bias:** Groq Llama 3.3 70B is used for both GraphRAG answer generation and as the pairwise judge. This is a potential source of systematic bias; judge results should be read alongside Tier 1 and RAGAS scores.
- **GraphRAG build requires Neo4j and OpenAI:** Reproducing the graph construction step requires a Neo4j instance (AuraDB or local) and an OpenAI API key. The frozen prediction CSVs allow evaluation without these.
- **`newevals/` is experimental:** The co-occurrence vs. semantic graph comparison is a standalone research experiment, not a production pipeline.

---

## Security

`.env`, API keys, FAISS vectorstore (`data/vectorstore/`), and generated `outputs/` (except committed prediction CSVs and `.gitkeep` files) are gitignored. Never commit your `.env` file or real API credentials. See `.gitignore` for the full exclusion list.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Answer generation (Llama 3.1 8B / 3.3 70B) and LLM judge |
| `OPENAI_API_KEY` | GraphRAG domain build (KG extraction, embeddings) and RAGAS |
| `LLM_PROVIDER` | `groq` (default) or `openai` |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j graph database |
| `EMBEDDING_MODEL` | HuggingFace embedding model for FAISS (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters for `src/ingest.py` |
| `TOP_K` / `FETCH_K` / `LAMBDA_MULT` | Retrieval parameters for vector RAG |

See `.env.example` for the full list including optional RAGAS throttling and cost-estimate rate variables.
