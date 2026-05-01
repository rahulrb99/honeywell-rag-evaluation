# GraphEval-Ragas

GraphEval-Ragas compares GraphRAG against a vector RAG baseline on Honeywell technical documentation. The final deliverable is a static dashboard that shows system wins, query-class performance, failure modes, and reproducible proxy metrics.

## Quickstart

This path builds the dashboard from supplied CSV files. It does not require Neo4j, Groq, OpenAI, RAGAS, FAISS, LangChain, or sentence-transformers.

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dashboard.txt
.\.venv\Scripts\python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv
```

Open:

```text
outputs/week3/week3_dashboard.html
```

See [docs/quickstart.md](docs/quickstart.md) for the exact input contract and optional flags.

## Input Contract

Dataset CSV required columns:

```text
id, question, ground_truth, contexts, query_class
```

Recommended dataset columns:

```text
source_product, source_fields, category, reasoning_type
```

Each RAG output CSV required columns:

```text
id, answer, retrieved_contexts
```

`contexts` and `retrieved_contexts` should be JSON arrays. A retrieved context can be a string or an object with a `text` field.

The pipeline validates required columns, ID uniqueness, dataset/prediction ID coverage, and JSON array formatting before writing dashboard artifacts.

## Milestones

| Milestone | What was built | Main artifacts |
|---|---|---|
| Week 1 | Vector RAG baseline with ingestion, FAISS retrieval, answer generation, and initial eval outputs. | `src/week1_vector_rag/`, `src/ingest.py`, `src/run_all.py` |
| Week 2 | GraphRAG integration using Neo4j-backed graph construction and GraphRAG prediction exports. | `src/week2_graph_rag/`, `graph_rag_baseline/` |
| Week 3 | Comparative benchmark layer, RAGAS-compatible packaging, failure analysis, kappa checks, and static dashboard. | `src/week3_benchmark/`, `src/grapheval_ragas/`, `src/run_week3_pipeline.py` |

## Final Pipeline

The final entry point is:

```powershell
python -m src.run_week3_pipeline
```

Typical supplied-output run:

```powershell
python -m src.run_week3_pipeline `
  --dataset data\eval\honeywell_hard_labels_28.csv `
  --graph-output outputs\graph_rag\honeywell_hard_labels_28_graphrag.csv `
  --vector-output outputs\week3\vector_rag_hard_labels_28_predictions.csv
```

If no vector output CSV is supplied, the pipeline can generate one:

```powershell
python -m src.run_week3_pipeline `
  --dataset data\eval\honeywell_hard_labels_28.csv `
  --graph-output outputs\graph_rag\honeywell_hard_labels_28_graphrag.csv `
  --vector-output outputs\week3\vector_rag_hard_labels_28_predictions.csv `
  --generate-vector-output
```

That optional path requires the full `requirements.txt` environment, a built vector store, and LLM credentials.

## Generated Outputs

The pipeline writes generated artifacts under `outputs/week3/`:

```text
week3_dashboard.html
week3_eval_results.csv
week3_summary_by_query_class.csv
week3_failure_analysis.csv
week3_metric_summary.json
week3_benchmarking_comparative_analysis.md
```

Optional inputs can add kappa and RAGAS smoke cards to the dashboard:

```powershell
python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv `
  --manual-review path\to\manual_review.csv `
  --llm-judge path\to\llm_judge.csv `
  --ragas-scores path\to\ragas_scores.json
```

If API credentials are available, run pairwise LLM-as-a-judge validation:

```powershell
python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv `
  --run-llm-judge
```

## Evaluation Notes

The dashboard uses deterministic proxy metrics for full-coverage reproducibility:

- entity recall proxy
- context precision proxy
- faithfulness proxy
- answer relevancy proxy
- answer correctness proxy

These metrics are designed for ranking, debugging, and failure analysis. They are not claimed to be perfect human correctness judgments. Human review, pairwise LLM-as-a-judge, and live RAGAS smoke scores are supported validation layers.

## Tests

Run the lightweight test suite:

```powershell
python -m pytest -p no:cacheprovider
```

Current tests cover metric helpers, failure-mode logic, CSV schema validation, result construction, and dashboard HTML rendering.

## Repository Map

```text
data/eval/                 Benchmark CSVs
data/raw/                  Honeywell source documents
docs/quickstart.md         Short dashboard build guide
reports/week3_runbook.md   Full reproduction notes
src/run_week3_pipeline.py  Final dashboard pipeline
src/week1_vector_rag/      Week 1 vector RAG baseline
src/week2_graph_rag/       Week 2 GraphRAG integration
src/week3_benchmark/       Week 3 comparison and dashboard code
src/grapheval_ragas/       RAGAS-compatible sample packaging
tests/                     Lightweight pytest suite
```

## Dependency Profiles

Use the minimal dashboard dependencies for normal grading:

```powershell
pip install -r requirements-dashboard.txt
```

Use the full dependencies only for from-scratch vector, GraphRAG, or live RAGAS work:

```powershell
pip install -r requirements.txt
```

## Security

Do not commit `.env`, API keys, local vector stores, or generated `outputs/` artifacts. Generated outputs are intentionally ignored and should be rebuilt from supplied inputs.
