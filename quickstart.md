# Quickstart

The main command is an end-to-end Week 3 runner:

```powershell
python -m src.run_week3_pipeline
```

It builds or updates GraphRAG, generates GraphRAG predictions, generates the vector RAG baseline, runs Week 3 metrics, runs live 5-row RAGAS smoke scoring, runs pairwise LLM-as-a-judge, and builds the dashboard.

## 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## 2. Configure environment

Create `.env` from `.env.example` and set:

```text
GROQ_API_KEY=...
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
```

## 3. Build vector store

```powershell
.\.venv\Scripts\python -m src.ingest
```

## 4. Run full pipeline

```powershell
.\.venv\Scripts\python -m src.run_week3_pipeline
```

Open:

```text
outputs/eval_outputs/dashboard.html
```

Generated artifacts:

```text
outputs/predictions/honeywell_hard_labels_28_graphrag.csv
outputs/predictions/vector_rag_hard_labels_28_predictions.csv
outputs/eval_outputs/eval_results.csv
outputs/eval_outputs/summary_by_query_class.csv
outputs/eval_outputs/failure_analysis.csv
outputs/eval_outputs/metric_summary.json
outputs/eval_outputs/ragas_smoke_graphrag_5_scores.json
outputs/eval_outputs/pairwise_llm_judge.csv
outputs/eval_outputs/dashboard.html
```

## Fast Dashboard-Only Mode

Use this when GraphRAG and vector prediction CSVs are already supplied:

```powershell
.\.venv\Scripts\pip install -r requirements-dashboard.txt
.\.venv\Scripts\python -m src.run_week3_pipeline `
  --dataset data\eval\honeywell_hard_labels_28.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv `
  --skip-graph-output-generation `
  --skip-vector-output-generation `
  --skip-live-ragas `
  --skip-llm-judge
```

## Input Contract

Dataset CSV required columns:

```text
id, question, ground_truth, contexts, query_class
```

RAG output CSV required columns:

```text
id, answer, retrieved_contexts
```

`contexts` and `retrieved_contexts` must be JSON arrays. Each retrieved context can be a string or an object with a `text` field.

## Useful Flags

```text
--skip-graph-domain-build       Use an existing Neo4j domain and only generate predictions.
--skip-graph-output-generation  Require the GraphRAG CSV to already exist.
--skip-vector-output-generation Require the vector CSV to already exist.
--skip-live-ragas               Skip live RAGAS smoke scoring.
--skip-llm-judge                Skip pairwise LLM-as-a-judge.
--force-regenerate-outputs      Recreate GraphRAG and vector prediction CSVs.
```

## Smoke Tests

```powershell
.\.venv\Scripts\python -m pytest -p no:cacheprovider
```
