# Quickstart

This path builds the dashboard from supplied CSV files. It does not require Neo4j, Groq, OpenAI, RAGAS, FAISS, LangChain, or sentence-transformers.

## 1. Install minimal dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dashboard.txt
```

## 2. Provide input files

Dataset CSV required columns:

```text
id, question, ground_truth, contexts, query_class
```

Recommended dataset columns:

```text
source_product, source_fields, category, reasoning_type
```

GraphRAG output CSV required columns:

```text
id, answer, retrieved_contexts
```

Vector/baseline output CSV required columns:

```text
id, answer, retrieved_contexts
```

`contexts` and `retrieved_contexts` should be JSON arrays. Each retrieved context can be either a string or an object with a `text` field.

The pipeline validates input files before building the dashboard. It reports clear errors for missing required columns, empty files, blank or duplicate `id` values, malformed JSON array columns, missing predictions for dataset IDs, and prediction IDs that do not exist in the dataset.

## 3. Generate dashboard

```powershell
.\.venv\Scripts\python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv
```

Open:

```text
outputs/week3/week3_dashboard.html
```

The pipeline also writes:

```text
outputs/week3/week3_eval_results.csv
outputs/week3/week3_summary_by_query_class.csv
outputs/week3/week3_failure_analysis.csv
outputs/week3/week3_metric_summary.json
outputs/week3/week3_benchmarking_comparative_analysis.md
```

## Optional: generate vector baseline

If a vector/baseline output CSV is not supplied:

```powershell
.\.venv\Scripts\python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output outputs\week3\vector_rag_predictions.csv `
  --generate-vector-output
```

This optional path requires the full `requirements.txt` environment, a built vector store, and LLM credentials.

## Optional: include agreement or RAGAS smoke scores

```powershell
.\.venv\Scripts\python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv `
  --manual-review path\to\manual_review.csv `
  --llm-judge path\to\llm_judge.csv `
  --ragas-scores path\to\ragas_scores.json
```

## Optional: run pairwise LLM judge

If API credentials are available, the pipeline can run a pairwise judge over GraphRAG vs vector answers and add judge/metric agreement cards to the dashboard:

```powershell
.\.venv\Scripts\python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv `
  --run-llm-judge
```

## Smoke tests

```powershell
.\.venv\Scripts\python -m pytest -p no:cacheprovider
```
