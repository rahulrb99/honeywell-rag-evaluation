# GraphEval-Ragas

GraphEval-Ragas compares GraphRAG against a vector RAG baseline on Honeywell technical documentation. The final deliverable is a static dashboard that shows system wins, query-class performance, failure modes, and reproducible proxy metrics.

## Quickstart

The default command runs the full Week 3 pipeline: GraphRAG generation, vector RAG baseline generation, metrics, 5-row live RAGAS smoke scoring, pairwise LLM-as-a-judge, and dashboard build.

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m src.ingest
.\.venv\Scripts\python -m src.run_week3_pipeline
```

Open:

```text
outputs/eval_outputs/dashboard.html
```

See [quickstart.md](quickstart.md) for the input contract, required environment variables, and fast dashboard-only mode.

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
| Week 1 | Vector RAG baseline with ingestion, FAISS retrieval, answer generation, and initial eval outputs. | `src/week1_vector_rag/`, `src/ingest.py` |
| Week 2 | GraphRAG integration using Neo4j-backed graph construction and GraphRAG prediction exports. | `src/week2_graph_rag/`, `graphrag_engine/` |
| Week 3 | Comparative benchmark layer, RAGAS-compatible packaging, failure analysis, kappa checks, and static dashboard. | `src/week3_benchmark/`, `src/grapheval_ragas/`, `src/run_week3_pipeline.py` |

## Final Pipeline

The final entry point is:

```powershell
python -m src.run_week3_pipeline
```

By default this command:

1. builds or updates the Neo4j GraphRAG domain,
2. writes GraphRAG predictions,
3. generates the vector RAG baseline when missing,
4. computes Week 3 metrics,
5. runs live 5-row RAGAS smoke scoring,
6. runs pairwise LLM-as-a-judge,
7. builds the static dashboard.

Fast dashboard-only run from supplied outputs:

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

If the Neo4j graph domain already exists and you only need fresh GraphRAG predictions:

```powershell
python -m src.run_week3_pipeline `
  --skip-graph-domain-build
```

The full path requires `requirements.txt`, `.env` credentials for Groq/OpenAI and Neo4j, and a built vector store from `python -m src.ingest`.

## Generated Outputs

The pipeline writes generated artifacts under `outputs/eval_outputs/`:

```text
dashboard.html
eval_results.csv
summary_by_query_class.csv
failure_analysis.csv
metric_summary.json
benchmarking_comparative_analysis.md
```

Manual review and existing judge/RAGAS artifacts can be supplied when available:

```powershell
python -m src.run_week3_pipeline `
  --dataset path\to\dataset.csv `
  --graph-output path\to\graphrag_outputs.csv `
  --vector-output path\to\vector_outputs.csv `
  --manual-review path\to\manual_review.csv `
  --llm-judge path\to\llm_judge.csv `
  --ragas-scores path\to\ragas_scores.json
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
outputs/predictions/       GraphRAG and vector RAG prediction CSVs
outputs/eval_outputs/     Evaluation summaries, RAGAS/judge artifacts, dashboard
quickstart.md              Short dashboard build guide
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
