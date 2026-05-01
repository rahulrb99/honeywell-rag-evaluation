# Week 3 GraphEval-Ragas Runbook

This runbook reproduces the Week 3 GraphEval-Ragas artifacts on the `codex/week3-benchmark` branch.

## 1. Run the Full Week 3 Evaluator

The full evaluator compares GraphRAG against a standard top-k vector search baseline RAG and writes the row-level Pandas DataFrame, query-class summary, failure analysis, and metric summary.

```powershell
.\.venv\Scripts\python -m src.week3_benchmark.run_week3_eval --no-generate-vector
```

Primary outputs:

- `outputs/week3/week3_eval_results.csv`
- `outputs/week3/week3_summary_by_query_class.csv`
- `outputs/week3/week3_failure_analysis.csv`
- `outputs/week3/week3_metric_summary.json`

If the vector baseline needs to be regenerated, omit `--no-generate-vector`.

## 2. Package GraphRAG Outputs as RAGAS Samples

This creates explicit RAGAS `SingleTurnSample` objects and wraps them in an `EvaluationDataset`.

```powershell
.\.venv\Scripts\python -m src.grapheval_ragas.build_dataset `
  --predictions outputs\graph_rag\honeywell_hard_labels_28_graphrag.csv `
  --system-name graphrag_hard_labels_28 `
  --output-json outputs\week3\grapheval_ragas_graphrag_28_samples.json `
  --output-csv outputs\week3\grapheval_ragas_graphrag_28_samples.csv
```

Smoke-subset packaging:

```powershell
.\.venv\Scripts\python -m src.grapheval_ragas.build_dataset `
  --predictions outputs\week3\ragas_smoke_graphrag_5.csv `
  --system-name graphrag_ragas_smoke_5 `
  --output-json outputs\week3\grapheval_ragas_smoke_5_samples.json `
  --output-csv outputs\week3\grapheval_ragas_smoke_5_samples.csv
```

## 3. Run the 5-Row RAGAS Smoke Validation

Full live RAGAS over all 28 rows was runtime-limited, so live RAGAS is run on five representative hard-query rows.

```powershell
$env:GRAPH_PREDICTIONS_CSV='outputs/week3/ragas_smoke_graphrag_5.csv'
$env:GRAPH_RAGAS_SCORES_JSON='outputs/week3/ragas_smoke_graphrag_5_scores.json'
$env:GRAPH_RAGAS_ROWS_CSV='outputs/week3/ragas_smoke_graphrag_5_rows.csv'
$env:RAGAS_MAX_WORKERS='1'
$env:RAGAS_TIMEOUT_SEC='900'
$env:GROQ_MAX_RETRIES='8'

.\.venv\Scripts\python -m src.week2_graph_rag.run_eval_week2
```

Smoke outputs:

- `outputs/week3/ragas_smoke_graphrag_5_scores.json`
- `outputs/week3/ragas_smoke_graphrag_5_rows.csv`

## 4. Expected Headline Results

- Full 28-row benchmark: GraphRAG wins `19`, vector RAG wins `4`, ties `5`.
- Hard thematic/multi-hop classes: GraphRAG win rate `0.75`.
- RAGAS smoke scores: faithfulness `0.8897`, answer relevancy `0.7340`, context precision `0.9611`, context recall `1.0000`.

## 5. Run Cohen's Kappa Agreement Checks

This compares human manual verdicts against the LLM-as-a-judge labels and the Week 3 metric winners.

```powershell
.\.venv\Scripts\python -m src.week3_benchmark.run_kappa_analysis
```

Kappa outputs:

- `outputs/week3/week3_kappa_analysis.csv`
- `outputs/week3/week3_kappa_summary.json`

Expected values:

- Human vs LLM-as-a-judge kappa: `0.0000`
- Human vs Week 3 metric-winner kappa: `0.2806`

## 6. Build the HTML Dashboard

The dashboard is a static HTML summary of the final Week 3 CSV/JSON outputs.

```powershell
.\.venv\Scripts\python -m src.week3_benchmark.build_week3_dashboard
```

Dashboard output:

- `outputs/week3/week3_dashboard.html`
