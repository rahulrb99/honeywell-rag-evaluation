# newevals Graph_RAG Evaluation

Standalone benchmark for comparing:

- Graph A: local NetworkX co-occurrence graph.
- Graph B: local hybrid semantic relation graph.

## Inputs

By default, this subproject reuses the existing repo PDFs from:

```text
data/raw/
```

If you add PDFs here, they take precedence:

```text
newevals/data/raw_pdfs/
```

By default, the runner converts the existing repo eval dataset:

```text
data/eval/honeywell_hard_labels_28.csv
```

into the frozen benchmark:

```text
newevals/qa/benchmark_v1.json
```

Create `benchmark_v2.json` instead of editing v1 when changing the benchmark questions for a new experiment series.

## Run

Build local artifacts:

```powershell
python -m newevals.src.run_pipeline
```

Build only Graph B from existing chunks/entities:

```powershell
python -m newevals.src.build_graph_b
```

Graph B v1 does not require Neo4j. By default it uses sentence-level rules plus
cached direct OpenAI relation extraction on the highest-signal chunks when
`OPENAI_API_KEY` is set. To force zero-credit mode:

```powershell
python -m newevals.src.build_graph_b --relation-mode rules
```

To spend a bounded amount for a better semantic graph:

```powershell
python -m newevals.src.build_graph_b --relation-mode hybrid --max-llm-chunks 48
```

The checked default is 48 LLM chunks, which is enough to add typed semantic
relations without making the build run for a long time.

Evaluate from existing chunks and graph exports:

```powershell
python -m newevals.src.evaluate
```

Open the dashboard:

```text
newevals/results/dashboard.html
```

## Main Outputs

```text
newevals/results/question_results.csv
newevals/results/aggregate_results.csv
newevals/results/aggregate_results.json
newevals/results/graph_metadata.json
newevals/results/run_manifest.json
newevals/results/dashboard.html
```
