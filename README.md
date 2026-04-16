# Week 1 Baseline RAG Evaluation (Honeywell)

This project provides a baseline RAG pipeline for document ingestion and answer evaluation.
It now includes a Graph-RAG-oriented evaluation track for these domains:

- Product Information and Compatibility
- Regulatory Compliance Navigation

## What this includes

- Ingestion + chunking + FAISS vector index build
- Retrieval + grounded answer generation using Groq
- Prediction export for evaluation
- RAGAS evaluation runners (generic and graph-rag domain view)
- Week 1 and Graph-RAG gold dataset templates

## Project structure

- `src/ingest.py` - build vector index from `data/raw/`
- `src/rag_pipeline.py` - query pipeline (retrieve + generate)
- `src/run_eval_week1.py` - run baseline predictions and RAGAS
- `src/run_eval_graph_rag.py` - run graph-rag evaluation + per-domain metrics
- `src/run_graph_rag_pipeline.py` - one command to ingest (if needed) + evaluate
- `data/eval/week1_gold_triplets.csv` - fill 20-30 QA/context rows
- `data/eval/graph_rag_gold_triplets.csv` - 30-question starter set for graph-rag domains
- `reports/week1_report.md` - Week 1 report template

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, then add your Groq key.
4. Put source documents into `data/raw/` as `.txt`, `.md`, or `.pdf`.
5. Build vector store:
   - `python -m src.ingest`
6. Fill one of the eval datasets:
   - `data/eval/week1_gold_triplets.csv` for generic baseline
   - `data/eval/graph_rag_gold_triplets.csv` for graph-rag domain split
7. Run evaluation (generic baseline):
   - `python -m src.run_eval_week1`
8. Run graph-rag evaluation:
   - `python -m src.run_eval_graph_rag`
9. Or run one command for ingest + graph-rag eval:
   - `python -m src.run_graph_rag_pipeline`

## Notes

- This baseline is intentionally simple so you can compare future Graph RAG systems against it.
- `src/run_eval_graph_rag.py` writes:
  - predictions: `outputs/graph_rag_predictions.csv`
  - metrics: `outputs/graph_rag_scores.json`
- Retrieval uses MMR with configurable defaults (`TOP_K=4`, `FETCH_K=12`, `LAMBDA_MULT=0.5`).
- Keep `TEMPERATURE=0` for reproducible evaluation comparisons.
- `LOG_RETRIEVED_CONTEXTS=true` logs raw retrieved chunk text before generation for debugging.
