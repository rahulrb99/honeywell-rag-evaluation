# Week 1 Baseline RAG Evaluation (Honeywell Product Catalog)

This starter project provides a simple baseline RAG pipeline and Week 1 evaluation skeleton.

## What this includes

- Ingestion + chunking + FAISS vector index build
- Retrieval + grounded answer generation using Groq
- Prediction export for evaluation
- RAGAS evaluation runner skeleton
- Week 1 gold dataset and report templates

## Project structure

- `src/ingest.py` - build vector index from `data/raw/`
- `src/rag_pipeline.py` - query pipeline (retrieve + generate)
- `src/run_eval_week1.py` - run baseline predictions and RAGAS
- `data/eval/week1_gold_triplets.csv` - fill 20-30 QA/context rows
- `reports/week1_report.md` - Week 1 report template

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, then add your Groq key.
4. Put source documents into `data/raw/` as `.txt`, `.md`, or `.pdf`.
5. Build vector store:
   - `python -m src.ingest`
6. Fill `data/eval/week1_gold_triplets.csv` with at least 20 rows.
7. Run evaluation:
   - `python -m src.run_eval_week1`

## Notes

- This is intentionally simple for Week 1.
- Other teams' advanced MDM/Graph RAG systems can be evaluated later using the same dataset/eval approach.
- Retrieval uses MMR with configurable defaults (`TOP_K=4`, `FETCH_K=12`, `LAMBDA_MULT=0.5`).
- Keep `TEMPERATURE=0` for reproducible evaluation comparisons.
- `LOG_RETRIEVED_CONTEXTS=true` logs raw retrieved chunk text before generation for debugging.
