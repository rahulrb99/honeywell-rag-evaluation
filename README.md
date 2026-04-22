# Week 1 Baseline RAG Evaluation (Honeywell)

This project provides a baseline RAG pipeline for document ingestion and answer evaluation.

## What this includes

- Ingestion + chunking + FAISS vector index build
- Retrieval + grounded answer generation using Groq
- Prediction export for evaluation
- RAGAS evaluation runner for Week 1 baseline
- Week 1 gold dataset template

## Project structure

- `src/ingest.py` - build vector index from `data/raw/`
- `src/rag_pipeline.py` - query pipeline (retrieve + generate)
- `src/run_rag_outputs.py` - generate raw RAG outputs CSV
- `src/run_eval_week1.py` - run RAGAS metrics on raw RAG outputs
- `src/run_eval_llm_judge.py` - run LLM-as-a-judge scoring
- `src/run_consistency_check.py` - run paraphrase consistency and judge-vs-RAGAS comparison
- `src/run_eval_retrieval_only.py` - retrieval coverage metrics: retrieval-only diagnostics + recall@k
- `src/run_eval_generation_only.py` - generation-only evaluation using gold contexts
- `src/build_eval_artifacts.py` - consolidate eval outputs into one row CSV and one summary JSON
- `src/build_failure_dashboard.py` - aggregate failures into an HTML dashboard
- `src/run_all.py` - one-command full pipeline (ingest + evals + dashboard)
- `data/eval/week1_gold_triplets_20.csv` - fill 20-30 QA/context rows
- `reports/week1_report.md` - Week 1 report template

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, then add your Groq key.
4. Put source documents into `data/raw/` as `.txt`, `.md`, or `.pdf`.
5. Build vector store:
   - `python -m src.ingest`
6. Fill `data/eval/week1_gold_triplets_20.csv` with at least 20 rows.
7. Generate raw RAG outputs:
   - `python -m src.run_rag_outputs`
8. Run RAGAS evaluation:
   - `python -m src.run_eval_week1`
9. Run LLM judge evaluation:
   - `python -m src.run_eval_llm_judge`
10. Run paraphrase consistency + comparison:
   - `python -m src.run_consistency_check`
11. Run retrieval coverage evaluation:
   - `python -m src.run_eval_retrieval_only`
12. Run generation-only evaluation (gold contexts):
   - `python -m src.run_eval_generation_only`
13. Build consolidated eval artifacts:
   - `python -m src.build_eval_artifacts`
14. Build failure dashboard:
   - `python -m src.build_failure_dashboard`
15. One-command dashboard refresh (PowerShell):
   - Dashboard only: `.\refresh_dashboard.ps1`
   - Recompute new eval outputs + dashboard: `.\refresh_dashboard.ps1 -Full`
16. One-command full pipeline:
   - Python: `python -m src.run_all`
   - PowerShell wrapper: `.\refresh_dashboard.ps1 -All`

## Notes

- This baseline is intentionally simple so you can compare future RAG variants against it.
- Retrieval uses MMR with configurable defaults (`TOP_K=4`, `FETCH_K=12`, `LAMBDA_MULT=0.5`).
- Keep `TEMPERATURE=0` for reproducible evaluation comparisons.
- `LOG_RETRIEVED_CONTEXTS=true` logs raw retrieved chunk text before generation for debugging.
- Judge artifacts are written under `outputs/`:
  - `week1_judge_scores.csv`
  - `week1_judge_scores.json`
  - `week1_judge_consistency.json`
  - `week1_judge_vs_ragas.json`
- Additional 4-eval artifacts:
  - `week1_eval_rows.csv`
  - `week1_eval_summary.json`
  - `dashboard/failure_dashboard.html`
  - Debug/intermediate files are written under `outputs/debug/`
- Useful env overrides for judge runs:
  - `JUDGE_MODEL`, `JUDGE_TEMPERATURE`
  - `PARAPHRASE_VARIANTS` (default 5), `CONSISTENCY_QUESTIONS` (default 5)
