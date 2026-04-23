# Week 2 Report - LLM-as-a-Judge Evaluation

## 1) Domain and dataset

- Domain: Honeywell product/support QA based on ingested docs in `data/raw/`
- Corpus source: same Week 1 baseline corpus
- Gold dataset used by current Week 2 code: `data/eval/week1_gold_triplets.csv` (`settings.eval_csv`)
- Prediction input used by judge: `outputs/week1_predictions.csv` (`settings.predictions_csv`)
- Evaluation target: score the same Week 1 outputs with an LLM judge and compare to Week 1 RAGAS

## 2) Evaluation approach

### Metrics planned

- Faithfulness (judge score, 1-5 and normalized 0-1)
- Answer Relevance (judge score, 1-5 and normalized 0-1)
- Completeness (judge score)
- Correctness (judge score, compared to provided ground truth)
- Conciseness (judge score)
- Consistency across paraphrase variants

### Why these metrics

- Faithfulness estimates whether the answer is grounded in retrieved evidence.
- Answer relevance estimates whether the response directly answers the question.
- Completeness/correctness/conciseness provide additional quality controls required by the rubric.
- Consistency testing checks robustness of judge scoring across paraphrased inputs.

## 3) LLM judge rubric and implementation

### Rubric (1-5 scale)

- Faithfulness: 1 = unsupported/hallucinated, 5 = fully supported by retrieved context
- Answer Relevance: 1 = off-topic, 5 = directly and completely answers the question
- Completeness: 1 = misses key info, 5 = covers key expected info
- Correctness: 1 = conflicts with reference, 5 = aligns with reference
- Conciseness: 1 = verbose/noisy, 5 = clear and compact

### Implementation summary

- Judge model (default): `llama-3.1-70b-versatile` via `JUDGE_MODEL` in `src/config.py`
- Fallback model if judge model is unavailable: `GROQ_MODEL` (default `llama-3.1-8b-instant`)
- Scoring prompt: strict JSON schema in `src/llm_judge.py` with per-metric rationale
- Pipeline implementation:
- `python -m src.run_week1_predictions` generates baseline outputs
- `python -m src.run_eval_llm_judge` scores each row using `score_answer(...)`
- Output artifacts:
- `outputs/week1_judge_scores.csv` (row-level scores + rationales)
- `outputs/week1_judge_scores.json` (aggregate means)

## 4) Consistency testing summary

- Implemented in `python -m src.run_consistency_check`
- Current defaults from env/config:
- `PARAPHRASE_VARIANTS=5` (plus original question = 6 variants total)
- `CONSISTENCY_QUESTIONS=5`
- Process:
- Generate paraphrases with the judge model
- Run baseline RAG answer generation per variant
- Score each variant with the LLM judge
- Compute per-question mean/std/min/max for each metric


## 5) Week 1 vs Week 2 comparison table 

| Metric | Week 1 (RAGAS) | Week 2 (LLM Judge) | Notes |
|---|---|---|---|
| Faithfulness | Not run yet | Not run yet | Code maps judge faithfulness (0-1) vs RAGAS faithfulness |
| Answer Relevancy | Not run yet | Not run yet | Code maps judge answer relevance (0-1) vs RAGAS answer_relevancy |
| Context Precision | Not run yet | N/A | RAGAS-only metric |
| Context Recall | Not run yet | N/A | RAGAS-only metric |
| Consistency (paraphrase variance) | N/A | Not run yet | Produced by `src/run_consistency_check.py` |

## 6) Screenshots checklist

- [ ] Judge rubric prompt/config
- [ ] Sample scored outputs with rationale
- [ ] Consistency test results across paraphrase variants
- [ ] Week 1 vs Week 2 comparison output/table

## 7) Observations and next steps

- Current implementation status:
- Week 2 LLM-as-a-judge layer is implemented in code (`src/llm_judge.py`, `src/run_eval_llm_judge.py`, `src/run_consistency_check.py`).
- Comparison logic against Week 1 RAGAS is implemented, but evaluation artifacts are not generated yet.
