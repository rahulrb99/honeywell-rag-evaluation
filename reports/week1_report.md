# Week 1 Report - Baseline RAG Evaluation

## 1) Domain and dataset

- Domain: Honeywell product catalog and compatibility documentation
- Corpus source: `data/raw/` (product sheets, compatibility notes, spec docs)
- Gold dataset: `data/eval/week1_gold_triplets.csv`
- Gold set size target: 20-30 QA/context triplets

## 2) Evaluation approach

### Metrics planned

- Faithfulness
- Answer Relevancy
- Context Precision
- (Optional) Context Recall

### Why these metrics

- Faithfulness checks hallucination risk against retrieved evidence.
- Relevancy checks if the answer addresses the asked question.
- Context precision checks whether retrieved chunks are useful/targeted.

## 3) What is Cohen's Kappa and when to use it

Cohen's Kappa measures agreement between two raters beyond chance.  
It is used in Week 3 when human reviewers annotate the same outputs and you need to verify label reliability.

## 4) Baseline pipeline summary

- Ingestion and chunking from `data/raw/`
- Vector indexing with FAISS
- Retrieval + Groq generation
- Evaluation with RAGAS

## 5) Baseline scorecard (fill after run)

| Metric | Score |
|---|---|
| Faithfulness | TBD |
| Answer Relevancy | TBD |
| Context Precision | TBD |
| Context Recall (optional) | TBD |

## 6) Screenshots checklist

- [ ] Ingestion/index build run
- [ ] Example retrieval + model answer
- [ ] RAGAS metrics output

## 7) Observations and next steps

- Initial strengths: TBD
- Initial failure patterns: TBD
- Week 2 prep: define LLM judge rubric and consistency test set
