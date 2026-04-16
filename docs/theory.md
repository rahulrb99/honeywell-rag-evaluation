# RAG Evaluation Theory (Week 1)

## 1) Evaluation metrics to use

Use a mixed metric set so we can evaluate both retrieval quality and answer quality.

- **Faithfulness (RAGAS):** checks whether the model answer is supported by retrieved context.
- **Answer Relevancy (RAGAS):** checks whether the answer addresses the user question.
- **Context Precision (RAGAS):** measures how much of retrieved context is actually useful.
- **Context Recall (RAGAS):** measures whether needed evidence was retrieved.
- **Context Hit Rate (project metric):** whether any expected gold context appears in retrieved chunks.
- **Latency (ms):** practical performance for user experience and system constraints.

Recommended reporting for this project:

- Report overall averages across all questions.
- Report per-domain scores for:
- `product_information_and_compatibility`
- `regulatory_compliance_navigation`
- Track score deltas across pipeline iterations (same eval set, same settings).

## 2) What Cohen's Kappa is and when to use it

**Cohen's Kappa** measures agreement between two raters while correcting for agreement expected by chance.

Formula:

`kappa = (p_o - p_e) / (1 - p_e)`

- `p_o`: observed agreement
- `p_e`: expected agreement by chance

When to use in this project:

- Two human evaluators label answers (for example: correct/incorrect, grounded/not grounded).
- You want inter-rater reliability, not just raw percent agreement.
- You are validating a rubric before large-scale manual evaluation.

How to interpret (rule-of-thumb):

- `< 0.20`: slight agreement
- `0.21 - 0.40`: fair
- `0.41 - 0.60`: moderate
- `0.61 - 0.80`: substantial
- `0.81 - 1.00`: near-perfect

## 3) Reference-based vs LLM-as-a-judge vs Human evaluation

### Reference-based metrics

What it is:
- Compare model outputs to gold answers/contexts.

Strengths:
- Repeatable and fast at scale.
- Good for regression testing and iteration tracking.

Limits:
- Sensitive to reference quality and wording.
- Can miss valid alternative phrasings unless metrics are semantic.

### LLM-as-a-judge

What it is:
- A separate LLM scores answer quality using a rubric.

Strengths:
- Captures nuanced quality dimensions better than lexical overlap.
- Scales better than full human review.

Limits:
- Judge model bias and instability.
- Needs prompt/rubric calibration and spot-checking.

### Human evaluation

What it is:
- Human raters evaluate correctness, grounding, completeness, and usefulness.

Strengths:
- Highest quality for nuanced/domain-specific judgment.
- Best for final decision-making and rubric validation.

Limits:
- Slow and expensive.
- Requires rater guidelines and agreement checks (use Kappa).

## Practical recommendation for this baseline

- Use **RAGAS + context hit + latency** on every run.
- Add **LLM-as-a-judge** for weekly broader quality checks.
- Use **human review** on a sampled subset, and compute **Cohen's Kappa** between two reviewers.

This gives fast iteration feedback while preserving trustworthy quality checks.
