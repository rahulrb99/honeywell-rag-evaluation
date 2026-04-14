# Integration Contract for Team Collaboration

This file defines the required interfaces and prerequisites between:
- RAG system owners
- Dataset/evaluation owners
- Reporting owners

The goal is to keep all components plug-compatible with minimal rework.

## 1) Query Input Contract (Eval -> RAG)

### JSON schema (required fields)

```json
{
  "query_id": "Q001",
  "question": "Is Sensor A compatible with Controller B v3.2?",
  "metadata": {
    "domain": "product_catalog",
    "difficulty": "medium"
  }
}
```

### Rules

- `query_id` (string, required, unique across dataset)
- `question` (string, required, non-empty)
- `metadata` (object, optional)

---

## 2) RAG Output Contract (RAG -> Eval/Report)

### JSON schema (minimum + recommended)

```json
{
  "query_id": "Q001",
  "answer": "Yes, Sensor A is compatible with Controller B from v3.1 onward.",
  "retrieved_contexts": [
    {
      "doc_id": "compat_matrix_2025",
      "chunk_id": "compat_matrix_2025#chunk_12",
      "text": "Sensor A supports Controller B firmware v3.1, v3.2, v3.3.",
      "score": 0.82,
      "source_type": "markdown",
      "source_path": "data/raw/compatibility_matrix.md"
    }
  ],
  "latency_ms": 1280,
  "model_name": "llama-3.1-8b-instant",
  "timestamp_utc": "2026-04-13T18:23:00Z"
}
```

### Required fields

- `query_id` (string)
- `answer` (string)
- `retrieved_contexts` (array, can be empty if no hit)
- For each retrieved context:
  - `doc_id` (string)
  - `chunk_id` (string)
  - `text` (string)

### Strongly recommended fields

- `score` (number)
- `source_type` (string)
- `source_path` (string)
- `latency_ms` (integer)
- `model_name` (string)
- `timestamp_utc` (ISO-8601 UTC string)

---

## 3) Gold Dataset Contract (Dataset Team)

The canonical Week 1 dataset file should be CSV with exact column names:

- `id`
- `question`
- `ground_truth`
- `contexts`
- `category`

Where:
- `id` is a stable unique question identifier
- `contexts` is a JSON list string (even if it contains one chunk)

### Example row format

```csv
id,question,ground_truth,contexts,category
1,Is Sensor A compatible with Controller B v3.2?,Sensor A is compatible from v3.1 onward including v3.2.,"[""Sensor A supports Controller B firmware v3.1, v3.2, v3.3""]",compatibility
```

---

## 4) Prediction Export Contract (RAG -> Eval Runner)

Predictions CSV must include:

- `id`
- `question`
- `answer`
- `retrieved_contexts` (JSON array string)
- `ground_truth`
- `contexts`

Optional:
- `latency_ms`
- `model_name`
- `run_id`

---

## 5) Ingestion Metadata Contract (Corpus Providers -> RAG)

### Document-level required metadata

- `doc_id` (stable unique ID)
- `source_path` (original location)
- `source_type` (`pdf`, `md`, `txt`, `xlsx`, etc.)

### Chunk-level required metadata

- `chunk_id` (`<doc_id>#chunk_<n>`)
- `doc_id` (parent doc)
- `text` (chunk content)

### Optional but useful

- `version`
- `updated_at_utc`
- `product_line`
- `jurisdiction` (for compliance use-cases)

---

## 6) Reproducibility and Benchmarking Prerequisites

- Fix model and version for each benchmark run.
- Use `temperature=0` for baseline comparisons.
- Keep retrieval `top_k` fixed while comparing runs.
- Normalize corpus text to UTF-8 before chunking.
- Use UTC ISO timestamps (`YYYY-MM-DDTHH:MM:SSZ`).
- Persist a `run_id` for traceability.

---

## 7) Ownership Boundaries

- **RAG team**: retrieval + generation + prediction exports
- **Dataset/Eval team**: gold data curation + metric scripts + judge/human labels
- **Report team**: scorecards, analysis, screenshots, final narrative

All teams must conform to this contract to ensure compatibility.

### Team Ownership Table

| Workstream | Owner(s) | Core Responsibilities |
|---|---|---|
| RAG system | Rahul | Build/maintain ingestion, retrieval, generation, and prediction export interfaces |
| Dataset creation + evals | Dhruti, Mike | Create gold dataset, run RAGAS/evaluation scripts, maintain metric outputs |
| Theory + report | Thanmay, Maya | Document methodology, explain metrics/theory, prepare final report narrative and summary tables |

---

## 8) Versioning Policy

- Contract version: `v1.0`
- If field names/types change, bump version and announce in team channel.
- Backward-incompatible changes require migration notes.

