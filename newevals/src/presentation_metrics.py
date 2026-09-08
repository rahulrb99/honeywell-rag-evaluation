from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import DEFAULT_MODEL
from .io_utils import read_json, write_json


JUDGE_PROMPT = """You are judging QA answers for a Honeywell technical-document GraphRAG experiment.
Compare each predicted answer to the gold answer.

Score:
- 1.0 = substantially correct and captures the key facts
- 0.5 = partially correct but missing important details or includes minor unsupported detail
- 0.0 = incorrect, refusal, or does not answer the question

Return only JSON:
{{"judgments":[{{"question_id":"...", "graph":"...", "semantic_score":0.0, "rationale":"..."}}]}}

Rows:
{rows}
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def judge_rows(question_rows: list[dict[str, str]], benchmark_rows: list[dict[str, Any]], model: str = DEFAULT_MODEL) -> list[dict[str, Any]]:
    gold_by_id = {str(row["id"]): row for row in benchmark_rows}
    payload = []
    for row in question_rows:
        gold = gold_by_id.get(row["question_id"], {})
        payload.append(
            {
                "question_id": row["question_id"],
                "graph": row["graph"],
                "question": row["question"],
                "gold_answer": gold.get("answer", ""),
                "predicted_answer": row.get("answer", ""),
            }
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for semantic judging.")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(rows=json.dumps(payload, ensure_ascii=True))}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = _extract_json(response.choices[0].message.content or "{}")
    judgments = parsed.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("Judge did not return a judgments array.")
    return judgments


def build_presentation_metrics(
    *,
    question_results_path: Path,
    benchmark_path: Path,
    out_rows_path: Path,
    out_summary_path: Path,
    out_html_path: Path,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    rows = _read_csv(question_results_path)
    benchmark = read_json(benchmark_path, default=[]) or []
    judgments = judge_rows(rows, benchmark, model=model)
    judgment_by_key = {(str(j["question_id"]), str(j["graph"])): j for j in judgments}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        judgment = judgment_by_key.get((row["question_id"], row["graph"]), {})
        score = float(judgment.get("semantic_score", 0.0) or 0.0)
        answer = row.get("answer", "").lower()
        refusal = "cannot provide" in answer or "unable to" in answer or "does not contain" in answer
        enriched.append(
            {
                **row,
                "semantic_score": score,
                "semantic_correct": score >= 0.75,
                "judge_rationale": str(judgment.get("rationale", "")),
                "refusal": refusal,
                "answer_produced": bool(row.get("answer", "").strip()) and not refusal,
            }
        )
    _write_csv(out_rows_path, enriched)

    summary = []
    for graph in ("graph_a", "graph_b"):
        subset = [row for row in enriched if row["graph"] == graph]
        if not subset:
            continue
        summary.append(
            {
                "graph": graph,
                "semantic_accuracy": round(sum(1 for row in subset if row["semantic_correct"]) / len(subset), 4),
                "avg_semantic_score": round(sum(float(row["semantic_score"]) for row in subset) / len(subset), 4),
                "answer_rate": round(sum(1 for row in subset if row["answer_produced"]) / len(subset), 4),
                "refusal_rate": round(sum(1 for row in subset if row["refusal"]) / len(subset), 4),
                "exact_match_accuracy": round(sum(1 for row in subset if str(row.get("correct")).lower() == "true") / len(subset), 4),
                "questions": len(subset),
            }
        )
    write_json(out_summary_path, summary)
    _write_csv(out_summary_path.with_suffix(".csv"), summary)
    out_html_path.write_text(_render_html(summary, enriched), encoding="utf-8")
    return {"summary": summary, "rows": enriched}


def _render_html(summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    def table(items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        cols = list(items[0].keys())
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = "".join("<tr>" + "".join(f"<td>{str(row.get(c, ''))[:500]}</td>" for c in cols) + "</tr>" for row in items)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Partial Graph 7 Presentation Metrics</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d1d5db;padding:8px;vertical-align:top}}th{{background:#f3f4f6}}.note{{background:#fff7ed;border:1px solid #fed7aa;padding:12px;border-radius:6px}}</style>
</head><body>
<h1>Partial Graph 7 Presentation Metrics</h1>
<div class="note">Use semantic accuracy for presentation. Exact-match accuracy is retained as a strict baseline and is expected to undercount long-form answers.</div>
<h2>Summary</h2>{table(summary)}
<h2>Judged Rows</h2>{table(rows)}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-results", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out-prefix", default="partial_graph7")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    base = Path("newevals/results")
    result = build_presentation_metrics(
        question_results_path=Path(args.question_results),
        benchmark_path=Path(args.benchmark),
        out_rows_path=base / f"presentation_rows_{args.out_prefix}.csv",
        out_summary_path=base / f"presentation_summary_{args.out_prefix}.json",
        out_html_path=base / f"presentation_metrics_{args.out_prefix}.html",
        model=args.model,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
