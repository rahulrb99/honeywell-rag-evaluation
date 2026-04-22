"""Build a simple HTML dashboard showing misses/failures across evaluations."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_settings


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return pd.read_csv(p)


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    return df.to_html(index=False, border=0, classes="table", escape=False)


def _safe_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _safe_value(df: pd.DataFrame, column: str, default: Any = None) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=object)
    return df[column].fillna(default)


def _norm_retrieval_hit(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "uncertain":
            return "uncertain"
    return value


def _norm_grounded_used(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v in {"false", "uncertain", ""}:
            return False
    return bool(value)


def _table_with_row_classes(df: pd.DataFrame, row_class_fn) -> str:
    """Render table with row-level CSS classes based on row content."""
    if df.empty:
        return "<p>No rows.</p>"
    base_cols = list(df.columns)
    with_class = df.copy()
    with_class["__row_class"] = with_class.apply(row_class_fn, axis=1)
    table_html = with_class.to_html(
        index=False,
        border=0,
        classes="table",
        escape=False,
        columns=base_cols,
    )
    rows = table_html.split("<tr>")
    if len(rows) <= 1:
        return table_html
    rebuilt = [rows[0], "<tr>" + rows[1]]
    for row_html, row_class in zip(rows[2:], with_class["__row_class"].tolist()):
        rebuilt.append(f'<tr class="{row_class}">' + row_html)
    return "".join(rebuilt)


def main() -> None:
    settings = get_settings()

    retrieval_df = _read_csv(settings.retrieval_only_csv)
    generation_df = _read_csv(settings.generation_only_csv)
    end_to_end_judge_df = _read_csv(settings.judge_scores_csv)
    grounded_df = _read_csv(settings.groundedness_csv)

    ragas = _read_json(settings.ragas_scores_json)
    retrieval_summary = _read_json(settings.retrieval_only_json)
    generation_summary = _read_json(settings.generation_only_json)
    recall_summary = _read_json(str(Path(settings.retrieval_recall_csv).with_suffix(".json")))

    # Normalize IDs for joins
    for df in (retrieval_df, generation_df, end_to_end_judge_df, grounded_df):
        df["id"] = df["id"].astype(str)

    merged = (
        retrieval_df[["id", "question", "category", "retrieval_hit", "recall_hit", "best_overlap_score", "failure_reason", "top_chunk_text"]]
        .merge(
            generation_df[["id", "judge_mean_1_to_5", "correctness_score", "answer"]],
            on="id",
            how="left",
            suffixes=("", "_gen"),
        )
        .merge(
            end_to_end_judge_df[["id", "judge_mean_1_to_5", "correctness_score"]],
            on="id",
            how="left",
            suffixes=("_generation_only", "_end_to_end"),
        )
        .merge(
            grounded_df[["id", "failure_tag", "max_chunk_sim", "used_status"]].rename(
                columns={"used_status": "grounded_used"}
            ),
            on="id",
            how="left",
        )
    )
    merged["recall_best_overlap_score"] = merged["best_overlap_score"]

    if "retrieval_hit" not in merged.columns:
        merged["retrieval_hit"] = None
    merged["correctness_score_generation_only"] = _safe_numeric(
        merged, "correctness_score_generation_only"
    )
    merged["correctness_score_end_to_end"] = _safe_numeric(
        merged, "correctness_score_end_to_end"
    )

    if "grounded_used" not in merged.columns:
        merged["grounded_used"] = False
    merged["grounded_used"] = merged["grounded_used"].apply(_norm_grounded_used)

    def classify_failure(row):
        retrieval_hit = row.get("retrieval_hit")
        grounded = row.get("grounded_used")
        correctness = row.get("correctness_score_end_to_end", 5)

        # Normalize values
        if isinstance(retrieval_hit, str):
            retrieval_hit_lower = retrieval_hit.lower()
            if retrieval_hit_lower == "true":
                retrieval_hit = True
            elif retrieval_hit_lower == "false":
                retrieval_hit = False
            elif retrieval_hit_lower == "uncertain":
                retrieval_hit = "uncertain"

        if retrieval_hit is False:
            return "retrieval_miss"

        if retrieval_hit == "uncertain":
            return "weak_context"

        if retrieval_hit is True and not grounded:
            return "unsupported_answer"

        if retrieval_hit is True and grounded and correctness < 3:
            return "incorrect_answer"

        return "correct"

    merged["failure_type"] = merged.apply(classify_failure, axis=1)
    merged["retrieval_hit"] = merged["retrieval_hit"].apply(_norm_retrieval_hit)
    merged["retrieval_hit_label"] = merged["retrieval_hit"].map(
        {True: "hit", False: "miss", "uncertain": "uncertain"}
    ).fillna("uncertain")

    failure_summary = (
        merged["failure_type"]
        .value_counts()
        .reset_index()
    )
    failure_summary.columns = ["failure_type", "count"]

    matrix = (
        merged.groupby(
            ["retrieval_hit_label", "failure_type"]
        )
        .size()
        .reset_index(name="count")
    )

    high_roi = merged[merged["failure_type"] == "incorrect_answer"]

    uncertain = merged[
        merged["retrieval_hit"].isin(["uncertain", None])
    ]

    failures = merged[merged["failure_type"] != "correct"].copy()
    failures = failures.sort_values(
        by=["failure_type", "id"],
        ascending=[True, True],
    )

    by_category = (
        failures.groupby("category", dropna=False)
        .agg(
            questions=("id", "count"),
            retrieval_misses=("failure_type", lambda s: int((s == "retrieval_miss").sum())),
            weak_context=("failure_type", lambda s: int((s == "weak_context").sum())),
            unsupported_answers=("failure_type", lambda s: int((s == "unsupported_answer").sum())),
            incorrect_answers=("failure_type", lambda s: int((s == "incorrect_answer").sum())),
        )
        .reset_index()
    )

    top_fail_cols = [
        "id",
        "question",
        "category",
        "retrieval_hit",
        "failure_reason",
        "correctness_score_generation_only",
        "correctness_score_end_to_end",
        "failure_tag",
        "top_chunk_text",
    ]
    top_failures = failures[top_fail_cols].head(25).copy()
    top_failures["top_chunk_text"] = top_failures["top_chunk_text"].fillna("").apply(
        lambda s: html.escape(str(s))
    )
    merged_preview = merged.head(25).copy()
    merged_preview["top_chunk_text"] = (
        _safe_value(merged_preview, "top_chunk_text", "")
        .astype(str)
        .str[:200]
        .apply(html.escape)
    )

    ragas_scores = ragas.get("scores", ragas)
    failure_counts = merged["failure_type"].value_counts()
    failure_counts_filtered = failure_counts.drop("correct", errors="ignore")
    top_issue = (
        failure_counts_filtered.idxmax()
        if not failure_counts_filtered.empty
        else "none"
    )
    retrieval_pct = (
        failure_counts.get("retrieval_miss", 0) / len(merged) if len(merged) else 0.0
    )
    unsupported_pct = (
        failure_counts.get("unsupported_answer", 0) / len(merged) if len(merged) else 0.0
    )
    incorrect_pct = (
        failure_counts.get("incorrect_answer", 0) / len(merged) if len(merged) else 0.0
    )
    weak_context_pct = (
        failure_counts.get("weak_context", 0) / len(merged) if len(merged) else 0.0
    )

    def highlight_row(row):
        if row["failure_type"] == "retrieval_miss":
            return "bad"
        elif row["failure_type"] == "weak_context":
            return "warn"
        elif row["failure_type"] in {"unsupported_answer", "incorrect_answer"}:
            return "warn"
        return "good"

    cards = {
        "rows_total": int(len(merged)),
        "retrieval_misses": int((merged["failure_type"] == "retrieval_miss").sum()),
        "weak_context_count": int((merged["failure_type"] == "weak_context").sum()),
        "unsupported_answers": int((merged["failure_type"] == "unsupported_answer").sum()),
        "incorrect_answers": int((merged["failure_type"] == "incorrect_answer").sum()),
        "retrieval_issue_pct": round(retrieval_pct * 100, 1),
        "unsupported_answer_pct": round(unsupported_pct * 100, 1),
        "incorrect_answer_pct": round(incorrect_pct * 100, 1),
        "weak_context_pct": round(weak_context_pct * 100, 1),
        "ragas_faithfulness": ragas_scores.get("faithfulness"),
        "ragas_context_precision": ragas_scores.get("context_precision"),
        "retrieval_hit_rate": retrieval_summary.get("retrieval_hit_rate_excluding_uncertain"),
        "generation_only_judge_mean": generation_summary.get("judge_mean_0_to_1"),
        "recall_at_k": recall_summary.get("recall_at_k"),
        "recall_at_k_strict": recall_summary.get("recall_at_k_strict"),
        "uncertain_rate": recall_summary.get("uncertain_rate"),
    }

    html_out = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RAG Failure Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa; }}
    .k {{ font-size: 12px; color: #555; margin-bottom: 4px; }}
    .v {{ font-size: 20px; font-weight: 700; }}
    .table {{ border-collapse: collapse; width: 100%; margin-top: 8px; margin-bottom: 20px; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }}
    .table th {{ background: #f3f3f3; }}
    .bad {{ background-color: #ffe6e6; }}
    .warn {{ background-color: #fff5cc; }}
    .good {{ background-color: #e6ffe6; }}
  </style>
</head>
<body>
  <h1>RAG Failure Dashboard</h1>
  <h2>Failure Type Definitions</h2>
  <ul>
    <li><b>retrieval_miss</b>: Ground truth not retrieved</li>
    <li><b>weak_context</b>: Retrieved context partially relevant or insufficient</li>
    <li><b>unsupported_answer</b>: Answer not grounded in retrieved context</li>
    <li><b>incorrect_answer</b>: Answer grounded but incorrect</li>
    <li><b>correct</b>: All signals aligned</li>
  </ul>
  <h2>Recommended Next Step</h2>
  <p><b>Primary Bottleneck:</b> {top_issue}</p>
  <p>Combined view of black-box retrieval, groundedness, and end-to-end correctness signals.</p>

  <div class="cards">
    <div class="card"><div class="k">Total Questions</div><div class="v">{cards["rows_total"]}</div></div>
    <div class="card"><div class="k">Retrieval Misses</div><div class="v">{cards["retrieval_misses"]}</div></div>
    <div class="card"><div class="k">Weak Context Cases</div><div class="v">{cards["weak_context_count"]}</div></div>
    <div class="card"><div class="k">Unsupported Answers</div><div class="v">{cards["unsupported_answers"]}</div></div>
    <div class="card"><div class="k">Incorrect Answers</div><div class="v">{cards["incorrect_answers"]}</div></div>
    <div class="card"><div class="k">Recall@K</div><div class="v">{cards["recall_at_k"]}</div></div>
    <div class="card"><div class="k">Recall@K (Strict)</div><div class="v">{cards["recall_at_k_strict"]}</div></div>
    <div class="card"><div class="k">Uncertain Rate</div><div class="v">{cards["uncertain_rate"]}</div></div>
    <div class="card"><div class="k">Retrieval Hit Rate (confident)</div><div class="v">{cards["retrieval_hit_rate"]}</div></div>
    <div class="card"><div class="k">Generation-only Judge Mean (0-1)</div><div class="v">{cards["generation_only_judge_mean"]}</div></div>
    <div class="card"><div class="k">RAGAS Faithfulness</div><div class="v">{cards["ragas_faithfulness"]}</div></div>
    <div class="card"><div class="k">RAGAS Context Precision</div><div class="v">{cards["ragas_context_precision"]}</div></div>
    <div class="card"><div class="k">% Retrieval Misses</div><div class="v">{cards["retrieval_issue_pct"]}%</div></div>
    <div class="card"><div class="k">% Unsupported Answers</div><div class="v">{cards["unsupported_answer_pct"]}%</div></div>
    <div class="card"><div class="k">% Incorrect Answers</div><div class="v">{cards["incorrect_answer_pct"]}%</div></div>
    <div class="card"><div class="k">% Weak Context</div><div class="v">{cards["weak_context_pct"]}%</div></div>
  </div>

  <h2>Failure Type Breakdown</h2>
  {_table(failure_summary)}

  <h2>Failure Matrix</h2>
  {_table(matrix)}

  <h2>Failure Breakdown by Category</h2>
  {_table(by_category)}

  <h2>High ROI Fixes (Incorrect Answers)</h2>
  <p>Cases where retrieval hit and grounding passed but end-to-end correctness failed.</p>
  {_table(high_roi.head(10))}

  <h2>Uncertain Cases (Needs Inspection)</h2>
  {_table(uncertain.head(10))}

  <h2>Top Failed Questions (up to 25)</h2>
  {_table(top_failures)}

  <h2>Raw Merge Preview (first 25 rows)</h2>
  {_table_with_row_classes(merged_preview, highlight_row)}
</body>
</html>
"""

    _ensure_parent(settings.failure_dashboard_html)
    with open(settings.failure_dashboard_html, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Saved failure dashboard -> {settings.failure_dashboard_html}")


if __name__ == "__main__":
    main()
