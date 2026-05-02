from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs" / "eval_outputs" / "eval_results.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "eval_outputs" / "summary_by_query_class.csv"
DEFAULT_METRICS = ROOT / "outputs" / "eval_outputs" / "metric_summary.json"
DEFAULT_KAPPA = ROOT / "outputs" / "eval_outputs" / "kappa_summary.json"
DEFAULT_RAGAS = ROOT / "outputs" / "eval_outputs" / "ragas_smoke_graphrag_5_scores.json"
DEFAULT_LLM_JUDGE = ROOT / "outputs" / "eval_outputs" / "pairwise_llm_judge_summary.json"
DEFAULT_OUT = ROOT / "outputs" / "eval_outputs" / "dashboard.html"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static Week 3 HTML dashboard.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--kappa", default=str(DEFAULT_KAPPA))
    parser.add_argument("--ragas", default=str(DEFAULT_RAGAS))
    parser.add_argument("--llm-judge-summary", default=str(DEFAULT_LLM_JUDGE))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _card(label: str, value: object, note: str = "") -> str:
    return (
        "<div class='card'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{_fmt(value)}</div>"
        f"<div class='note'>{html.escape(note)}</div>"
        "</div>"
    )


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    cols = [col for col in columns if col in df.columns]
    head = "".join(f"<th>{html.escape(col)}</th>" for col in cols)
    rows = []
    for _, row in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{_fmt(row[col])}</td>" for col in cols) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _bar_segments(counts: dict[str, int]) -> str:
    total = max(1, sum(int(v) for v in counts.values()))
    labels = [("graph", "GraphRAG"), ("vector", "Vector"), ("tie", "Tie"), ("neither", "Neither")]
    parts = []
    for key, label in labels:
        count = int(counts.get(key, 0))
        width = 100 * count / total
        parts.append(
            f"<div class='seg {key}' style='width:{width:.2f}%' title='{label}: {count}'>{count if width > 7 else ''}</div>"
        )
    return "<div class='bar'>" + "".join(parts) + "</div>"


def main() -> None:
    args = _parse_args()
    results = pd.read_csv(args.results)
    summary = pd.read_csv(args.summary)
    metrics = _read_json(Path(args.metrics))
    kappa = _read_json(Path(args.kappa))
    ragas = _read_json(Path(args.ragas))
    llm_judge = _read_json(Path(args.llm_judge_summary))
    output = Path(args.output)

    winner_counts = metrics.get("winner_counts", {})
    hard_counts = metrics.get("hard_query_winner_counts", {})
    metric_means = metrics.get("metric_means", {})
    failure_counts = metrics.get("failure_mode_counts", {})
    llm_judge_counts = llm_judge.get("winner_counts", {})
    failure_df = pd.DataFrame(
        [{"failure_mode": key, "count": value} for key, value in failure_counts.items()]
    ).sort_values("count", ascending=False)

    examples = results.sort_values(["winner", "query_class", "graph_score"], ascending=[True, True, False]).head(10)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GraphEval-Ragas Week 3 Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #17202a; }}
    header {{ background: #0f172a; color: white; padding: 24px 32px; }}
    main {{ padding: 24px 32px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 32px; font-size: 20px; }}
    .sub {{ color: #cbd5e1; margin: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 16px; }}
    .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #526070; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .note {{ font-size: 12px; color: #667085; margin-top: 6px; min-height: 16px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee7; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #edf0f5; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; font-size: 12px; text-transform: uppercase; color: #475467; }}
    td {{ font-size: 14px; }}
    .bar {{ display: flex; height: 34px; border-radius: 6px; overflow: hidden; border: 1px solid #d9dee7; background: white; }}
    .seg {{ color: white; font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; }}
    .graph {{ background: #2563eb; }}
    .vector {{ background: #dc6803; }}
    .tie {{ background: #667085; }}
    .neither {{ background: #b42318; }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; color: #475467; font-size: 13px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
    .section {{ margin-top: 18px; }}
  </style>
</head>
<body>
<header>
  <h1>GraphEval-Ragas Week 3 Dashboard</h1>
  <p class="sub">GraphRAG vs top-k vector RAG on the 28-question Honeywell hard benchmark</p>
</header>
<main>
  <div class="grid">
    {_card("Rows Evaluated", metrics.get("rows", len(results)))}
    {_card("GraphRAG Wins", winner_counts.get("graph", 0), "overall")}
    {_card("Vector Wins", winner_counts.get("vector", 0), "overall")}
    {_card("Ties", winner_counts.get("tie", 0), "overall")}
    {_card("Hard-Query Graph Win Rate", metrics.get("hard_query_graph_win_rate", ""), "comparison/multi-hop/theme/relationship")}
    {_card("RAGAS Faithfulness", ragas.get("faithfulness", ""), "5-row smoke")}
    {_card("RAGAS Context Precision", ragas.get("context_precision", ""), "5-row smoke")}
    {_card("Kappa: Human vs Week 3", kappa.get("human_vs_week3_metric", {}).get("cohens_kappa", ""), "agreement check")}
    {_card("LLM Judge Graph Wins", llm_judge_counts.get("graph", ""), "pairwise judge")}
    {_card("Judge/Metric Agreement", llm_judge.get("judge_metric_agreement_rate", ""), "pairwise judge")}
  </div>

  <h2>Winner Distribution</h2>
  {_bar_segments(winner_counts)}
  <div class="legend">
    <span><span class="dot graph"></span>GraphRAG</span>
    <span><span class="dot vector"></span>Vector</span>
    <span><span class="dot tie"></span>Tie</span>
    <span><span class="dot neither"></span>Neither</span>
  </div>

  <h2>Hard-Query Winner Distribution</h2>
  {_bar_segments(hard_counts)}

  <h2>Metric Means</h2>
  <div class="grid">
    {_card("Graph Entity Recall", metric_means.get("graph_entity_recall", ""))}
    {_card("Vector Entity Recall", metric_means.get("vector_entity_recall", ""))}
    {_card("Graph Context Precision", metric_means.get("graph_context_precision", ""))}
    {_card("Vector Context Precision", metric_means.get("vector_context_precision", ""))}
    {_card("Graph Faithfulness", metric_means.get("graph_faithfulness", ""))}
    {_card("Vector Faithfulness", metric_means.get("vector_faithfulness", ""))}
    {_card("Graph Answer Relevancy", metric_means.get("graph_answer_relevancy", ""))}
    {_card("Vector Answer Relevancy", metric_means.get("vector_answer_relevancy", ""))}
  </div>

  <h2>Results by Query Class</h2>
  {_table(summary, ["query_class", "rows", "graph_wins", "vector_wins", "ties", "graph_win_rate", "graph_entity_recall_mean", "graph_context_precision_mean"])}

  <h2>Failure Modes</h2>
  {_table(failure_df, ["failure_mode", "count"])}

  <h2>Representative Rows</h2>
  {_table(examples, ["id", "query_class", "winner", "failure_mode", "graph_score", "vector_score", "question"])}
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(f"Saved dashboard -> {output}")


if __name__ == "__main__":
    main()
