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
DEFAULT_OUTPUT_AUDIT = ROOT / "outputs" / "eval_outputs" / "output_quality_audit.json"
DEFAULT_METRIC_WIN_RATES = ROOT / "outputs" / "eval_outputs" / "metric_win_rates_overall.csv"
DEFAULT_REVIEW_QUEUE = ROOT / "outputs" / "eval_outputs" / "review_queue.csv"
DEFAULT_GRAPH_AUDIT = ROOT / "outputs" / "eval_outputs" / "graph_extraction_summary.json"
DEFAULT_RAGAS_COMPARISON = ROOT / "outputs" / "eval_outputs" / "ragas_system_comparison.csv"
DEFAULT_DISAGREEMENT = ROOT / "outputs" / "eval_outputs" / "metric_disagreement_analysis.csv"
DEFAULT_DIFFICULTY = ROOT / "outputs" / "eval_outputs" / "difficulty_vs_win_rate.csv"
DEFAULT_COST = ROOT / "outputs" / "eval_outputs" / "cost_latency_summary.json"
DEFAULT_GRAPH_COVERAGE = ROOT / "outputs" / "eval_outputs" / "graph_coverage_summary_by_class.csv"
DEFAULT_BOOTSTRAP = ROOT / "outputs" / "eval_outputs" / "bootstrap_confidence_intervals.csv"
DEFAULT_DISAGREEMENT_SUMMARY = ROOT / "outputs" / "eval_outputs" / "metric_disagreement_summary.json"
DEFAULT_GRAPH_COVERAGE_INTERPRETATION = ROOT / "outputs" / "eval_outputs" / "graph_coverage_interpretation.json"
DEFAULT_OUT = ROOT / "outputs" / "eval_outputs" / "dashboard.html"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static Week 3 HTML dashboard.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--kappa", default=str(DEFAULT_KAPPA))
    parser.add_argument("--ragas", default=str(DEFAULT_RAGAS))
    parser.add_argument("--llm-judge-summary", default=str(DEFAULT_LLM_JUDGE))
    parser.add_argument("--output-audit", default=str(DEFAULT_OUTPUT_AUDIT))
    parser.add_argument("--metric-win-rates", default=str(DEFAULT_METRIC_WIN_RATES))
    parser.add_argument("--review-queue", default=str(DEFAULT_REVIEW_QUEUE))
    parser.add_argument("--graph-audit", default=str(DEFAULT_GRAPH_AUDIT))
    parser.add_argument("--ragas-comparison", default=str(DEFAULT_RAGAS_COMPARISON))
    parser.add_argument("--disagreement", default=str(DEFAULT_DISAGREEMENT))
    parser.add_argument("--difficulty", default=str(DEFAULT_DIFFICULTY))
    parser.add_argument("--cost-latency", default=str(DEFAULT_COST))
    parser.add_argument("--graph-coverage", default=str(DEFAULT_GRAPH_COVERAGE))
    parser.add_argument("--bootstrap", default=str(DEFAULT_BOOTSTRAP))
    parser.add_argument("--disagreement-summary", default=str(DEFAULT_DISAGREEMENT_SUMMARY))
    parser.add_argument("--graph-coverage-interpretation", default=str(DEFAULT_GRAPH_COVERAGE_INTERPRETATION))
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
    if df.empty:
        return "<p class='note'>No rows available.</p>"
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
    output_audit = _read_json(Path(args.output_audit))
    graph_audit = _read_json(Path(args.graph_audit))
    metric_win_rates = (
        pd.read_csv(args.metric_win_rates) if Path(args.metric_win_rates).exists() else pd.DataFrame()
    )
    review_queue = (
        pd.read_csv(args.review_queue) if Path(args.review_queue).exists() else pd.DataFrame()
    )
    ragas_comparison = (
        pd.read_csv(args.ragas_comparison) if Path(args.ragas_comparison).exists() else pd.DataFrame()
    )
    disagreement = (
        pd.read_csv(args.disagreement) if Path(args.disagreement).exists() else pd.DataFrame()
    )
    difficulty = pd.read_csv(args.difficulty) if Path(args.difficulty).exists() else pd.DataFrame()
    graph_coverage = (
        pd.read_csv(args.graph_coverage) if Path(args.graph_coverage).exists() else pd.DataFrame()
    )
    bootstrap = pd.read_csv(args.bootstrap) if Path(args.bootstrap).exists() else pd.DataFrame()
    cost_latency = _read_json(Path(args.cost_latency))
    disagreement_summary = _read_json(Path(args.disagreement_summary))
    graph_coverage_interpretation = _read_json(Path(args.graph_coverage_interpretation))
    output = Path(args.output)

    winner_counts = metrics.get("winner_counts", {})
    hard_counts = metrics.get("hard_query_winner_counts", {})
    metric_means = metrics.get("metric_means", {})
    failure_counts = metrics.get("failure_mode_counts", {})
    llm_judge_counts = llm_judge.get("winner_counts", {})
    graph_quality = output_audit.get("systems", {}).get("graph", {})
    vector_quality = output_audit.get("systems", {}).get("vector", {})
    disagreement_count = (
        int(disagreement["has_disagreement"].sum())
        if "has_disagreement" in disagreement.columns
        else ""
    )
    latency_note = (
        "Latency not supplied for some prediction rows."
        if int(cost_latency.get("latency_missing_rows", 0) or 0) > 0
        else ""
    )
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
    .warn {{ color: #b42318; font-weight: 700; }}
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
    {_card("RAGAS Faithfulness", ragas.get("faithfulness", ""), "GraphRAG RAGAS; smoke or full per run mode")}
    {_card("RAGAS Context Precision", ragas.get("context_precision", ""), "GraphRAG RAGAS; smoke or full per run mode")}
    {_card("Kappa: Human vs Week 3", kappa.get("human_vs_week3_metric", {}).get("cohens_kappa", ""), "agreement check")}
    {_card("LLM Judge Graph Wins", llm_judge_counts.get("graph", ""), "pairwise judge")}
    {_card("Judge/Metric Agreement", llm_judge.get("judge_metric_agreement_rate", ""), "pairwise judge")}
    {_card("Graph Error Answers", graph_quality.get("error_answers", ""), "output audit")}
    {_card("Vector Error Answers", vector_quality.get("error_answers", ""), "output audit")}
    {_card("Graph Nodes", graph_audit.get("node_count", ""), "Neo4j audit")}
    {_card("Graph Relationships", graph_audit.get("relationship_count", ""), "Neo4j audit")}
    {_card("Disagreement Rows", disagreement_count, "metric/RAGAS/judge conflicts")}
    {_card("Estimated Cost", cost_latency.get("total_cost_est_usd", ""), "configured rate estimate")}
    {_card("Latency Missing Rows", cost_latency.get("latency_missing_rows", ""), "N/A when prediction CSV omits latency_ms")}
  </div>
  <p class="note">{html.escape(cost_latency.get("cost_warning", ""))}</p>
  <p class="note">{html.escape(latency_note)}</p>

  <h2>Metric Tiers</h2>
  <table><thead><tr><th>Tier</th><th>Name</th><th>Purpose</th></tr></thead><tbody>
    <tr><td>Tier 1</td><td>Deterministic reproducible metrics</td><td>Full-set scoring and bootstrap intervals, n={len(results)}</td></tr>
    <tr><td>Tier 2</td><td>RAGAS LLM metrics</td><td>LLM-based faithfulness/relevancy/context scoring when enabled</td></tr>
    <tr><td>Tier 3</td><td>Pairwise LLM judge</td><td>Pairwise preference and confidence when enabled</td></tr>
    <tr><td>Tier 4</td><td>Derived graph coverage proxy metrics</td><td>Structural coverage over GraphRAG retrieved contexts</td></tr>
    <tr><td>Tier 5</td><td>Operational cost/latency estimates</td><td>Runtime and token-cost estimates</td></tr>
  </tbody></table>

  <h2>Output Quality Audit</h2>
  <div class="grid">
    {_card("Graph Empty Answers", graph_quality.get("empty_answers", ""))}
    {_card("Graph Not Found", graph_quality.get("not_found_answers", ""))}
    {_card("Graph Avg Contexts", graph_quality.get("avg_context_count", ""))}
    {_card("Vector Empty Answers", vector_quality.get("empty_answers", ""))}
    {_card("Vector Not Found", vector_quality.get("not_found_answers", ""))}
    {_card("Vector Avg Contexts", vector_quality.get("avg_context_count", ""))}
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

  <h2>Per-Metric Win Rates</h2>
  {_table(metric_win_rates, ["metric", "rows", "graph_wins", "vector_wins", "ties", "graph_win_rate", "vector_win_rate", "mean_delta"])}

  <h2>RAGAS System Comparison</h2>
  {_table(ragas_comparison.head(15), ["id", "query_class", "ragas_winner", "faithfulness_winner", "answer_relevancy_winner", "context_precision_winner", "context_recall_winner"])}

  <h2>Metric Disagreement</h2>
  <div class="grid">
    {_card("Rows With Disagreement", disagreement_summary.get("rows_with_disagreement", ""))}
    {_card("Deterministic vs RAGAS", disagreement_summary.get("deterministic_vs_ragas_conflicts", ""))}
    {_card("Deterministic vs Judge", disagreement_summary.get("deterministic_vs_judge_conflicts", ""))}
    {_card("RAGAS vs Judge", disagreement_summary.get("ragas_vs_judge_conflicts", ""))}
    {_card("Low-Confidence Judge Rows", disagreement_summary.get("low_judge_confidence_rows", ""))}
  </div>
  {_table(disagreement.head(15), ["id", "query_class", "deterministic_winner", "ragas_winner", "llm_judge_winner", "judge_confidence", "disagreement_flags"])}

  <h2>Difficulty vs Win Rate</h2>
  {_table(difficulty, ["difficulty_band", "rows", "mean_difficulty", "graph_win_rate", "vector_win_rate"])}

  <h2>Graph Coverage Metrics</h2>
  <div class="grid">
    {_card("Median Graph Coverage", graph_coverage_interpretation.get("median_graph_coverage_score", ""))}
    {_card("High-Coverage Graph Losses", graph_coverage_interpretation.get("high_coverage_graph_losses", ""))}
    {_card("Low-Coverage Graph Losses", graph_coverage_interpretation.get("low_coverage_graph_losses", ""))}
  </div>
  {_table(graph_coverage, ["query_class", "rows", "graph_coverage_score_mean", "graph_expected_entity_coverage_mean", "relation_signal_coverage_mean", "structural_density_mean"])}

  <h2>Bootstrap Confidence Intervals</h2>
  {_table(bootstrap, ["metric", "scope", "n", "point_estimate", "ci_lower", "ci_upper", "status"])}

  <h2>Cost and Latency Estimate</h2>
  <pre>{html.escape(json.dumps(cost_latency, indent=2))}</pre>

  <h2>Failure Modes</h2>
  {_table(failure_df, ["failure_mode", "count"])}

  <h2>Review Queue Preview</h2>
  {_table(review_queue.head(15), ["id", "query_class", "winner", "review_reasons", "failure_mode", "question"])}

  <h2>Graph Audit</h2>
  <div class="grid">
    {_card("Domain", graph_audit.get("domain", ""))}
    {_card("Audit Status", graph_audit.get("status", ""))}
    {_card("Entity Label Types", graph_audit.get("entity_label_count", ""))}
    {_card("Relationship Types", graph_audit.get("relationship_type_count", ""))}
  </div>

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
