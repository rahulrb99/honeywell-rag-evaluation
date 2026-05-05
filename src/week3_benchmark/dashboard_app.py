from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "eval_outputs"


def _read_json(name: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    path = out_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(name: str, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    path = out_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_cards(
    metrics: dict[str, Any],
    audit: dict[str, Any],
    graph_audit: dict[str, Any],
    cost_summary: dict[str, Any],
) -> None:
    winners = metrics.get("winner_counts", {})
    graph_quality = audit.get("systems", {}).get("graph", {})
    vector_quality = audit.get("systems", {}).get("vector", {})
    cards = st.columns(6)
    cards[0].metric("Rows", metrics.get("rows", ""))
    cards[1].metric("Graph Wins", winners.get("graph", 0))
    cards[2].metric("Vector Wins", winners.get("vector", 0))
    cards[3].metric("Hard Graph Win Rate", metrics.get("hard_query_graph_win_rate", ""))
    cards[4].metric("Graph Errors", graph_quality.get("error_answers", ""))
    cards[5].metric("Graph Nodes", graph_audit.get("node_count", "N/A"))
    cards = st.columns(4)
    cards[0].metric("Graph Avg Contexts", graph_quality.get("avg_context_count", ""))
    cards[1].metric("Vector Avg Contexts", vector_quality.get("avg_context_count", ""))
    cards[2].metric("Graph Relationships", graph_audit.get("relationship_count", "N/A"))
    cards[3].metric("Graph Audit Status", graph_audit.get("status", "missing"))
    cards = st.columns(4)
    cards[0].metric("Estimated Cost", cost_summary.get("total_cost_est_usd", ""))
    cards[1].metric("Estimated Tokens", cost_summary.get("total_input_tokens_est", ""))
    cards[2].metric("Latency Missing", cost_summary.get("latency_missing_rows", ""))
    cards[3].metric("Cost Rates", "set" if cost_summary.get("cost_rates_configured") else "missing")


def _show_graph_summary(graph_audit: dict[str, Any]) -> None:
    if not graph_audit:
        st.info("Graph audit artifacts are not available for this dataset.")
        return

    st.caption(
        f"Domain: {graph_audit.get('domain', 'not recorded')} | "
        f"Generated: {graph_audit.get('generated_at_utc', 'not recorded')} | "
        f"Status: {graph_audit.get('status', 'missing')}"
    )
    cards = st.columns(5)
    cards[0].metric("Nodes", graph_audit.get("node_count", "N/A"))
    cards[1].metric("Relationships", graph_audit.get("relationship_count", "N/A"))
    cards[2].metric("Entity Labels", graph_audit.get("entity_label_count", "N/A"))
    cards[3].metric("Relationship Types", graph_audit.get("relationship_type_count", "N/A"))
    claim_count = graph_audit.get("claim_count")
    cards[4].metric("Claims", claim_count if claim_count is not None else "N/A")
    if graph_audit.get("claim_note"):
        st.info(str(graph_audit["claim_note"]))


def _show_ragas_summary(ragas: dict[str, Any]) -> None:
    if not ragas:
        st.info("RAGAS summary is not available for this dataset.")
        return
    cards = st.columns(4)
    for idx, metric in enumerate(
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ):
        value = ragas.get(metric)
        cards[idx].metric(
            metric.replace("_", " ").title(),
            f"{float(value):.3f}" if isinstance(value, (int, float)) else "N/A",
        )


def _derive_ragas_summary(ragas_comparison: pd.DataFrame) -> dict[str, float]:
    if ragas_comparison.empty:
        return {}
    summary: dict[str, float] = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        graph_col = f"graph_{metric}"
        vector_col = f"vector_{metric}"
        if graph_col in ragas_comparison.columns:
            summary[f"graph_{metric}"] = round(
                float(pd.to_numeric(ragas_comparison[graph_col], errors="coerce").mean()), 4
            )
        if vector_col in ragas_comparison.columns:
            summary[f"vector_{metric}"] = round(
                float(pd.to_numeric(ragas_comparison[vector_col], errors="coerce").mean()), 4
            )
    return summary


def _show_ragas_system_summary(ragas: dict[str, Any], ragas_comparison: pd.DataFrame) -> None:
    derived = _derive_ragas_summary(ragas_comparison)
    if not ragas and not derived:
        st.info("RAGAS summary is not available for this dataset.")
        return
    if derived:
        rows = []
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            rows.append(
                {
                    "metric": metric,
                    "graph_mean": derived.get(f"graph_{metric}", "N/A"),
                    "vector_mean": derived.get(f"vector_{metric}", "N/A"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        _show_ragas_summary(ragas)


def _show_judge_summary(judge: dict[str, Any]) -> None:
    if not judge:
        st.info("LLM judge summary is not available for this dataset.")
        return
    cards = st.columns(4)
    cards[0].metric("Judge Rows", judge.get("rows", "N/A"))
    cards[1].metric("Agreement", judge.get("judge_metric_agreement_rate", "N/A"))
    cards[2].metric("Mean Confidence", judge.get("mean_confidence", "N/A"))
    cards[3].metric("Judge Model", judge.get("judge_model", "N/A"))
    winner_counts = judge.get("winner_counts", {})
    if winner_counts:
        st.dataframe(
            pd.DataFrame(
                [{"judge_winner": key, "count": value} for key, value in winner_counts.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _derive_judge_summary(disagreement: pd.DataFrame) -> dict[str, Any]:
    if disagreement.empty or "llm_judge_winner" not in disagreement.columns:
        return {}
    confidence = (
        pd.to_numeric(disagreement.get("judge_confidence", pd.Series(dtype=float)), errors="coerce")
        if "judge_confidence" in disagreement.columns
        else pd.Series(dtype=float)
    )
    return {
        "rows": int(len(disagreement)),
        "judge_model": "derived from row-level disagreement",
        "winner_counts": disagreement["llm_judge_winner"].fillna("missing").value_counts().to_dict(),
        "mean_confidence": round(float(confidence.mean()), 4) if not confidence.empty else "N/A",
        "judge_metric_agreement_rate": "N/A",
    }


def _show_judge_system_summary(judge: dict[str, Any], disagreement: pd.DataFrame) -> None:
    _show_judge_summary(judge or _derive_judge_summary(disagreement))


def _show_cost_summary(cost_summary: dict[str, Any]) -> None:
    if not cost_summary:
        st.info("Cost and latency summary is not available for this dataset.")
        return
    cards = st.columns(5)
    cards[0].metric("Audit Rows", cost_summary.get("rows", "N/A"))
    cards[1].metric("Total Latency ms", cost_summary.get("total_latency_ms", "N/A"))
    cards[2].metric("Input Tokens est.", cost_summary.get("total_input_tokens_est", "N/A"))
    cards[3].metric("Output Tokens est.", cost_summary.get("total_output_tokens_est", "N/A"))
    cards[4].metric("Cost est. USD", cost_summary.get("total_cost_est_usd", "N/A"))
    stage_rows = cost_summary.get("by_stage_system", [])
    if stage_rows:
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)


def _show_count_table(title: str, values: dict[str, Any], key_name: str) -> None:
    if not values:
        return
    st.write(f"**{title}**")
    st.dataframe(
        pd.DataFrame([{key_name: key, "count": value} for key, value in values.items()]),
        use_container_width=True,
        hide_index=True,
    )


def _show_autoq_summary(autoq_summary: dict[str, Any]) -> None:
    if not autoq_summary:
        return
    cards = st.columns(5)
    cards[0].metric("AutoQ Rows", autoq_summary.get("rows", "N/A"))
    cards[1].metric("Model", autoq_summary.get("model", "N/A"))
    cards[2].metric("Product Records", autoq_summary.get("product_record_count", "N/A"))
    cards[3].metric("Rejected Attempts", autoq_summary.get("rejected_attempts", "N/A"))
    cards[4].metric("Fallback Rows", autoq_summary.get("fallback_rows", "N/A"))
    st.caption(f"Source split: {autoq_summary.get('source_split', 'not recorded')}")
    cols = st.columns(3)
    with cols[0]:
        _show_count_table("Query Classes", autoq_summary.get("query_class_counts", {}), "query_class")
    with cols[1]:
        _show_count_table("Reasoning Types", autoq_summary.get("reasoning_type_counts", {}), "reasoning_type")
    with cols[2]:
        _show_count_table("Source Products", autoq_summary.get("source_product_counts", {}), "source_products")


def _show_disagreement_summary(summary: dict[str, Any]) -> None:
    if not summary:
        st.info("Metric disagreement summary is not available for this dataset.")
        return
    cards = st.columns(5)
    cards[0].metric("Rows", summary.get("rows", "N/A"))
    cards[1].metric("Rows With Disagreement", summary.get("rows_with_disagreement", "N/A"))
    cards[2].metric("Deterministic vs RAGAS", summary.get("deterministic_vs_ragas_conflicts", "N/A"))
    cards[3].metric("Deterministic vs Judge", summary.get("deterministic_vs_judge_conflicts", "N/A"))
    cards[4].metric("RAGAS vs Judge", summary.get("ragas_vs_judge_conflicts", "N/A"))

    cols = st.columns(4)
    with cols[0]:
        _show_count_table(
            "Deterministic Winners",
            summary.get("deterministic_winner_counts", {}),
            "winner",
        )
    with cols[1]:
        _show_count_table("RAGAS Winners", summary.get("ragas_winner_counts", {}), "winner")
    with cols[2]:
        _show_count_table(
            "LLM Judge Winners",
            summary.get("llm_judge_winner_counts", {}),
            "winner",
        )
    with cols[3]:
        _show_count_table(
            "Judge Confidence",
            summary.get("judge_confidence_buckets", {}),
            "bucket",
        )


def _derive_disagreement_summary(disagreement: pd.DataFrame) -> dict[str, Any]:
    if disagreement.empty:
        return {}
    flags = disagreement.get("disagreement_flags", pd.Series(dtype=str)).fillna("none")
    confidence = pd.to_numeric(
        disagreement.get("judge_confidence", pd.Series(dtype=float)), errors="coerce"
    )
    return {
        "rows": int(len(disagreement)),
        "rows_with_disagreement": int(
            disagreement.get("has_disagreement", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "deterministic_vs_ragas_conflicts": int(flags.str.contains("deterministic_vs_ragas").sum()),
        "deterministic_vs_judge_conflicts": int(flags.str.contains("deterministic_vs_judge").sum()),
        "ragas_vs_judge_conflicts": int(flags.str.contains("ragas_vs_judge").sum()),
        "low_judge_confidence_rows": int(flags.str.contains("low_judge_confidence").sum()),
        "judge_confidence_buckets": {
            "high": int((confidence >= 0.8).sum()),
            "medium": int(((confidence >= 0.5) & (confidence < 0.8)).sum()),
            "low": int(((confidence >= 0.0) & (confidence < 0.5)).sum()),
            "missing": int(confidence.isna().sum()),
        },
        "deterministic_winner_counts": disagreement.get(
            "deterministic_winner", pd.Series(dtype=str)
        ).fillna("missing").value_counts().to_dict(),
        "ragas_winner_counts": disagreement.get("ragas_winner", pd.Series(dtype=str))
        .fillna("missing")
        .value_counts()
        .to_dict(),
        "llm_judge_winner_counts": disagreement.get(
            "llm_judge_winner", pd.Series(dtype=str)
        ).fillna("missing").value_counts().to_dict(),
    }


def _show_disagreement_system_summary(
    summary: dict[str, Any], disagreement: pd.DataFrame
) -> None:
    _show_disagreement_summary(summary or _derive_disagreement_summary(disagreement))


def _show_graph_coverage_interpretation(summary: dict[str, Any]) -> None:
    if not summary:
        st.info("Graph coverage interpretation is not available for this dataset.")
        return
    cards = st.columns(4)
    cards[0].metric("Rows", summary.get("rows", "N/A"))
    cards[1].metric("Median Coverage", summary.get("median_graph_coverage_score", "N/A"))
    cards[2].metric("High Coverage Losses", summary.get("high_coverage_graph_losses", "N/A"))
    cards[3].metric("Low Coverage Losses", summary.get("low_coverage_graph_losses", "N/A"))
    st.caption(
        f"High: {summary.get('high_coverage_threshold', 'not recorded')} | "
        f"Low: {summary.get('low_coverage_threshold', 'not recorded')}"
    )
    interpretation = summary.get("interpretation", {})
    if interpretation:
        st.info(
            "High coverage loss: "
            f"{interpretation.get('high_coverage_loss', 'not recorded')}\n\n"
            "Low coverage loss: "
            f"{interpretation.get('low_coverage_loss', 'not recorded')}"
        )
    coverage_by_winner = summary.get("coverage_by_winner", [])
    if coverage_by_winner:
        st.dataframe(pd.DataFrame(coverage_by_winner), use_container_width=True, hide_index=True)


def _derive_graph_coverage_interpretation(
    graph_coverage: pd.DataFrame, graph_coverage_winner: pd.DataFrame
) -> dict[str, Any]:
    if graph_coverage.empty and graph_coverage_winner.empty:
        return {}
    score_col = "graph_coverage_score_mean"
    median_score = (
        round(float(pd.to_numeric(graph_coverage[score_col], errors="coerce").median()), 4)
        if score_col in graph_coverage.columns and not graph_coverage.empty
        else "N/A"
    )
    losses = graph_coverage_winner[
        graph_coverage_winner.get("winner", pd.Series(dtype=str)).astype(str).isin(["vector", "neither"])
    ] if not graph_coverage_winner.empty and "winner" in graph_coverage_winner.columns else pd.DataFrame()
    return {
        "rows": int(graph_coverage["rows"].sum()) if "rows" in graph_coverage.columns else "",
        "median_graph_coverage_score": median_score,
        "high_coverage_threshold": "dataset-specific coverage scores; see table below",
        "low_coverage_threshold": "dataset-specific coverage scores; see table below",
        "high_coverage_graph_losses": int(losses["rows"].sum()) if "rows" in losses.columns else "N/A",
        "low_coverage_graph_losses": "N/A",
        "coverage_by_winner": graph_coverage_winner.to_dict("records")
        if not graph_coverage_winner.empty
        else [],
        "interpretation": {
            "high_coverage_loss": "When graph coverage is high but GraphRAG loses, inspect synthesis and answer generation.",
            "low_coverage_loss": "When graph coverage is low and GraphRAG loses, inspect retrieval and graph coverage.",
        },
    }


def _show_graph_coverage_system_interpretation(
    summary: dict[str, Any],
    graph_coverage: pd.DataFrame,
    graph_coverage_winner: pd.DataFrame,
) -> None:
    _show_graph_coverage_interpretation(
        summary or _derive_graph_coverage_interpretation(graph_coverage, graph_coverage_winner)
    )


def _filter_results(results: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    filtered = results.copy()
    cols = st.columns(3)
    if "query_class" in filtered.columns:
        values = sorted(filtered["query_class"].dropna().astype(str).unique())
        selected = cols[0].selectbox(
            "Query class", ["All", *values], index=0, key=f"{key_prefix}_query_class"
        )
        if selected != "All":
            filtered = filtered[filtered["query_class"].astype(str) == selected]
    if "winner" in filtered.columns:
        values = sorted(filtered["winner"].dropna().astype(str).unique())
        selected = cols[1].selectbox(
            "Winner", ["All", *values], index=0, key=f"{key_prefix}_winner"
        )
        if selected != "All":
            filtered = filtered[filtered["winner"].astype(str) == selected]
    if "failure_mode" in filtered.columns:
        values = sorted(filtered["failure_mode"].dropna().astype(str).unique())
        selected = cols[2].selectbox(
            "Failure mode", ["All", *values], index=0, key=f"{key_prefix}_failure_mode"
        )
        if selected != "All":
            filtered = filtered[filtered["failure_mode"].astype(str) == selected]
    return filtered


def _plot_metric_means(metrics: dict[str, Any]) -> None:
    metric_means = metrics.get("metric_means", {})
    rows = []
    for key, value in metric_means.items():
        if "_" not in key:
            continue
        system, metric = key.split("_", 1)
        rows.append({"system": system, "metric": metric, "mean": value})
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Metric means are not available yet.")
        return
    fig = px.bar(df, x="metric", y="mean", color="system", barmode="group", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def _show_row_drilldown(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("No row-level results available.")
        return
    labels = [
        f"{row['id']} | {row.get('query_class', '')} | {str(row.get('question', ''))[:90]}"
        for _, row in results.iterrows()
    ]
    selected = st.selectbox("Select question", labels, key="row_drilldown_question")
    idx = labels.index(selected)
    row = results.iloc[idx]

    st.write("**Question**")
    st.write(row.get("question", ""))
    st.write("**Ground truth**")
    st.write(row.get("ground_truth", ""))
    cols = st.columns(2)
    cols[0].write("**GraphRAG answer**")
    cols[0].write(row.get("graph_answer", ""))
    cols[1].write("**Vector answer**")
    cols[1].write(row.get("vector_answer", ""))
    score_cols = st.columns(4)
    score_cols[0].metric("Winner", row.get("winner", ""))
    score_cols[1].metric("Failure", row.get("failure_mode", ""))
    score_cols[2].metric("Graph score", row.get("graph_score", ""))
    score_cols[3].metric("Vector score", row.get("vector_score", ""))
    extra_cols = st.columns(4)
    extra_cols[0].metric("RAGAS winner", row.get("ragas_winner", ""))
    extra_cols[1].metric("LLM judge", row.get("llm_judge_winner", ""))
    extra_cols[2].metric("Difficulty", row.get("difficulty_score", ""))
    extra_cols[3].metric("Disagreement", row.get("disagreement_flags", ""))

    with st.expander("Retrieved contexts"):
        for label, col in (("GraphRAG", "graph_retrieved_contexts"), ("Vector", "vector_retrieved_contexts")):
            st.write(f"**{label}**")
            raw = row.get(col, "[]")
            try:
                contexts = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                contexts = [raw]
            for i, context in enumerate(contexts or [], start=1):
                text = context.get("text", "") if isinstance(context, dict) else str(context)
                st.caption(f"Context {i}")
                st.write(text)


def main() -> None:
    st.set_page_config(page_title="GraphEval-Ragas Dashboard", layout="wide")
    st.title("GraphEval-Ragas Interactive Dashboard")
    st.caption("Reads local artifacts from outputs/eval_outputs. RAGAS values reflect the selected run mode: smoke or full.")

    dataset_dirs = {
        path.name: path
        for path in sorted(OUT_DIR.iterdir())
        if path.is_dir() and (path / "eval_results.csv").exists()
    }
    if (OUT_DIR / "eval_results.csv").exists():
        dataset_dirs = {"root": OUT_DIR, **dataset_dirs}
    dataset_names = list(dataset_dirs.keys()) or ["missing"]
    default_dataset = "combined" if "combined" in dataset_names else dataset_names[0]
    selected_dataset = st.sidebar.selectbox(
        "Dataset",
        dataset_names,
        index=dataset_names.index(default_dataset),
    )
    active_dir = dataset_dirs.get(selected_dataset, OUT_DIR)

    results = _read_csv("eval_results.csv", active_dir)
    summary = _read_csv("summary_by_query_class.csv", active_dir)
    failures = _read_csv("failure_analysis.csv", active_dir)
    per_metric = _read_csv("metric_win_rates_overall.csv", active_dir)
    per_metric_class = _read_csv("metric_win_rates_by_class.csv", active_dir)
    review_queue = _read_csv("review_queue.csv", active_dir)
    disagreement = _read_csv("metric_disagreement_analysis.csv", active_dir)
    difficulty = _read_csv("query_difficulty.csv", active_dir)
    difficulty_win = _read_csv("difficulty_vs_win_rate.csv", active_dir)
    cost_audit = _read_csv("cost_latency_audit.csv", active_dir)
    graph_coverage = _read_csv("graph_coverage_summary_by_class.csv", active_dir)
    graph_coverage_winner = _read_csv("graph_coverage_vs_winner.csv", active_dir)
    bootstrap = _read_csv("bootstrap_confidence_intervals.csv", active_dir)
    ragas_comparison = _read_csv("ragas_system_comparison.csv", active_dir)
    autoq_samples = _read_csv("autoq_sample_questions_by_class.csv", active_dir)
    graph_entities = _read_csv("graph_entity_counts.csv", active_dir)
    graph_relationships = _read_csv("graph_relationship_counts.csv", active_dir)
    sample_triples = _read_csv("sample_triples.csv", active_dir)

    if not disagreement.empty:
        merge_keys = (
            ["dataset_name", "id"]
            if "dataset_name" in results.columns and "dataset_name" in disagreement.columns
            else ["id"]
        )
        disagreement_cols = [
            col
            for col in [
                *merge_keys,
                "ragas_winner",
                "llm_judge_winner",
                "disagreement_flags",
            ]
            if col in disagreement.columns
        ]
        results = results.merge(
            disagreement[disagreement_cols],
            on=merge_keys,
            how="left",
        )
    if not difficulty.empty:
        merge_keys = (
            ["dataset_name", "id"]
            if "dataset_name" in results.columns and "dataset_name" in difficulty.columns
            else ["id"]
        )
        difficulty_cols = [
            col
            for col in [*merge_keys, "difficulty_score", "difficulty_band"]
            if col in difficulty.columns
        ]
        results = results.merge(
            difficulty[difficulty_cols],
            on=merge_keys,
            how="left",
        )

    metrics = _read_json("metric_summary.json", active_dir)
    audit = _read_json("output_quality_audit.json", active_dir)
    graph_audit = _read_json("graph_extraction_summary.json", active_dir)
    autoq_summary = _read_json("autoq_summary.json", active_dir)
    ragas = _read_json("ragas_graphrag_scores.json", active_dir) or _read_json(
        "ragas_smoke_graphrag_5_scores.json", active_dir
    )
    judge = _read_json("pairwise_llm_judge_summary.json", active_dir)
    cost_summary = _read_json("cost_latency_summary.json", active_dir)
    disagreement_summary = _read_json("metric_disagreement_summary.json", active_dir)
    graph_coverage_interpretation = _read_json("graph_coverage_interpretation.json", active_dir)

    if results.empty or not metrics:
        st.warning("Core evaluation artifacts are missing. Run python -m src.run_week3_pipeline first.")
        return

    tab_overview, tab_dataset, tab_metrics, tab_depth, tab_failures, tab_rows, tab_graph = st.tabs(
        ["Overview", "Dataset", "Metrics", "Depth", "Failures", "Row Drilldown", "Graph Audit"]
    )

    with tab_overview:
        _metric_cards(metrics, audit, graph_audit, cost_summary)
        if cost_summary.get("cost_warning"):
            st.warning(cost_summary["cost_warning"])
        if int(cost_summary.get("latency_missing_rows", 0) or 0) > 0:
            st.info("Latency not supplied for some prediction rows.")
        st.subheader("Winner Distribution")
        winner_counts = metrics.get("winner_counts", {})
        if winner_counts:
            fig = px.pie(
                pd.DataFrame(
                    [{"winner": key, "count": value} for key, value in winner_counts.items()]
                ),
                names="winner",
                values="count",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("RAGAS Summary")
        _show_ragas_system_summary(ragas, ragas_comparison)
        st.subheader("LLM Judge Summary")
        _show_judge_system_summary(judge, disagreement)
        st.subheader("Cost and Latency")
        _show_cost_summary(cost_summary)
        st.subheader("Metric Tiers")
        st.dataframe(
            pd.DataFrame(
                [
                    {"tier": "Tier 1", "name": "Deterministic reproducible metrics", "purpose": f"Full-set scoring and bootstrap intervals, n={len(results)}"},
                    {"tier": "Tier 2", "name": "RAGAS LLM metrics", "purpose": "LLM-based faithfulness/relevancy/context scoring when enabled"},
                    {"tier": "Tier 3", "name": "Pairwise LLM judge", "purpose": "Pairwise preference and confidence when enabled"},
                    {"tier": "Tier 4", "name": "Derived graph coverage proxy metrics", "purpose": "Structural coverage over GraphRAG retrieved contexts"},
                    {"tier": "Tier 5", "name": "Operational cost/latency estimates", "purpose": "Runtime and token-cost estimates"},
                ]
            ),
            use_container_width=True,
        )

    with tab_dataset:
        st.subheader("Query Class Summary")
        st.dataframe(summary, use_container_width=True)
        if autoq_summary:
            st.subheader("AutoQ Summary")
            _show_autoq_summary(autoq_summary)
        if not autoq_samples.empty:
            st.subheader("AutoQ Sample Questions")
            st.dataframe(autoq_samples, use_container_width=True)

    with tab_metrics:
        st.subheader("Metric Means")
        _plot_metric_means(metrics)
        st.subheader("Per-Metric Win Rates")
        st.dataframe(per_metric, use_container_width=True)
        if not per_metric_class.empty:
            heat = per_metric_class.pivot(
                index="query_class", columns="metric", values="graph_win_rate"
            )
            fig = px.imshow(heat, color_continuous_scale="RdYlGn", zmin=0, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("RAGAS GraphRAG vs Vector")
        if ragas_comparison.empty:
            st.warning(
                f"No RAGAS comparison rows found for `{selected_dataset}`. "
                "Choose `manual_28`, `autoq_35`, or `combined`, or rerun with `--ragas-mode full`."
            )
        else:
            winner_counts = ragas_comparison["ragas_winner"].value_counts().reset_index()
            winner_counts.columns = ["ragas_winner", "count"]
            fig = px.bar(winner_counts, x="ragas_winner", y="count")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ragas_comparison, use_container_width=True, hide_index=True)

    with tab_depth:
        st.subheader("Metric Disagreement")
        _show_disagreement_system_summary(disagreement_summary, disagreement)
        st.dataframe(disagreement, use_container_width=True)
        st.subheader("Difficulty vs Win Rate")
        if not difficulty_win.empty:
            fig = px.bar(difficulty_win, x="difficulty_band", y=["graph_win_rate", "vector_win_rate"], barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(difficulty_win, use_container_width=True)
        st.subheader("Cost/Latency Audit")
        st.dataframe(cost_audit, use_container_width=True)
        st.subheader("Bootstrap Confidence Intervals")
        st.dataframe(bootstrap, use_container_width=True)
        st.subheader("Graph Coverage by Query Class")
        _show_graph_coverage_system_interpretation(
            graph_coverage_interpretation, graph_coverage, graph_coverage_winner
        )
        st.dataframe(graph_coverage, use_container_width=True)
        st.subheader("Graph Coverage by Winner")
        st.dataframe(graph_coverage_winner, use_container_width=True)

    with tab_failures:
        st.subheader("Failure Modes")
        filtered = _filter_results(results, f"{selected_dataset}_failures")
        failure_source = filtered if "failure_mode" in filtered.columns else failures
        if not failure_source.empty and "failure_mode" in failure_source.columns:
            failure_counts = failure_source["failure_mode"].value_counts().reset_index()
            failure_counts.columns = ["failure_mode", "count"]
            fig = px.bar(failure_counts, x="failure_mode", y="count")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(failure_counts, use_container_width=True, hide_index=True)
        else:
            st.info("Failure mode rows are not available for this dataset.")
        st.subheader("Review Queue")
        st.dataframe(review_queue if not review_queue.empty else filtered, use_container_width=True)

    with tab_rows:
        _show_row_drilldown(_filter_results(results, f"{selected_dataset}_rows"))

    with tab_graph:
        st.subheader("Graph Extraction Summary")
        if selected_dataset == "combined" and not graph_audit:
            st.info(
                "Graph audit is captured per dataset. Select `manual_28` or `autoq_35` "
                "to inspect Neo4j node, relationship, label, and sample-triple details."
            )
        else:
            _show_graph_summary(graph_audit)
        cols = st.columns(2)
        cols[0].dataframe(graph_entities, use_container_width=True)
        cols[1].dataframe(graph_relationships, use_container_width=True)
        st.subheader("Sample Triples")
        st.dataframe(sample_triples, use_container_width=True)


if __name__ == "__main__":
    main()
