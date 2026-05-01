from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd


def test_dashboard_builder_renders_static_html() -> None:
    tmp_path = Path(__file__).resolve().parents[1] / "outputs" / "_pytest_dashboard"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    results = tmp_path / "results.csv"
    summary = tmp_path / "summary.csv"
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "dashboard.html"

    pd.DataFrame(
        [
            {
                "id": "q1",
                "query_class": "single_hop_fact",
                "winner": "graph",
                "failure_mode": "retrieval_missed_entities",
                "graph_score": 0.9,
                "vector_score": 0.2,
                "question": "What does the DAA2 amplifier support?",
            }
        ]
    ).to_csv(results, index=False)
    pd.DataFrame(
        [
            {
                "query_class": "single_hop_fact",
                "rows": 1,
                "graph_wins": 1,
                "vector_wins": 0,
                "ties": 0,
                "graph_win_rate": 1.0,
                "graph_entity_recall_mean": 1.0,
                "graph_context_precision_mean": 1.0,
            }
        ]
    ).to_csv(summary, index=False)
    metrics.write_text(
        json.dumps(
            {
                "rows": 1,
                "winner_counts": {"graph": 1},
                "hard_query_winner_counts": {},
                "hard_query_graph_win_rate": 0.0,
                "metric_means": {"graph_entity_recall": 1.0},
                "failure_mode_counts": {"retrieval_missed_entities": 1},
            }
        ),
        encoding="utf-8",
    )

    old_argv = sys.argv[:]
    sys.argv = [
        "build_week3_dashboard",
        "--results",
        str(results),
        "--summary",
        str(summary),
        "--metrics",
        str(metrics),
        "--kappa",
        str(tmp_path / "missing_kappa.json"),
        "--ragas",
        str(tmp_path / "missing_ragas.json"),
        "--output",
        str(output),
    ]
    try:
        runpy.run_module("src.week3_benchmark.build_week3_dashboard", run_name="__main__")
    finally:
        sys.argv = old_argv

    html = output.read_text(encoding="utf-8")
    assert "GraphEval-Ragas Week 3 Dashboard" in html
    assert "Rows Evaluated" in html
    assert "retrieval_missed_entities" in html

    shutil.rmtree(tmp_path)
