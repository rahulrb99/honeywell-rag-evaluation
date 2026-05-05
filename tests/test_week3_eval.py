from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.week3_benchmark import run_week3_eval as eval_mod
from src.week3_benchmark.analyze_metric_disagreement import build_metric_disagreement
from src.week3_benchmark.analyze_metric_disagreement import summarize_disagreement
from src import run_week3_pipeline
from src.week3_benchmark.audit_cost_latency import (
    build_cost_latency_audit,
    estimate_cost,
    estimate_tokens,
)
from src.week3_benchmark.bootstrap_confidence_intervals import build_bootstrap_intervals
from src.week3_benchmark.build_review_queue import build_review_queue
from src.week3_benchmark.combine_dataset_outputs import combine_outputs
from src.week3_benchmark.compare_metric_win_rates import build_per_metric_comparison
from src.week3_benchmark.graph_coverage_metrics import interpret_graph_coverage, score_graph_coverage_row
from src.week3_benchmark.metrics_metadata import METRICS_VERSION
from src.week3_benchmark.run_manifest import build_manifest
from src.week3_benchmark.run_full_ragas_comparison import build_ragas_comparison
from src.week3_benchmark.run_output_quality_audit import _audit_system
from src.week3_benchmark.score_query_difficulty import build_query_difficulty, difficulty_score


def test_metric_helpers_score_supported_and_missing_answers() -> None:
    contexts = ["The DAA2 amplifier supports digital audio evacuation."]

    assert eval_mod._entity_recall(["DAA2 amplifier"], contexts) == (1.0, ["DAA2 amplifier"])
    assert eval_mod._entity_recall(["remote microphone"], contexts) == (0.0, [])
    assert eval_mod._faithfulness("The DAA2 amplifier supports evacuation audio.", contexts) > 0
    assert eval_mod._faithfulness("Not found in provided context.", contexts) == 0.0
    assert (
        eval_mod._answer_relevancy(
            "What does the DAA2 amplifier support?",
            "The DAA2 amplifier supports evacuation audio.",
            "It supports evacuation audio.",
        )
        > 0
    )


def test_failure_mode_prioritizes_retrieval_misses() -> None:
    row = pd.Series(
        {
            "winner": "graph",
            "graph_score": 0.8,
            "vector_score": 0.2,
            "vector_entity_recall": 0.0,
            "vector_context_precision": 1.0,
            "vector_faithfulness": 1.0,
            "vector_answer_relevancy": 1.0,
        }
    )

    assert eval_mod._failure_mode(row) == "retrieval_missed_entities"


def test_build_results_compares_graph_and_vector_rows() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "What does the DAA2 amplifier support?",
                "ground_truth": "The DAA2 amplifier supports evacuation audio.",
                "contexts": json.dumps(["The DAA2 amplifier supports evacuation audio."]),
                "query_class": "single_hop_fact",
                "source_product": "DAA2 amplifier",
                "source_fields": json.dumps(["support"]),
            }
        ]
    )
    graph = pd.DataFrame(
        [
            {
                "id": "q1",
                "answer": "The DAA2 amplifier supports evacuation audio.",
                "retrieved_contexts": json.dumps(
                    [{"text": "The DAA2 amplifier supports evacuation audio."}]
                ),
            }
        ]
    )
    vector = pd.DataFrame(
        [
            {
                "id": "q1",
                "answer": "Not found in provided context.",
                "retrieved_contexts": json.dumps([{"text": "Unrelated speaker context."}]),
            }
        ]
    )

    results = eval_mod._build_results(benchmark, graph, vector, pd.DataFrame())

    assert len(results) == 1
    assert results.loc[0, "winner"] == "graph"
    assert results.loc[0, "graph_entity_recall"] == 1.0
    assert results.loc[0, "vector_entity_recall"] == 0.0


def test_validate_columns_reports_missing_names() -> None:
    df = pd.DataFrame([{"id": "q1", "answer": "x"}])

    with pytest.raises(ValueError, match="retrieved_contexts"):
        eval_mod._validate_columns(df, eval_mod.REQUIRED_PREDICTION_COLUMNS, "prediction CSV")


def test_validate_inputs_reports_duplicate_ids() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "Question 1?",
                "ground_truth": "Answer",
                "contexts": json.dumps(["context"]),
                "query_class": "single_hop_fact",
            },
            {
                "id": "q1",
                "question": "Question 1 duplicate?",
                "ground_truth": "Answer",
                "contexts": json.dumps(["context"]),
                "query_class": "single_hop_fact",
            },
        ]
    )
    predictions = pd.DataFrame(
        [{"id": "q1", "answer": "Answer", "retrieved_contexts": json.dumps(["context"])}]
    )

    with pytest.raises(ValueError, match="duplicate id values: q1"):
        eval_mod._validate_inputs(benchmark, predictions, predictions)


def test_validate_inputs_reports_malformed_json_array() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "Question?",
                "ground_truth": "Answer",
                "contexts": "not-json",
                "query_class": "single_hop_fact",
            }
        ]
    )
    predictions = pd.DataFrame(
        [{"id": "q1", "answer": "Answer", "retrieved_contexts": json.dumps(["context"])}]
    )

    with pytest.raises(ValueError, match="contexts.*JSON arrays.*q1"):
        eval_mod._validate_inputs(benchmark, predictions, predictions)


def test_validate_inputs_reports_prediction_id_mismatch() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "Question?",
                "ground_truth": "Answer",
                "contexts": json.dumps(["context"]),
                "query_class": "single_hop_fact",
            }
        ]
    )
    graph = pd.DataFrame(
        [{"id": "q2", "answer": "Answer", "retrieved_contexts": json.dumps(["context"])}]
    )
    vector = pd.DataFrame(
        [{"id": "q1", "answer": "Answer", "retrieved_contexts": json.dumps(["context"])}]
    )

    with pytest.raises(ValueError, match="missing predictions for dataset ids: q1"):
        eval_mod._validate_inputs(benchmark, graph, vector)


def test_output_quality_audit_flags_error_answers_and_empty_contexts() -> None:
    benchmark = pd.DataFrame([{"id": "q1", "query_class": "multi_hop"}])
    predictions = pd.DataFrame(
        [
            {
                "id": "q1",
                "answer": "Not found in provided context. (Error code: 429 rate limit)",
                "retrieved_contexts": json.dumps([]),
            }
        ]
    )

    rows = _audit_system(benchmark, predictions, "graph")

    assert rows[0]["answer_error"] is True
    assert rows[0]["answer_not_found"] is True
    assert rows[0]["context_empty"] is True
    assert "answer_error" in rows[0]["quality_flags"]


def test_per_metric_comparison_classifies_graph_vector_and_tie() -> None:
    results = pd.DataFrame(
        [
            {
                "id": "q1",
                "query_class": "comparison",
                "winner": "graph",
                "failure_mode": "none",
                "graph_entity_recall": 0.9,
                "vector_entity_recall": 0.2,
                "graph_context_precision": 0.5,
                "vector_context_precision": 0.52,
                "graph_faithfulness": 0.1,
                "vector_faithfulness": 0.8,
                "graph_answer_relevancy": 0.7,
                "vector_answer_relevancy": 0.7,
                "graph_answer_correctness": 0.6,
                "vector_answer_correctness": 0.4,
            }
        ]
    )

    comparison = build_per_metric_comparison(results)
    winners = dict(zip(comparison["metric"], comparison["winner"], strict=False))

    assert winners["entity_recall"] == "graph"
    assert winners["context_precision"] == "tie"
    assert winners["faithfulness"] == "vector"
    assert winners["answer_relevancy"] == "tie"


def test_review_queue_includes_vector_hard_query_win() -> None:
    results = pd.DataFrame(
        [
            {
                "id": "q1",
                "query_class": "multi_hop",
                "winner": "vector",
                "failure_mode": "retrieval_missed_entities",
                "graph_score": 0.1,
                "vector_score": 0.8,
                "graph_entity_recall": 0.0,
                "graph_answer_correctness": 0.0,
                "question": "Question?",
                "ground_truth": "Truth",
                "graph_answer": "Bad",
                "vector_answer": "Good",
            }
        ]
    )

    queue = build_review_queue(results, {})

    assert len(queue) == 1
    assert "vector_won_hard_query" in queue.loc[0, "review_reasons"]


def test_ragas_comparison_classifies_system_winner() -> None:
    graph_rows = pd.DataFrame(
        [{"id": "q1", "query_class": "multi_hop", "faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.8}]
    )
    vector_rows = pd.DataFrame(
        [{"id": "q1", "query_class": "multi_hop", "faithfulness": 0.2, "answer_relevancy": 0.3, "context_precision": 0.4, "context_recall": 0.5}]
    )

    comparison, summary = build_ragas_comparison(graph_rows, vector_rows, {}, {})

    assert comparison.loc[0, "ragas_winner"] == "graph"
    assert summary["winner_counts"]["graph"] == 1


def test_metric_disagreement_flags_conflicting_judge() -> None:
    results = pd.DataFrame(
        [{"id": "q1", "question": "Q?", "query_class": "comparison", "winner": "graph", "failure_mode": "none"}]
    )
    ragas = pd.DataFrame([{"id": "q1", "ragas_winner": "graph"}])
    judge = pd.DataFrame([{"id": "q1", "winner": "vector", "confidence": 0.9, "rationale": "Vector is better."}])

    disagreement = build_metric_disagreement(results, ragas, judge)

    assert bool(disagreement.loc[0, "has_disagreement"]) is True
    assert "deterministic_vs_judge" in disagreement.loc[0, "disagreement_flags"]


def test_query_difficulty_scores_hard_multi_hop() -> None:
    row = pd.Series(
        {
            "id": "q1",
            "question": "Q?",
            "query_class": "multi_hop",
            "expected_entities": json.dumps(["A", "B"]),
            "source_fields": json.dumps(["field_a", "field_b"]),
            "contexts": json.dumps(["one", "two"]),
            "ground_truth": "This answer requires enough tokens to look like a synthesis answer with multiple facts.",
        }
    )

    score, band, reasons = difficulty_score(row)

    assert score >= 4
    assert band in {"hard", "very_hard"}
    assert "multiple_expected_entities" in reasons


def test_cost_estimation_uses_configured_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_GROQ_INPUT_PER_1M", "1")
    monkeypatch.setenv("COST_GROQ_OUTPUT_PER_1M", "2")

    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_cost("groq", 1_000_000, 500_000) == 2.0


def test_combine_outputs_keeps_dataset_names() -> None:
    tmp_path = Path(__file__).resolve().parents[1] / "outputs" / "_pytest_combine"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    for name in ["manual_28", "autoq_35"]:
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame(
            [
                {
                    "id": f"{name}-q1",
                    "query_class": "multi_hop",
                    "winner": "graph",
                    "graph_score": 0.8,
                    "vector_score": 0.2,
                    "graph_entity_recall": 1.0,
                    "vector_entity_recall": 0.0,
                    "graph_context_precision": 1.0,
                    "vector_context_precision": 0.0,
                    "graph_faithfulness": 1.0,
                    "vector_faithfulness": 0.0,
                    "graph_answer_relevancy": 1.0,
                    "vector_answer_relevancy": 0.0,
                    "failure_mode": "none",
                }
            ]
        ).to_csv(
            folder / "eval_results.csv", index=False
        )

    out_dir = combine_outputs(tmp_path, ["manual_28", "autoq_35"])
    combined = pd.read_csv(out_dir / "eval_results.csv")

    assert set(combined["dataset_name"]) == {"manual_28", "autoq_35"}
    shutil.rmtree(tmp_path)


def test_graph_coverage_formula_scores_entities_relations_and_density() -> None:
    row = pd.Series(
        {
            "id": "q1",
            "question": "Q?",
            "query_class": "relationship_reasoning",
            "winner": "graph",
            "expected_entities": json.dumps(["NFC-RM", "NFC-50/100"]),
            "source_fields": json.dumps(["compatible_with", "supports"]),
            "graph_retrieved_contexts": json.dumps(
                [
                    {"text": "NFC-RM Related: COMPATIBLE_WITH NFC-50/100 and supports paging."},
                    {"text": "Unrelated context."},
                ]
            ),
        }
    )

    scored = score_graph_coverage_row(row)

    assert scored["graph_expected_entity_coverage"] == 1.0
    assert scored["relation_signal_coverage"] == 1.0
    assert scored["structural_density"] == 0.5
    assert scored["graph_coverage_score"] == 0.9


def test_bootstrap_suppresses_small_query_class_ci() -> None:
    results = pd.DataFrame(
        [
            {
                "id": "q1",
                "query_class": "tiny",
                "winner": "graph",
                "graph_score": 0.8,
                "vector_score": 0.2,
            },
            {
                "id": "q2",
                "query_class": "tiny",
                "winner": "vector",
                "graph_score": 0.1,
                "vector_score": 0.7,
            },
        ]
    )

    intervals = build_bootstrap_intervals(results, resamples=20, seed=1, min_class_n=5)
    tiny = intervals[intervals["scope"] == "query_class:tiny"].iloc[0]

    assert tiny["status"] == "insufficient_n_for_ci"


def test_cost_latency_marks_missing_latency() -> None:
    results = pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "Q?",
                "ground_truth": "A",
                "graph_answer": "A",
                "vector_answer": "B",
                "graph_retrieved_contexts": json.dumps(["A context"]),
                "vector_retrieved_contexts": json.dumps(["B context"]),
            }
        ]
    )
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "_pytest_latency"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    graph = output_dir / "graph.csv"
    pd.DataFrame([{"id": "q1", "answer": "A", "retrieved_contexts": json.dumps(["A"])}]).to_csv(
        graph, index=False
    )

    audit = build_cost_latency_audit(results, graph_predictions=str(graph))

    prediction = audit[audit["stage"] == "prediction"].iloc[0]
    assert prediction["latency_status"] == "latency_not_supplied"
    shutil.rmtree(output_dir)


def test_methods_appendix_includes_metrics_version_and_dataset_hash() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "_pytest_report"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    dataset = output_dir / "dataset.csv"
    dataset.write_text("id,question,ground_truth,contexts,query_class\nq1,Q,A,\"[]\",tiny\n", encoding="utf-8")
    results = pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "Q",
                "ground_truth": "A",
                "query_class": "tiny",
                "winner": "graph",
                "failure_mode": "none",
                "graph_score": 1.0,
                "vector_score": 0.0,
                "graph_entity_recall": 1.0,
                "vector_entity_recall": 0.0,
                "graph_context_precision": 1.0,
                "vector_context_precision": 0.0,
                "graph_faithfulness": 1.0,
                "vector_faithfulness": 0.0,
                "graph_answer_relevancy": 1.0,
                "vector_answer_relevancy": 0.0,
            }
        ]
    )
    results.attrs["benchmark_path"] = str(dataset)
    summary_by_class = eval_mod._summarize_by_query_class(results)
    summary = eval_mod._metric_summary(results, summary_by_class)
    report = output_dir / "report.md"

    eval_mod._write_report(report, results, summary_by_class, summary)
    text = report.read_text(encoding="utf-8")

    assert "Methods Appendix" in text
    assert METRICS_VERSION in text
    assert "Dataset SHA256" in text
    assert "Tier 1" in text
    shutil.rmtree(output_dir)


def test_manifest_records_hashes_and_no_environment() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "_pytest_manifest"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    dataset = output_dir / "dataset.csv"
    graph = output_dir / "graph.csv"
    vector = output_dir / "vector.csv"
    dataset.write_text("dataset", encoding="utf-8")
    graph.write_text("graph", encoding="utf-8")
    vector.write_text("vector", encoding="utf-8")

    manifest = build_manifest(
        output_dir,
        str(dataset),
        str(graph),
        str(vector),
        input_sets=[
            {
                "name": "manual_28",
                "dataset": str(dataset),
                "graph_output": str(graph),
                "vector_output": str(vector),
            }
        ],
        argv=["--leaderboard"],
        ragas_mode="skip",
        skip_llm_judge=True,
        combined_from=["manual_28", "autoq_35"],
    )

    assert manifest["metrics_version"] == METRICS_VERSION
    assert manifest["inputs"]["dataset"]["sha256"]
    assert manifest["input_sets"][0]["name"] == "manual_28"
    assert manifest["input_sets"][0]["dataset"]["sha256"]
    assert manifest["combined_from"] == ["manual_28", "autoq_35"]
    assert "environment" not in manifest
    assert manifest["layers"]["ragas"] is False
    assert manifest["layers"]["llm_judge"] is False
    shutil.rmtree(output_dir)


def test_graph_coverage_interpretation_uses_median_threshold() -> None:
    coverage = pd.DataFrame(
        [
            {"id": "q1", "winner": "vector", "graph_coverage_score": 0.9, "graph_expected_entity_coverage": 1.0, "relation_signal_coverage": 1.0, "structural_density": 0.5},
            {"id": "q2", "winner": "vector", "graph_coverage_score": 0.1, "graph_expected_entity_coverage": 0.0, "relation_signal_coverage": 0.0, "structural_density": 0.5},
            {"id": "q3", "winner": "graph", "graph_coverage_score": 0.5, "graph_expected_entity_coverage": 0.5, "relation_signal_coverage": 0.5, "structural_density": 0.5},
        ]
    )

    interpretation = interpret_graph_coverage(coverage)

    assert interpretation["median_graph_coverage_score"] == 0.5
    assert interpretation["high_coverage_graph_losses"] == 1
    assert interpretation["low_coverage_graph_losses"] == 1


def test_disagreement_summary_schema_and_confidence_buckets() -> None:
    disagreement = pd.DataFrame(
        [
            {
                "deterministic_winner": "graph",
                "ragas_winner": "vector",
                "llm_judge_winner": "vector",
                "judge_confidence": 0.9,
                "disagreement_flags": "deterministic_vs_ragas;deterministic_vs_judge",
                "has_disagreement": True,
            },
            {
                "deterministic_winner": "tie",
                "ragas_winner": "missing",
                "llm_judge_winner": "neither",
                "judge_confidence": 0.4,
                "disagreement_flags": "low_judge_confidence",
                "has_disagreement": False,
            },
        ]
    )

    summary = summarize_disagreement(disagreement)

    assert summary["deterministic_vs_ragas_conflicts"] == 1
    assert summary["deterministic_vs_judge_conflicts"] == 1
    assert summary["ragas_vs_judge_conflicts"] == 0
    assert summary["judge_confidence_buckets"]["high"] == 1
    assert summary["judge_confidence_buckets"]["low"] == 1


def test_leaderboard_sets_generation_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run_one_dataset(args):
        captured["skip_graph_domain_build"] = args.skip_graph_domain_build
        captured["skip_graph_output_generation"] = args.skip_graph_output_generation
        captured["skip_vector_output_generation"] = args.skip_vector_output_generation
        captured["skip_graph_audit"] = args.skip_graph_audit

    monkeypatch.setattr(run_week3_pipeline, "_run_one_dataset", fake_run_one_dataset)
    monkeypatch.setattr(
        "sys.argv",
        [
            "src.run_week3_pipeline",
            "--leaderboard",
            "--dataset",
            "dataset.csv",
            "--graph-output",
            "graph.csv",
            "--vector-output",
            "vector.csv",
            "--ragas-mode",
            "skip",
            "--skip-llm-judge",
        ],
    )

    run_week3_pipeline.main()

    assert captured == {
        "skip_graph_domain_build": True,
        "skip_graph_output_generation": True,
        "skip_vector_output_generation": True,
        "skip_graph_audit": True,
    }


def test_leaderboard_never_calls_generation_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "_pytest_leaderboard"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    dataset = output_dir / "dataset.csv"
    graph = output_dir / "graph.csv"
    vector = output_dir / "vector.csv"
    dataset.write_text(
        "id,question,ground_truth,contexts,query_class\nq1,Q,A,\"[]\",single_hop_fact\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"id": "q1", "answer": "A", "retrieved_contexts": json.dumps([])}]).to_csv(
        graph, index=False
    )
    pd.DataFrame([{"id": "q1", "answer": "A", "retrieved_contexts": json.dumps([])}]).to_csv(
        vector, index=False
    )

    def fake_run_module(module: str, args: list[str] | None = None) -> None:
        calls.append(module)

    monkeypatch.setattr(run_week3_pipeline, "_run_module", fake_run_module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "src.run_week3_pipeline",
            "--leaderboard",
            "--dataset",
            str(dataset),
            "--graph-output",
            str(graph),
            "--vector-output",
            str(vector),
            "--ragas-mode",
            "skip",
            "--skip-llm-judge",
            "--out-dir",
            str(output_dir / "eval"),
        ],
    )

    with pytest.raises(FileNotFoundError, match="dashboard inputs"):
        run_week3_pipeline.main()

    forbidden = {
        "src.ingest",
        "src.week2_graph_rag.build_honeywell_graphrag_domain",
        "src.week2_graph_rag.run_graphrag_predictions",
    }
    assert not forbidden.intersection(calls)
    shutil.rmtree(output_dir)


def test_leaderboard_allow_graph_audit_keeps_audit_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run_one_dataset(args):
        captured["skip_graph_audit"] = args.skip_graph_audit

    monkeypatch.setattr(run_week3_pipeline, "_run_one_dataset", fake_run_one_dataset)
    monkeypatch.setattr(
        "sys.argv",
        [
            "src.run_week3_pipeline",
            "--leaderboard",
            "--leaderboard-allow-graph-audit",
            "--dataset",
            "dataset.csv",
            "--graph-output",
            "graph.csv",
            "--vector-output",
            "vector.csv",
            "--ragas-mode",
            "skip",
            "--skip-llm-judge",
        ],
    )

    run_week3_pipeline.main()

    assert captured["skip_graph_audit"] is False


def test_combined_manifest_uses_dataset_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run_one_dataset(args):
        return None

    def fake_combine(base_dir, dataset_names):
        captured["combined_names"] = dataset_names

    def fake_write_manifest(args, out_dir, combined_from=None, input_sets=None):
        captured["combined_from"] = combined_from
        captured["input_sets"] = input_sets

    def fake_render(args, combined_dir):
        captured["rendered"] = str(combined_dir)

    monkeypatch.setattr(run_week3_pipeline, "_run_one_dataset", fake_run_one_dataset)
    monkeypatch.setattr(run_week3_pipeline, "combine_dataset_outputs", fake_combine)
    monkeypatch.setattr(run_week3_pipeline, "write_run_manifest", fake_write_manifest)
    monkeypatch.setattr(run_week3_pipeline, "render_combined_outputs", fake_render)
    monkeypatch.setattr(
        "sys.argv",
        ["src.run_week3_pipeline", "--dataset-mode", "both", "--ragas-mode", "skip", "--skip-llm-judge"],
    )

    run_week3_pipeline.main()

    assert captured["combined_from"] == ["manual_28", "autoq_35"]
    assert [item["name"] for item in captured["input_sets"]] == ["manual_28", "autoq_35"]
    assert captured["input_sets"][0]["dataset"].endswith("honeywell_hard_labels_28.csv")
    assert captured["input_sets"][1]["dataset"].endswith("honeywell_autoq_35.csv")
