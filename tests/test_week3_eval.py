from __future__ import annotations

import json

import pandas as pd
import pytest

from src.week3_benchmark import run_week3_eval as eval_mod


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
