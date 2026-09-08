from __future__ import annotations

import csv
import json

import networkx as nx

from newevals.src.build_graph_a import build_cooccurrence_graph
from newevals.src.build_graph_b import build_semantic_relation_graph
from newevals.src.chunking import chunk_page_records
from newevals.src.dashboard import generate_dashboard
from newevals.src.graph_stats import compute_graph_stats
from newevals.src.graphrag_adapter import export_semantic_graph
from newevals.src.manifest import build_run_manifest
from newevals.src.normalize import normalize_entity
from newevals.src.qa_pipeline import answer_graph_a, answer_graph_b
from newevals.src.retrieval import retrieve_graph_a


def test_normalize_entity_aliases_and_voltage_variants():
    assert normalize_entity("BACNET") == "bacnet"
    assert normalize_entity("BACnet protocol") == "bacnet"
    assert normalize_entity("24 Volts") == "24v"
    assert normalize_entity("24VDC") == "24v"


def test_chunking_is_deterministic_and_overlapping():
    pages = [{"doc_id": "manual", "source_document": "m.pdf", "page": 1, "text": " ".join(f"t{i}" for i in range(12))}]
    chunks = chunk_page_records(pages, chunk_tokens=5, overlap=2)
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "manual#chunk_0000",
        "manual#chunk_0001",
        "manual#chunk_0002",
        "manual#chunk_0003",
    ]
    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]


def test_graph_a_cooccurrence_uses_normalized_entities():
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "page_start": 1}]
    cache = {"c1": {"protocols": ["BACNET"], "voltages": ["24 Volts"], "components": ["controller"]}}
    graph = build_cooccurrence_graph(chunks, cache)
    assert "bacnet" in graph
    assert "24v" in graph
    assert graph.has_edge("bacnet", "24v")
    assert graph["bacnet"]["24v"]["supporting_chunk_ids"] == "c1"


def test_graph_b_rule_based_semantic_edges():
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "page_start": 1,
            "text": "The DAA2 amplifier supports BACnet and operates at 24 VDC.",
        }
    ]
    cache = {
        "c1": {
            "products": ["daa2"],
            "protocols": ["bacnet"],
            "voltages": ["24v"],
            "components": ["amplifier"],
            "error_codes": [],
        }
    }
    graph = build_semantic_relation_graph(chunks, cache, relation_mode="rules")
    assert graph.has_edge("daa2", "bacnet")
    assert graph["daa2"]["bacnet"]["relation"] == "supports_protocol"
    assert graph.has_edge("daa2", "24v")
    assert graph["daa2"]["24v"]["relation"] == "requires_voltage"
    assert graph["daa2"]["24v"]["supporting_chunk_ids"] == "c1"


def test_retrieval_includes_supporting_chunk_ids(monkeypatch):
    graph = nx.Graph()
    graph.add_edge("bacnet", "controller", relation="co_occurs_in_chunk", supporting_chunk_ids="c1|c2")

    def fake_extract_question_entities(question, question_id, model):
        return {"protocols": ["bacnet"], "products": [], "voltages": [], "components": [], "error_codes": []}

    monkeypatch.setattr("newevals.src.retrieval.extract_question_entities", fake_extract_question_entities)
    result = retrieve_graph_a(graph, "BACnet controller?", "q1", answer_entity="controller")
    assert result["reachable"] is True
    assert result["supporting_chunk_ids"] == ["c1", "c2"]


def test_graph_b_heuristic_uses_ranked_supporting_chunks():
    retrieval = {
        "question_entities": ["fcm 1"],
        "shortest_path": ["fcm 1", "control module"],
        "retrieved_edges": [
            ("fcm 1", "control module", {"relation": "is_device_category", "supporting_chunk_ids": "c2"})
        ],
        "supporting_chunk_ids": ["c1", "c2"],
    }
    chunks = {
        "c1": {"text": "Unrelated front matter and document revision history."},
        "c2": {"text": "The FCM-1 is described as a control module for SLC applications."},
    }

    result = answer_graph_b("What device category is the FCM-1 described with?", retrieval, chunks)

    assert "control module" in result["answer"].lower()
    assert result["citations"][0] == "c2"


def test_openai_qa_usage_is_consistent_for_both_graphs(monkeypatch):
    calls = []

    class _FakeMessage:
        content = '{"answer": "ok", "citations": []}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("newevals.src.qa_pipeline.USE_OPENAI_QA", True)
    monkeypatch.setattr("newevals.src.qa_pipeline.OpenAI", lambda: _FakeClient())

    retrieval = {
        "question_entities": ["bacnet"],
        "shortest_path": ["bacnet", "controller"],
        "retrieved_edges": [("bacnet", "controller", {"relation": "supports", "supporting_chunk_ids": "c1"})],
        "supporting_chunk_ids": ["c1"],
    }
    chunks = {"c1": {"text": "The controller supports BACnet."}}

    answer_graph_a("Does the controller support BACnet?", retrieval, chunks, model="model-a")
    answer_graph_b("Does the controller support BACnet?", retrieval, chunks, model="model-b")

    assert [call["model"] for call in calls] == ["model-a", "model-b"]


def test_graph_stats_directed_uses_weak_components():
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    graph.add_node("c")
    stats = compute_graph_stats(graph)
    assert stats["nodes"] == 3
    assert stats["edges"] == 1
    assert stats["connected_components"] == 2
    assert stats["largest_component_size"] == 2


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, *_args, **_kwargs):
        return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


def test_export_semantic_graph_from_neo4j_rows(tmp_path):
    rows = [{"source": "BACNET", "relation": "SUPPORTS", "target": "24 Volts", "weight": 0.8, "chunk_id": "c9"}]
    graph = export_semantic_graph(driver=_FakeDriver(rows), path=tmp_path / "g.graphml")
    assert graph.has_edge("bacnet", "24v")
    assert graph["bacnet"]["24v"]["supporting_chunk_ids"] == "c9"


def test_manifest_generation_records_reproducibility_fields():
    manifest = build_run_manifest(num_docs=8, num_chunks=214, model="gpt-4.1-mini", benchmark_file="benchmark_v1.json")
    assert manifest["seed"] == 42
    assert manifest["graph_a_type"] == "co_occurrence"
    assert manifest["graph_b_type"] == "semantic_relation"
    assert manifest["k_hops"] == 3


def test_dashboard_generation_from_fixture_files(tmp_path):
    aggregate = tmp_path / "aggregate.csv"
    with aggregate.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["graph", "reachability_at_3", "avg_hops", "path_precision", "qa_accuracy", "avg_latency"])
        writer.writeheader()
        writer.writerow({"graph": "graph_a", "reachability_at_3": "0.5", "avg_hops": "2", "path_precision": "0.25", "qa_accuracy": "0.5", "avg_latency": "0.1"})
        writer.writerow({"graph": "graph_b", "reachability_at_3": "1.0", "avg_hops": "1", "path_precision": "0.75", "qa_accuracy": "1.0", "avg_latency": "0.2"})

    questions = tmp_path / "questions.csv"
    with questions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "question", "graph", "reachable", "hops", "path_precision", "correct", "latency", "supporting_chunk_ids", "answer"])
        writer.writeheader()
        writer.writerow({"question_id": "q1", "question": "Q?", "graph": "graph_a", "reachable": "True", "hops": "1", "path_precision": "1", "correct": "True", "latency": "0.1", "supporting_chunk_ids": "[]", "answer": "A"})

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"graph_a": {"nodes": 1, "edges": 0}, "graph_b": {"nodes": 2, "edges": 1}}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"run_id": "test-run", "model": "gpt-4.1-mini"}), encoding="utf-8")

    out = generate_dashboard(aggregate, questions, metadata, manifest, tmp_path / "dashboard.html")
    html = out.read_text(encoding="utf-8")
    assert "Graph Evaluation Dashboard" in html
    assert "test-run" in html
