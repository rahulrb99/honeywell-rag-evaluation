from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from .config import AGGREGATE_RESULTS_PATH, DASHBOARD_PATH, GRAPH_METADATA_PATH, QUESTION_RESULTS_PATH, RUN_MANIFEST_PATH
from .io_utils import read_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(value: str | float | None) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return ""


def _num(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return html.escape(str(value))


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _bar(label: str, a_value: float, b_value: float, inverse: bool = False) -> str:
    max_value = max(a_value, b_value, 1e-9)
    if inverse:
        a_width = (1 - min(a_value / max_value, 1)) * 100
        b_width = (1 - min(b_value / max_value, 1)) * 100
    else:
        a_width = a_value / max_value * 100
        b_width = b_value / max_value * 100
    return f"""
    <div class="bar-row">
      <div class="bar-label">{html.escape(label)}</div>
      <div class="bar-wrap"><span class="bar a" style="width:{a_width:.1f}%"></span></div>
      <div class="bar-wrap"><span class="bar b" style="width:{b_width:.1f}%"></span></div>
    </div>
    """


def generate_dashboard(
    aggregate_path: Path = AGGREGATE_RESULTS_PATH,
    question_path: Path = QUESTION_RESULTS_PATH,
    metadata_path: Path = GRAPH_METADATA_PATH,
    manifest_path: Path = RUN_MANIFEST_PATH,
    out_path: Path = DASHBOARD_PATH,
) -> Path:
    aggregate = _read_csv(aggregate_path)
    questions = _read_csv(question_path)
    metadata = read_json(metadata_path, default={}) or {}
    manifest = read_json(manifest_path, default={}) or {}

    by_graph = {row.get("graph", ""): row for row in aggregate}
    graph_a = by_graph.get("graph_a", {})
    graph_b = by_graph.get("graph_b", {})

    bars = "".join(
        [
            _bar("Reachability@3", float(graph_a.get("reachability_at_3", 0) or 0), float(graph_b.get("reachability_at_3", 0) or 0)),
            _bar("Path Precision", float(graph_a.get("path_precision", 0) or 0), float(graph_b.get("path_precision", 0) or 0)),
            _bar("QA Accuracy", float(graph_a.get("qa_accuracy", 0) or 0), float(graph_b.get("qa_accuracy", 0) or 0)),
            _bar("Latency", float(graph_a.get("avg_latency", 0) or 0), float(graph_b.get("avg_latency", 0) or 0), inverse=True),
        ]
    )

    meta_rows = []
    for graph_name, stats in metadata.items():
        row = {"graph": graph_name}
        row.update(stats)
        meta_rows.append(row)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Graph Evaluation Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; color: #1f2933; background: #f7f8fa; }}
    header {{ background: #111827; color: white; padding: 24px 32px; }}
    main {{ padding: 24px 32px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 32px; font-size: 20px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #dde2e8; border-radius: 8px; padding: 16px; }}
    .metric {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dde2e8; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e5e9ef; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; }}
    .bars {{ background: white; border: 1px solid #dde2e8; border-radius: 8px; padding: 16px; }}
    .bar-row {{ display: grid; grid-template-columns: 160px 1fr 1fr; gap: 10px; align-items: center; margin: 10px 0; }}
    .bar-wrap {{ height: 16px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }}
    .bar {{ display: block; height: 100%; }}
    .bar.a {{ background: #2563eb; }}
    .bar.b {{ background: #059669; }}
    .legend {{ margin-top: 8px; color: #4b5563; }}
    code {{ background: #e5e7eb; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Graph Evaluation Dashboard</h1>
    <div>Run <code>{html.escape(str(manifest.get("run_id", "unknown")))}</code></div>
  </header>
  <main>
    <section class="cards">
      <div class="card">Graph A Reachability<div class="metric">{_pct(graph_a.get("reachability_at_3"))}</div></div>
      <div class="card">Graph B Reachability<div class="metric">{_pct(graph_b.get("reachability_at_3"))}</div></div>
      <div class="card">Graph A QA Accuracy<div class="metric">{_pct(graph_a.get("qa_accuracy"))}</div></div>
      <div class="card">Graph B QA Accuracy<div class="metric">{_pct(graph_b.get("qa_accuracy"))}</div></div>
      <div class="card">Graph A Latency<div class="metric">{_num(graph_a.get("avg_latency", ""))}s</div></div>
      <div class="card">Graph B Latency<div class="metric">{_num(graph_b.get("avg_latency", ""))}s</div></div>
    </section>

    <h2>Metric Comparison</h2>
    <div class="bars">{bars}<div class="legend"><span style="color:#2563eb">Graph A</span> vs <span style="color:#059669">Graph B</span></div></div>

    <h2>Aggregate Results</h2>
    {_table(aggregate, ["graph", "reachability_at_3", "avg_hops", "path_precision", "qa_accuracy", "avg_latency"])}

    <h2>Graph Structure</h2>
    {_table(meta_rows, ["graph", "nodes", "edges", "avg_degree", "connected_components", "largest_component_size"])}

    <h2>Run Manifest</h2>
    {_table([{k: v for k, v in manifest.items()}], list(manifest.keys())) if manifest else ""}

    <h2>Per-Question Results</h2>
    {_table(questions, ["question_id", "question", "graph", "reachable", "hops", "path_precision", "correct", "latency", "supporting_chunk_ids", "answer"])}
  </main>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def main() -> None:
    path = generate_dashboard()
    print(f"Wrote dashboard to {path}")


if __name__ == "__main__":
    main()
