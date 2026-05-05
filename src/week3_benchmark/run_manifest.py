from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.week3_benchmark.metrics_metadata import METRICS_VERSION, file_sha256, git_sha


ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--graph-output", default="")
    parser.add_argument("--vector-output", default="")
    parser.add_argument("--input-set-json", default="")
    parser.add_argument("--argv-json", default="[]")
    parser.add_argument("--ragas-mode", default="skip")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--combined-from", nargs="*", default=[])
    return parser.parse_args()


def _file_entry(path: str) -> dict[str, str]:
    if not path:
        return {"path": "", "sha256": ""}
    p = Path(path)
    return {"path": str(p), "sha256": file_sha256(p)}


def build_manifest(
    out_dir: Path,
    dataset: str = "",
    graph_output: str = "",
    vector_output: str = "",
    input_sets: list[dict[str, str]] | None = None,
    argv: list[str] | None = None,
    ragas_mode: str = "skip",
    skip_llm_judge: bool = False,
    combined_from: list[str] | None = None,
) -> dict[str, Any]:
    combined_from = combined_from or []
    input_sets = input_sets or []
    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "argv": argv or [],
        "output_directory": str(out_dir),
        "metrics_version": METRICS_VERSION,
        "git_sha": git_sha(ROOT) or "unknown",
        "python_version": platform.python_version(),
        "python_full": platform.python_build(),
        "layers": {
            "deterministic": True,
            "ragas": ragas_mode != "skip",
            "llm_judge": not skip_llm_judge,
            "graph_coverage": True,
            "bootstrap": True,
            "cost_latency": True,
        },
        "inputs": {
            "dataset": _file_entry(dataset),
            "graph_output": _file_entry(graph_output),
            "vector_output": _file_entry(vector_output),
        },
    }
    if input_sets:
        manifest["input_sets"] = [
            {
                "name": item.get("name", ""),
                "dataset": _file_entry(item.get("dataset", "")),
                "graph_output": _file_entry(item.get("graph_output", "")),
                "vector_output": _file_entry(item.get("vector_output", "")),
            }
            for item in input_sets
        ]
    if combined_from:
        manifest["combined_from"] = combined_from
        manifest["combined_eval_results"] = _file_entry(str(out_dir / "eval_results.csv"))
    return manifest


def write_manifest(
    out_dir: Path,
    dataset: str = "",
    graph_output: str = "",
    vector_output: str = "",
    input_sets: list[dict[str, str]] | None = None,
    argv: list[str] | None = None,
    ragas_mode: str = "skip",
    skip_llm_judge: bool = False,
    combined_from: list[str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        out_dir,
        dataset,
        graph_output,
        vector_output,
        input_sets,
        argv,
        ragas_mode,
        skip_llm_judge,
        combined_from,
    )
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = _parse_args()
    path = write_manifest(
        Path(args.out_dir),
        args.dataset,
        args.graph_output,
        args.vector_output,
        json.loads(args.input_set_json) if args.input_set_json else None,
        json.loads(args.argv_json),
        args.ragas_mode,
        args.skip_llm_judge,
        args.combined_from,
    )
    print(f"Saved run manifest -> {path}")


if __name__ == "__main__":
    main()
