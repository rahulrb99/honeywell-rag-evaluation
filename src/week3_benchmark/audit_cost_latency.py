from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.week3_benchmark.run_week3_eval import _context_texts


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval_outputs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate cost and latency for evaluation artifacts.")
    parser.add_argument("--results", default=str(DEFAULT_OUT_DIR / "eval_results.csv"))
    parser.add_argument("--graph-predictions", default="")
    parser.add_argument("--vector-predictions", default="")
    parser.add_argument("--judge", default=str(DEFAULT_OUT_DIR / "pairwise_llm_judge.csv"))
    parser.add_argument("--ragas-comparison", default=str(DEFAULT_OUT_DIR / "ragas_system_comparison.csv"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def estimate_tokens(text: object) -> int:
    return max(1, int(len(str(text or "")) / 4))


def _rate(provider: str, direction: str) -> float:
    key = f"COST_{provider.upper()}_{direction.upper()}_PER_1M"
    try:
        return float(os.getenv(key, "0") or 0.0)
    except ValueError:
        return 0.0


def configured_cost_rates(provider: str) -> dict[str, object]:
    input_key = f"COST_{provider.upper()}_INPUT_PER_1M"
    output_key = f"COST_{provider.upper()}_OUTPUT_PER_1M"
    input_raw = os.getenv(input_key, "")
    output_raw = os.getenv(output_key, "")
    return {
        "provider": provider,
        "input_rate_key": input_key,
        "output_rate_key": output_key,
        "input_rate_configured": bool(str(input_raw).strip()),
        "output_rate_configured": bool(str(output_raw).strip()),
        "cost_rates_configured": bool(str(input_raw).strip()) and bool(str(output_raw).strip()),
    }


def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    return round((input_tokens / 1_000_000) * _rate(provider, "input") + (output_tokens / 1_000_000) * _rate(provider, "output"), 8)


def _prediction_latency(path: str, system: str) -> list[dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    df = pd.read_csv(path)
    if "latency_ms" not in df.columns:
        return [
            {
                "id": row.get("id", ""),
                "stage": "prediction",
                "system": system,
                "latency_ms": "",
                "latency_status": "latency_not_supplied",
                "input_tokens_est": 0,
                "output_tokens_est": estimate_tokens(row.get("answer", "")),
                "cost_est_usd": 0.0,
            }
            for _, row in df.iterrows()
        ]
    return [
        {
            "id": row.get("id", ""),
            "stage": "prediction",
            "system": system,
            "latency_ms": row.get("latency_ms", 0),
            "latency_status": "ok",
            "input_tokens_est": 0,
            "output_tokens_est": estimate_tokens(row.get("answer", "")),
            "cost_est_usd": 0.0,
        }
        for _, row in df.iterrows()
    ]


def build_cost_latency_audit(
    results: pd.DataFrame,
    graph_predictions: str = "",
    vector_predictions: str = "",
    judge: pd.DataFrame | None = None,
    ragas: pd.DataFrame | None = None,
    provider: str = "groq",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_prediction_latency(graph_predictions, "graph"))
    rows.extend(_prediction_latency(vector_predictions, "vector"))
    for _, row in results.iterrows():
        for system in ("graph", "vector"):
            contexts = "\n".join(_context_texts(row.get(f"{system}_retrieved_contexts", "")))
            answer = row.get(f"{system}_answer", "")
            prompt_text = "\n\n".join([str(row.get("question", "")), str(row.get("ground_truth", "")), contexts])
            input_tokens = estimate_tokens(prompt_text)
            output_tokens = estimate_tokens(answer)
            rows.append(
                {
                    "id": row.get("id", ""),
                    "stage": "ragas_estimate",
                    "system": system,
                    "latency_ms": 0,
                    "latency_status": "not_measured",
                    "input_tokens_est": input_tokens,
                    "output_tokens_est": output_tokens,
                    "cost_est_usd": estimate_cost(provider, input_tokens, output_tokens),
                }
            )
    if judge is not None and not judge.empty:
        for _, row in results.iterrows():
            input_text = "\n\n".join(
                [
                    str(row.get("question", "")),
                    str(row.get("ground_truth", "")),
                    str(row.get("graph_answer", "")),
                    str(row.get("vector_answer", "")),
                    "\n".join(_context_texts(row.get("graph_retrieved_contexts", ""))),
                    "\n".join(_context_texts(row.get("vector_retrieved_contexts", ""))),
                ]
            )
            output_text = ""
            if "id" in judge.columns:
                match = judge[judge["id"].astype(str) == str(row.get("id", ""))]
                if not match.empty:
                    output_text = str(match.iloc[0].get("rationale", ""))
            input_tokens = estimate_tokens(input_text)
            output_tokens = estimate_tokens(output_text)
            rows.append(
                {
                    "id": row.get("id", ""),
                    "stage": "llm_judge_estimate",
                    "system": "judge",
                    "latency_ms": 0,
                    "latency_status": "not_measured",
                    "input_tokens_est": input_tokens,
                    "output_tokens_est": output_tokens,
                    "cost_est_usd": estimate_cost(provider, input_tokens, output_tokens),
                }
            )
    return pd.DataFrame(rows)


def summarize_cost_latency(audit: pd.DataFrame) -> dict[str, object]:
    if audit.empty:
        return {"rows": 0, "total_cost_est_usd": 0.0}
    latency_status = audit["latency_status"] if "latency_status" in audit.columns else pd.Series([])
    grouped = (
        audit.groupby(["stage", "system"], dropna=False)
        .agg(
            rows=("id", "count"),
            latency_ms_total=("latency_ms", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            input_tokens_est=("input_tokens_est", "sum"),
            output_tokens_est=("output_tokens_est", "sum"),
            cost_est_usd=("cost_est_usd", "sum"),
        )
        .reset_index()
        .round({"cost_est_usd": 8})
    )
    return {
        "rows": int(len(audit)),
        "total_latency_ms": int(pd.to_numeric(audit["latency_ms"], errors="coerce").fillna(0).sum()),
        "latency_missing_rows": int((latency_status == "latency_not_supplied").sum()),
        "total_input_tokens_est": int(audit["input_tokens_est"].fillna(0).sum()),
        "total_output_tokens_est": int(audit["output_tokens_est"].fillna(0).sum()),
        "total_cost_est_usd": round(float(audit["cost_est_usd"].fillna(0).sum()), 8),
        "by_stage_system": grouped.to_dict(orient="records"),
    }


def main() -> None:
    args = _parse_args()
    results = pd.read_csv(args.results)
    judge = pd.read_csv(args.judge) if Path(args.judge).exists() else pd.DataFrame()
    ragas = pd.read_csv(args.ragas_comparison) if Path(args.ragas_comparison).exists() else pd.DataFrame()
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    audit = build_cost_latency_audit(
        results,
        args.graph_predictions,
        args.vector_predictions,
        judge,
        ragas,
        provider=provider,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "cost_latency_audit.csv", index=False)
    summary = summarize_cost_latency(audit)
    summary.update(configured_cost_rates(provider))
    if not summary.get("cost_rates_configured"):
        summary["cost_warning"] = "Cost rates are not fully configured; dollar estimates may be zero."
    (out_dir / "cost_latency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved cost/latency audit -> {out_dir}")


if __name__ == "__main__":
    main()
