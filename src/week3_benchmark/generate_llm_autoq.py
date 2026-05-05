from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.config import get_settings
from src.product_records import ProductRecord, load_product_records


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "eval" / "honeywell_autoq_35.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "eval_outputs" / "autoq_summary.json"
DEFAULT_COUNTS = ROOT / "outputs" / "eval_outputs" / "autoq_query_class_counts.csv"
DEFAULT_SAMPLES = ROOT / "outputs" / "eval_outputs" / "autoq_sample_questions_by_class.csv"
DEFAULT_REJECTED = ROOT / "outputs" / "eval_outputs" / "autoq_rejected_questions.csv"

TARGET_COUNTS = {
    "local_specific": 5,
    "local_global": 5,
    "comparison": 5,
    "multi_hop": 7,
    "relationship_reasoning": 5,
    "global_specific": 4,
    "global_thematic": 4,
}

CLASS_GUIDANCE = {
    "local_specific": (
        "Ask for one concrete fact, value, capability, or limit about one product."
    ),
    "local_global": (
        "Ask for a compact answer combining multiple facts about one product or a small related set."
    ),
    "comparison": (
        "Ask for a comparison between two or more products, limits, capabilities, or configurations."
    ),
    "multi_hop": (
        "Ask a question that requires combining at least two evidence facts to answer completely."
    ),
    "relationship_reasoning": (
        "Ask how products, components, systems, wiring, or usage contexts relate to each other."
    ),
    "global_specific": (
        "Ask for a specific synthesis across a product family or category, not a single value lookup."
    ),
    "global_thematic": (
        "Ask for a grounded pattern or theme across the evidence, while still requiring concrete examples."
    ),
}

REASONING_BY_CLASS = {
    "local_specific": "single_hop",
    "local_global": "aggregation",
    "comparison": "comparison",
    "multi_hop": "synthesis",
    "relationship_reasoning": "relationship_reasoning",
    "global_specific": "cross_product_synthesis",
    "global_thematic": "theme_summary",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class EvidencePacket:
    query_class: str
    records: list[ProductRecord]

    @property
    def source_product(self) -> str:
        return "|".join(sorted({record.product_name for record in self.records}))

    @property
    def source_fields(self) -> list[str]:
        return sorted({record.field_name for record in self.records})

    @property
    def contexts(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for record in self.records:
            text = str(record.evidence_text).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out

    def prompt_block(self) -> str:
        rows = []
        for idx, record in enumerate(self.records, start=1):
            rows.append(
                "\n".join(
                    [
                        f"Evidence {idx}",
                        f"product_name: {record.product_name}",
                        f"field_name: {record.field_name}",
                        f"field_value: {record.field_value}",
                        f"source_doc: {record.doc_id}",
                        f"evidence_text: {record.evidence_text}",
                    ]
                )
            )
        return "\n\n".join(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a high-quality 35-row LLM AutoQ benchmark from Honeywell evidence."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--counts", default=str(DEFAULT_COUNTS))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--rejected", default=str(DEFAULT_REJECTED))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts-per-row", type=int, default=4)
    parser.add_argument(
        "--provider",
        choices=("auto", "openai", "groq"),
        default=os.getenv("AUTOQ_PROVIDER", "auto"),
        help="LLM provider for generation. Use groq when OPENAI_API_KEY is stale/invalid.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AUTOQ_MODEL", ""),
        help="OpenAI-compatible model used for AutoQ generation.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Write evidence-assembled fallback rows if LLM generation fails.",
    )
    return parser.parse_args()


def _resolve_provider_and_model(provider: str, model: str) -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    selected = provider
    if selected == "auto":
        selected = "openai" if openai_key else "groq"
    if selected == "openai":
        return selected, model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    if selected == "groq":
        return selected, model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if groq_key:
        return "groq", model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return "openai", model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _build_client(provider: str) -> OpenAI:
    load_dotenv(ROOT / ".env")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if provider == "openai" and openai_key:
        return OpenAI(api_key=openai_key)
    if provider == "groq" and groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    raise ValueError(f"Set the required API key for AUTOQ_PROVIDER={provider} in the root .env.")


def _tokens(text: object) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(tok) > 2 and tok not in STOPWORDS
    }


def _clean_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")
    return parsed


def _validate_candidate(
    obj: dict[str, Any],
    packet: EvidencePacket,
    seen_questions: set[str],
) -> tuple[bool, str]:
    question = str(obj.get("question", "")).strip()
    answer = str(obj.get("ground_truth", "")).strip()
    if len(question) < 24:
        return False, "question too short"
    min_answer_len = 1 if packet.query_class == "local_specific" else 10
    if len(answer) < min_answer_len and not re.search(r"\d", answer):
        return False, "ground truth too short"
    question_key = re.sub(r"\s+", " ", question.lower()).strip()
    if question_key in seen_questions:
        return False, "duplicate question"
    if packet.query_class in {"comparison", "multi_hop", "relationship_reasoning"}:
        if len(packet.records) < 2:
            return False, "hard class needs at least two evidence records"
    context_tokens = _tokens(" ".join(packet.contexts))
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
        context_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(packet.contexts)))
        if not answer_numbers or not (answer_numbers & context_numbers):
            return False, "answer has no useful tokens"
    else:
        overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
        if overlap < 0.25:
            return False, f"answer not sufficiently grounded ({overlap:.2f})"
    if not any(product.lower() in (question + " " + answer).lower() for product in packet.source_product.split("|")):
        if packet.query_class not in {"global_thematic", "global_specific"}:
            return False, "missing source product mention"
    return True, "ok"


def _call_autoq_llm(client: OpenAI, model: str, packet: EvidencePacket) -> dict[str, Any]:
    system = (
        "You generate rigorous retrieval-augmented-generation benchmark questions. "
        "Use only the supplied Honeywell evidence. Do not invent facts. "
        "Return one strict JSON object and no markdown."
    )
    user = f"""Generate one high-quality AutoQ benchmark row.

Target query_class: {packet.query_class}
Class requirement: {CLASS_GUIDANCE[packet.query_class]}

Evidence:
{packet.prompt_block()}

Return JSON with exactly these keys:
{{
  "question": "string",
  "ground_truth": "string",
  "category": "short category label",
  "reasoning_type": "{REASONING_BY_CLASS[packet.query_class]}",
  "source_fields": ["field_name values used"],
  "support_notes": "brief explanation of which evidence supports the answer"
}}

Rules:
- The ground_truth must be answerable from the evidence above.
- For comparison, multi_hop, relationship_reasoning, global_specific, and global_thematic, use at least two evidence facts.
- Avoid generic questions. Ask something a technical evaluator would use to stress retrieval.
- Keep the question concise but specific.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
        max_tokens=700,
        response_format={"type": "json_object"},
    )
    return _clean_json(response.choices[0].message.content or "")


def _records_by_product(records: list[ProductRecord]) -> dict[str, list[ProductRecord]]:
    grouped: dict[str, list[ProductRecord]] = defaultdict(list)
    for record in records:
        grouped[record.product_name].append(record)
    return dict(grouped)


def _packet_plan(records: list[ProductRecord], seed: int) -> list[EvidencePacket]:
    rng = random.Random(seed)
    by_product = _records_by_product(records)
    products = sorted(by_product)
    packets: list[EvidencePacket] = []

    by_key = {(record.product_name, record.field_name): record for record in records}

    def pick(keys: list[tuple[str, str]]) -> list[ProductRecord]:
        selected = [by_key[key] for key in keys if key in by_key]
        if not selected:
            raise ValueError(f"No records found for keys: {keys}")
        return selected

    curated: dict[str, list[list[tuple[str, str]]]] = {
        "local_specific": [
            [("L-PCM06B", "dispersion_angle")],
            [("L-Series", "maximum_supervisory_voltage")],
            [("NFC-FFT", "remote_handset_capacity")],
            [("NFC-RM", "push_to_talk_paging")],
            [("RK-ZONE8", "monitored_contact_inputs")],
        ],
        "local_global": [
            [("L-PCM06B", "rated_input_voltage"), ("L-PCM06B", "max_power"), ("L-PCM06B", "power_tapping")],
            [("L-Series", "speaker_voltage_power_settings"), ("L-Series", "nominal_speaker_voltage"), ("L-Series", "frequency_range")],
            [("NFC-FFT", "primary_function"), ("NFC-FFT", "handset_control"), ("NFC-FFT", "wiring_configuration")],
            [("NFC-RM", "compatibility"), ("NFC-RM", "remote_microphone_capacity"), ("NFC-RM", "paging_function")],
            [("RK-ZONE8", "system_additions"), ("RK-ZONE8", "primary_usage"), ("RK-ZONE8", "max_devices_per_system")],
        ],
        "comparison": [
            [("NFC-FFT", "wiring_configuration"), ("NFC-RM", "wiring_configuration")],
            [("NFC-FFT", "remote_handset_capacity"), ("NFC-RM", "remote_microphone_capacity"), ("NFC-FFT", "phone_jack_capacity")],
            [("L-Series", "speaker_voltage_power_settings"), ("L-PCM06B", "power_tapping"), ("L-PCM06B", "rated_input_voltage")],
            [("RK-ZONE8", "main_power_supply_voltage"), ("RK-ZONE8", "backup_power_supply_voltage"), ("L-PCM06B", "rated_input_voltage")],
            [("L-Series", "frequency_range"), ("L-PCM06B", "effective_frequency_response_range")],
        ],
        "multi_hop": [
            [("RK-ZONE8", "system_additions"), ("RK-ZONE8", "max_devices_per_system")],
            [("RK-ZONE8", "main_power_supply_voltage"), ("RK-ZONE8", "backup_power_supply_voltage"), ("RK-ZONE8", "rated_power")],
            [("NFC-RM", "compatibility"), ("NFC-RM", "remote_microphone_capacity"), ("NFC-RM", "paging_function")],
            [("NFC-FFT", "phone_jack_capacity"), ("NFC-FFT", "remote_handset_capacity"), ("NFC-FFT", "handset_control")],
            [("L-Series", "speaker_voltage_power_settings"), ("L-Series", "candela_settings"), ("L-Series", "tone_capability")],
            [("L-PCM06B", "rated_input_voltage"), ("L-PCM06B", "max_power"), ("L-PCM06B", "power_tapping")],
            [("L-PCM06B", "usage_locations"), ("L-PCM06B", "connection_type"), ("L-PCM06B", "dimensions")],
        ],
        "relationship_reasoning": [
            [("NFC-RM", "compatibility"), ("NFC-RM", "paging_function"), ("NFC-FFT", "handset_control")],
            [("RK-ZONE8", "monitored_contact_input_link"), ("RK-ZONE8", "line_input"), ("RK-ZONE8", "contact_outputs")],
            [("RK-ZONE8", "automatic_fault_diagnosis"), ("RK-ZONE8", "fault_display"), ("RK-ZONE8", "monitored_contact_inputs")],
            [("L-Series", "tone_capability"), ("L-Series", "speaker_voltage_power_settings"), ("L-Series", "mounting_back_box")],
            [("L-PCM06B", "usage_locations"), ("L-PCM06B", "connection_type"), ("L-PCM06B", "effective_frequency_response_range")],
        ],
        "global_specific": [
            [("NFC-FFT", "primary_function"), ("NFC-RM", "compatibility"), ("NFC-RM", "paging_function"), ("NFC-FFT", "remote_handset_capacity")],
            [("L-Series", "speaker_voltage_power_settings"), ("L-Series", "candela_settings"), ("L-PCM06B", "power_tapping"), ("L-PCM06B", "max_power")],
            [("RK-ZONE8", "system_additions"), ("RK-ZONE8", "contact_outputs"), ("RK-ZONE8", "dc_power_output"), ("RK-ZONE8", "max_devices_per_system")],
            [("NFC-FFT", "wiring_configuration"), ("NFC-RM", "wiring_configuration"), ("RK-ZONE8", "monitored_contact_input_link"), ("RK-ZONE8", "line_input")],
        ],
        "global_thematic": [
            [("NFC-FFT", "primary_function"), ("NFC-RM", "paging_function"), ("RK-ZONE8", "monitored_contact_input_link"), ("L-Series", "tone_capability")],
            [("RK-ZONE8", "automatic_fault_diagnosis"), ("RK-ZONE8", "fault_display"), ("L-Series", "maximum_supervisory_voltage"), ("NFC-FFT", "wiring_configuration")],
            [("L-Series", "speaker_voltage_power_settings"), ("L-PCM06B", "power_tapping"), ("RK-ZONE8", "dc_power_output"), ("RK-ZONE8", "main_power_supply_voltage")],
            [("L-PCM06B", "usage_locations"), ("NFC-RM", "remote_console_role"), ("RK-ZONE8", "primary_usage"), ("NFC-FFT", "handset_control")],
        ],
    }

    if all(len(curated.get(cls, [])) >= count for cls, count in TARGET_COUNTS.items()):
        for cls, count in TARGET_COUNTS.items():
            for keys in curated[cls][:count]:
                packets.append(EvidencePacket(query_class=cls, records=pick(keys)))
        return packets

    def sample_product_records(product: str, count: int) -> list[ProductRecord]:
        pool = by_product[product]
        return rng.sample(pool, min(count, len(pool)))

    def sample_cross_product(count: int, min_products: int = 2) -> list[ProductRecord]:
        selected_products = rng.sample(products, min(min_products, len(products)))
        out = [rng.choice(by_product[product]) for product in selected_products]
        remaining = [record for record in records if record not in out]
        while len(out) < count and remaining:
            record = rng.choice(remaining)
            remaining.remove(record)
            out.append(record)
        return out

    for cls, target in TARGET_COUNTS.items():
        for idx in range(target):
            if cls == "local_specific":
                product = products[idx % len(products)]
                packet_records = sample_product_records(product, 1)
            elif cls == "local_global":
                product = products[idx % len(products)]
                packet_records = sample_product_records(product, 3)
            elif cls in {"comparison", "relationship_reasoning"}:
                packet_records = sample_cross_product(3, min_products=2)
            elif cls == "multi_hop":
                if idx % 2 == 0:
                    product = products[idx % len(products)]
                    packet_records = sample_product_records(product, 3)
                else:
                    packet_records = sample_cross_product(3, min_products=2)
            else:
                packet_records = sample_cross_product(5, min_products=3)
            packets.append(EvidencePacket(query_class=cls, records=packet_records))
    return packets


def _fallback_row(packet: EvidencePacket, row_id: int) -> dict[str, Any]:
    products = packet.source_product.replace("|", ", ")
    fields = ", ".join(packet.source_fields)
    contexts = packet.contexts
    question = f"What do the supplied Honeywell records show about {fields} for {products}?"
    if packet.query_class == "comparison":
        question = f"How do the supplied Honeywell records compare {fields} across {products}?"
    elif packet.query_class == "multi_hop":
        question = f"How should the evidence about {fields} be combined to describe {products}?"
    elif packet.query_class == "relationship_reasoning":
        question = f"What relationship between products or capabilities is supported by the evidence for {products}?"
    elif packet.query_class == "global_thematic":
        question = f"What recurring technical theme is visible across the supplied Honeywell evidence for {products}?"
    answer = " ".join(contexts[:3])
    return {
        "id": row_id,
        "question": question,
        "ground_truth": answer,
        "contexts": json.dumps(contexts, ensure_ascii=False),
        "category": packet.query_class,
        "query_class": packet.query_class,
        "reasoning_type": REASONING_BY_CLASS[packet.query_class],
        "source_product": packet.source_product,
        "source_fields": json.dumps(packet.source_fields, ensure_ascii=False),
        "source_split": "autoq_llm_35",
        "generation_method": "fallback_from_evidence",
        "support_notes": "Fallback row assembled directly from evidence after LLM generation failed.",
    }


def _generate_rows(
    records: list[ProductRecord],
    client: OpenAI,
    model: str,
    seed: int,
    max_attempts_per_row: int,
    allow_fallback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    packets = _packet_plan(records, seed)

    for idx, packet in enumerate(packets, start=1):
        accepted: dict[str, Any] | None = None
        for attempt in range(1, max_attempts_per_row + 1):
            try:
                candidate = _call_autoq_llm(client, model, packet)
                ok, reason = _validate_candidate(candidate, packet, seen_questions)
                if not ok:
                    rejected.append(
                        {
                            "target_id": idx,
                            "query_class": packet.query_class,
                            "attempt": attempt,
                            "reason": reason,
                            "question": candidate.get("question", ""),
                            "ground_truth": candidate.get("ground_truth", ""),
                            "source_product": packet.source_product,
                            "source_fields": json.dumps(packet.source_fields),
                        }
                    )
                    continue
                accepted = candidate
                break
            except Exception as exc:
                rejected.append(
                    {
                        "target_id": idx,
                        "query_class": packet.query_class,
                        "attempt": attempt,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "question": "",
                        "ground_truth": "",
                        "source_product": packet.source_product,
                        "source_fields": json.dumps(packet.source_fields),
                    }
                )
                time.sleep(min(2 * attempt, 8))

        if accepted is None and allow_fallback:
            row = _fallback_row(packet, idx)
        elif accepted is None:
            reasons = [
                item["reason"]
                for item in rejected
                if item.get("target_id") == idx
            ]
            raise RuntimeError(
                f"AutoQ generation failed for row {idx} ({packet.query_class}). "
                f"Last reasons: {' | '.join(reasons[-3:])}"
            )
        else:
            question = str(accepted["question"]).strip()
            row = {
                "id": idx,
                "question": question,
                "ground_truth": str(accepted["ground_truth"]).strip(),
                "contexts": json.dumps(packet.contexts, ensure_ascii=False),
                "category": str(accepted.get("category") or packet.query_class).strip(),
                "query_class": packet.query_class,
                "reasoning_type": REASONING_BY_CLASS[packet.query_class],
                "source_product": packet.source_product,
                "source_fields": json.dumps(packet.source_fields, ensure_ascii=False),
                "source_split": "autoq_llm_35",
                "generation_method": "llm_autoq",
                "support_notes": str(accepted.get("support_notes", "")).strip(),
            }
            seen_questions.add(re.sub(r"\s+", " ", question.lower()).strip())
        rows.append(row)
        print(f"[{idx:02d}/{len(packets)}] {packet.query_class}: {row['question']}")
    return rows, rejected


def _write_audits(
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    args: argparse.Namespace,
    product_record_count: int,
    model: str,
) -> None:
    out_path = Path(args.output)
    summary_path = Path(args.summary)
    counts_path = Path(args.counts)
    samples_path = Path(args.samples)
    rejected_path = Path(args.rejected)

    for path in (out_path, summary_path, counts_path, samples_path, rejected_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    counts = (
        df.groupby(["query_class", "reasoning_type"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["query_class", "reasoning_type"])
    )
    counts.to_csv(counts_path, index=False)

    samples = (
        df.sort_values("id")
        .groupby("query_class", dropna=False)
        .head(3)[["query_class", "id", "question", "source_product"]]
    )
    samples.to_csv(samples_path, index=False)

    pd.DataFrame(rejected).to_csv(rejected_path, index=False)

    summary = {
        "rows": len(rows),
        "model": model,
        "source_split": "autoq_llm_35",
        "product_record_count": product_record_count,
        "query_class_counts": dict(Counter(row["query_class"] for row in rows)),
        "reasoning_type_counts": dict(Counter(row["reasoning_type"] for row in rows)),
        "source_product_counts": dict(Counter(row["source_product"] for row in rows)),
        "rejected_attempts": len(rejected),
        "fallback_rows": sum(1 for row in rows if row["generation_method"] != "llm_autoq"),
        "outputs": {
            "dataset": str(out_path.relative_to(ROOT)),
            "counts": str(counts_path.relative_to(ROOT)),
            "samples": str(samples_path.relative_to(ROOT)),
            "rejected": str(rejected_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved AutoQ dataset -> {out_path}")
    print(f"Saved AutoQ summary -> {summary_path}")
    print(f"Saved query-class counts -> {counts_path}")
    print(f"Saved sample questions -> {samples_path}")
    print(f"Saved rejected attempts -> {rejected_path}")


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    records = load_product_records(settings.product_records_path)
    if not records:
        raise FileNotFoundError(
            f"No product records found at {settings.product_records_path}. "
            "Run `python -m src.ingest` first."
        )

    provider, model = _resolve_provider_and_model(args.provider, args.model)
    client = _build_client(provider)
    print(f"AutoQ provider: {provider}")
    print(f"AutoQ model:    {model}")
    rows, rejected = _generate_rows(
        records=records,
        client=client,
        model=model,
        seed=args.seed,
        max_attempts_per_row=args.max_attempts_per_row,
        allow_fallback=args.allow_fallback,
    )
    _write_audits(rows, rejected, args, len(records), model)


if __name__ == "__main__":
    main()
