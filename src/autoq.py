from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.product_records import ProductRecord, load_product_records


@dataclass(frozen=True)
class AutoQRow:
    question: str
    ground_truth: str
    contexts: list[str]
    category: str
    query_class: str
    reasoning_type: str
    source_product: str
    source_fields: list[str]
    source_split: str = "autoq"


def _record_map(records: list[ProductRecord]) -> dict[str, ProductRecord]:
    return {record.field_name: record for record in records}


def _answer_for_record(record: ProductRecord) -> str:
    product = record.product_name
    value = record.field_value.strip().rstrip(".")
    field = record.field_name
    if field == "system_additions":
        return f"{product} adds {value} to the INTEVIO system."
    if field == "primary_usage":
        return f"{product} is {value}."
    if field == "monitored_contact_inputs":
        return f"{product} has {value}."
    if field == "monitored_contact_input_link":
        return f"{product} monitored contact inputs can be {value}."
    if field == "line_input":
        return value + "."
    if field == "contact_outputs":
        return f"{product} has {value}."
    if field == "dc_power_output":
        return f"{product} provides a {value}."
    if field == "automatic_fault_diagnosis":
        return f"It supervises faults such as {value}."
    if field == "fault_display":
        return f"Fault information is {value}."
    if field == "main_power_supply_voltage":
        return f"The {product} main power supply voltage is {value}."
    if field == "backup_power_supply_voltage":
        return f"The {product} backup power supply voltage is {value}."
    if field == "rated_power":
        return f"The {product} rated power is {value}."
    if field == "max_devices_per_system":
        return value + "."
    if field == "speaker_voltage_power_settings":
        return value + "."
    if field == "candela_settings":
        return f"Field-selectable candela settings on wall units: {value}."
    if field == "tone_capability":
        return value + "."
    if field == "mounting_back_box":
        return f"They mount using a {value}."
    if field == "nominal_speaker_voltage":
        return f"The nominal speaker voltages are {value}."
    if field == "maximum_supervisory_voltage":
        return f"The maximum supervisory voltage for L-Series speakers is {value}."
    if field == "strobe_flash_rate":
        return f"The L-Series strobe flash rate is {value}."
    if field == "frequency_range":
        return f"The L-Series speaker frequency range is {value}."
    if field == "rated_input_voltage":
        return f"Rated input voltage {value}."
    if field == "usage_locations":
        return f"The {product} is designed for locations such as {value}."
    if field == "max_power":
        return f"The {product} maximum power rating is {value}."
    if field == "power_tapping":
        return f"The {product} power tapping settings are {value}."
    if field == "effective_frequency_response_range":
        return f"The effective frequency response range of the {product} is {value}."
    if field == "dispersion_angle":
        return f"The dispersion angle of the {product} is {value}."
    if field == "connection_type":
        return f"The {product} uses {value}."
    if field == "dimensions":
        return f"The dimensions of the {product} are {value}."
    return record.evidence_text.rstrip(".") + "."


def _manual_style_category(field_name: str) -> str:
    mapping = {
        "system_additions": "product_description",
        "primary_usage": "usage",
        "monitored_contact_inputs": "capacity",
        "monitored_contact_input_link": "function",
        "line_input": "io",
        "contact_outputs": "io",
        "dc_power_output": "specification",
        "automatic_fault_diagnosis": "fault_monitoring",
        "fault_display": "ui",
        "main_power_supply_voltage": "specification",
        "backup_power_supply_voltage": "specification",
        "rated_power": "specification",
        "max_devices_per_system": "capacity",
        "speaker_voltage_power_settings": "configuration",
        "candela_settings": "configuration",
        "tone_capability": "capability",
        "mounting_back_box": "installation",
        "nominal_speaker_voltage": "specification",
        "maximum_supervisory_voltage": "specification",
        "strobe_flash_rate": "specification",
        "frequency_range": "specification",
        "rated_input_voltage": "specification",
        "usage_locations": "usage_environment",
        "max_power": "specification",
        "power_tapping": "configuration",
        "effective_frequency_response_range": "specification",
        "dispersion_angle": "specification",
        "connection_type": "io",
        "dimensions": "specification",
    }
    return mapping.get(field_name, "specification")


def _row_from_record(
    record: ProductRecord,
    question: str,
    query_class: str,
    reasoning_type: str,
) -> AutoQRow:
    return AutoQRow(
        question=question,
        ground_truth=_answer_for_record(record),
        contexts=[record.evidence_text],
        category=_manual_style_category(record.field_name),
        query_class=query_class,
        reasoning_type=reasoning_type,
        source_product=record.product_name,
        source_fields=[record.field_name],
    )


def _multi_row(
    question: str,
    records: list[ProductRecord],
    ground_truth: str,
    category: str,
    source_product: str,
    query_class: str = "multi_hop",
    reasoning_type: str = "synthesis",
) -> AutoQRow:
    return AutoQRow(
        question=question,
        ground_truth=ground_truth,
        contexts=[record.evidence_text for record in records],
        category=category,
        query_class=query_class,
        reasoning_type=reasoning_type,
        source_product=source_product,
        source_fields=[record.field_name for record in records],
    )


def generate_autoq_rows(records: list[ProductRecord]) -> list[AutoQRow]:
    by_product: dict[str, dict[str, ProductRecord]] = {}
    for record in records:
        by_product.setdefault(record.product_name, {})[record.field_name] = record

    rows: list[AutoQRow] = []

    rk = by_product.get("RK-ZONE8", {})
    if rk:
        rows.extend(
            [
                _row_from_record(rk["system_additions"], "What expansion capabilities does RK-ZONE8 contribute to an INTEVIO deployment?", "paraphrase", "single_hop"),
                _row_from_record(rk["primary_usage"], "In what kind of deployment is RK-ZONE8 mainly intended to be used?", "paraphrase", "single_hop"),
                _row_from_record(rk["monitored_contact_input_link"], "What system can the RK-ZONE8 monitored inputs interface with to trigger alert and evacuation messages?", "paraphrase", "single_hop"),
                _row_from_record(rk["line_input"], "Which RK-ZONE8 input is used when an external audio source needs to be connected?", "paraphrase", "single_hop"),
                _row_from_record(rk["contact_outputs"], "What output capability does RK-ZONE8 provide for activating external devices?", "paraphrase", "single_hop"),
                _row_from_record(rk["dc_power_output"], "What external-device power output is available on RK-ZONE8?", "paraphrase", "single_hop"),
                _row_from_record(rk["automatic_fault_diagnosis"], "Which fault categories are monitored by the RK-ZONE8 automatic fault diagnosis function?", "paraphrase", "list_retrieval"),
                _row_from_record(rk["fault_display"], "Where does the operator see RK-ZONE8 fault status?", "paraphrase", "single_hop"),
                _row_from_record(rk["main_power_supply_voltage"], "What AC supply range is specified for RK-ZONE8 main power?", "paraphrase", "single_hop"),
                _row_from_record(rk["backup_power_supply_voltage"], "What backup DC range is specified for RK-ZONE8?", "paraphrase", "single_hop"),
                _row_from_record(rk["rated_power"], "How much rated power does RK-ZONE8 consume?", "paraphrase", "single_hop"),
                _row_from_record(rk["max_devices_per_system"], "What is the system-level device limit for RK-ZONE8 expanders?", "paraphrase", "single_hop"),
            ]
        )
        rows.extend(
            [
                _multi_row(
                    "How does RK-ZONE8 expand the INTEVIO system, and how many RK-ZONE8 units can a single system support?",
                    [rk["system_additions"], rk["max_devices_per_system"]],
                    "RK-ZONE8 adds 8 speaker zones, 8 monitored contact inputs, and 8 contact outputs to the INTEVIO system. Up to 15 RK-ZONE8 devices can be connected on one system.",
                    "product_description",
                    "RK-ZONE8",
                ),
                _multi_row(
                    "What power-related specifications are listed for RK-ZONE8 main supply, backup supply, and rated power?",
                    [rk["main_power_supply_voltage"], rk["backup_power_supply_voltage"], rk["rated_power"]],
                    "The RK-ZONE8 main power supply voltage is AC 100-240 V 50/60 Hz. The backup power supply voltage is 21.5 VDC to 28.5 VDC. The rated power is 35 W.",
                    "specification",
                    "RK-ZONE8",
                ),
                _multi_row(
                    "What input and output interfaces are highlighted for RK-ZONE8 when integrating external sources and devices?",
                    [rk["line_input"], rk["contact_outputs"], rk["dc_power_output"]],
                    "RK-ZONE8 provides a single line input for an external audio source, 8 contact outputs used to activate external devices, and a 24 VDC output to power external devices.",
                    "io",
                    "RK-ZONE8",
                ),
            ]
        )

    ls = by_product.get("L-Series", {})
    if ls:
        rows.extend(
            [
                _row_from_record(ls["speaker_voltage_power_settings"], "Which selectable speaker voltage and wattage options are provided for L-Series speakers?", "paraphrase", "list_retrieval"),
                _row_from_record(ls["candela_settings"], "Which candela levels can be chosen on wall L-Series speaker strobes?", "paraphrase", "list_retrieval"),
                _row_from_record(ls["tone_capability"], "What special evacuation tone capability is documented for L-Series devices used with a compatible FACP?", "paraphrase", "single_hop"),
                _row_from_record(ls["mounting_back_box"], "What mounting hardware arrangement is specified for L-Series wall speaker and speaker strobe units?", "paraphrase", "single_hop"),
                _row_from_record(ls["nominal_speaker_voltage"], "What nominal speaker voltages are specified for L-Series speakers?", "paraphrase", "single_hop"),
                _row_from_record(ls["maximum_supervisory_voltage"], "What supervisory voltage limit is listed for L-Series speakers?", "paraphrase", "single_hop"),
                _row_from_record(ls["strobe_flash_rate"], "At what flash frequency do L-Series strobes operate?", "paraphrase", "single_hop"),
                _row_from_record(ls["frequency_range"], "What audio frequency range is specified for L-Series speakers?", "paraphrase", "single_hop"),
            ]
        )
        rows.extend(
            [
                _multi_row(
                    "For L-Series wall units, what selectable speaker voltage and power settings are offered, and what candela settings are available?",
                    [ls["speaker_voltage_power_settings"], ls["candela_settings"]],
                    "Selectable speaker voltage settings are 25 and 70.7 Vrms, selectable power settings are 1/4, 1/2, 1, and 2 watts, and wall-unit candela settings are 15, 30, 75, 95, 110, 135, and 185.",
                    "configuration",
                    "L-Series",
                ),
                _multi_row(
                    "What core L-Series electrical specs cover nominal speaker voltage, maximum supervisory voltage, and strobe flash behavior?",
                    [ls["nominal_speaker_voltage"], ls["maximum_supervisory_voltage"], ls["strobe_flash_rate"]],
                    "Nominal speaker voltage is 25 Volts or 70.7 Volts (nominal), maximum supervisory voltage is 50 VDC, and strobe flash rate is 1 flash per second.",
                    "specification",
                    "L-Series",
                ),
            ]
        )

    lp = by_product.get("L-PCM06B", {})
    if lp:
        rows.extend(
            [
                _row_from_record(lp["rated_input_voltage"], "What input voltage is specified for the L-PCM06B ceiling loudspeaker?", "paraphrase", "single_hop"),
                _row_from_record(lp["usage_locations"], "Which indoor location types are named as intended environments for L-PCM06B?", "paraphrase", "list_retrieval"),
                _row_from_record(lp["max_power"], "What maximum power value is listed for L-PCM06B?", "paraphrase", "single_hop"),
                _row_from_record(lp["power_tapping"], "What selectable power-tap settings are listed for L-PCM06B?", "paraphrase", "list_retrieval"),
                _row_from_record(lp["effective_frequency_response_range"], "What effective frequency response span is documented for L-PCM06B?", "paraphrase", "single_hop"),
                _row_from_record(lp["dispersion_angle"], "What dispersion angle is specified for L-PCM06B?", "paraphrase", "single_hop"),
                _row_from_record(lp["connection_type"], "What terminal connection style does L-PCM06B use?", "paraphrase", "single_hop"),
                _row_from_record(lp["dimensions"], "What physical dimensions are given for L-PCM06B?", "paraphrase", "single_hop"),
            ]
        )
        rows.extend(
            [
                _multi_row(
                    "Which L-PCM06B electrical specifications describe maximum power, rated input voltage, and power tapping?",
                    [lp["max_power"], lp["rated_input_voltage"], lp["power_tapping"]],
                    "The L-PCM06B maximum power rating is 9 W, the rated input voltage is 70 V, and the power tapping settings are 1.5 W, 3 W, and 6 W.",
                    "specification",
                    "L-PCM06B",
                ),
                _multi_row(
                    "Which L-PCM06B details describe where it is used, how it connects, and how large it is?",
                    [lp["usage_locations"], lp["connection_type"], lp["dimensions"]],
                    "The L-PCM06B is designed for locations such as hotels, warehouses, schools, office buildings, stadiums, and restaurants. It uses push-in terminal blocks, and its dimensions are 200 mm x 100 mm.",
                    "usage_environment",
                    "L-PCM06B",
                ),
                _multi_row(
                    "What performance-related L-PCM06B specs are given for frequency response and dispersion angle?",
                    [lp["effective_frequency_response_range"], lp["dispersion_angle"]],
                    "The L-PCM06B effective frequency response range is 110 Hz to 20 kHz, and the dispersion angle is 180 degrees.",
                    "specification",
                    "L-PCM06B",
                ),
            ]
        )

    deduped: list[AutoQRow] = []
    seen: set[str] = set()
    for row in rows:
        key = row.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_autoq_dataframe(records: list[ProductRecord]) -> pd.DataFrame:
    rows = generate_autoq_rows(records)
    payload = []
    for idx, row in enumerate(rows, start=1):
        payload.append(
            {
                "id": idx,
                "question": row.question,
                "ground_truth": row.ground_truth,
                "contexts": json.dumps(row.contexts),
                "category": row.category,
                "query_class": row.query_class,
                "reasoning_type": row.reasoning_type,
                "source_product": row.source_product,
                "source_fields": json.dumps(row.source_fields),
                "source_split": row.source_split,
            }
        )
    return pd.DataFrame(payload)


def main() -> None:
    settings = get_settings()
    records = load_product_records(settings.product_records_path)
    if not records:
        raise FileNotFoundError(
            f"No product records found at {settings.product_records_path}. Run ingestion first."
        )
    df = build_autoq_dataframe(records)
    out_path = Path("data/eval/autoq_honeywell_structured.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Loaded {len(records)} records from -> {settings.product_records_path}")
    print(f"Generated {len(df)} AutoQ questions -> {out_path}")


if __name__ == "__main__":
    main()
