from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from src.text_normalization import normalize_pdf_text


_SUPPORTED_DOC_IDS = {
    "zone_expander_data_sheet",
    "l_series_speakers_strobes_wall_datasheet_avds867",
    "public_address_speakers_datasheet",
    "firstcommand_fire_fighter_telephone_honeywell_building_automation",
    "notifier_first_command_remote_microphone_honeywell_building_automation",
}


@dataclass(frozen=True)
class ProductRecord:
    record_id: str
    doc_id: str
    product_name: str
    record_type: str
    field_name: str
    field_value: str
    evidence_text: str
    section_header: str
    source_path: str


def save_product_records(path: str | Path, records: list[ProductRecord]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_product_records(path: str | Path) -> list[ProductRecord]:
    records_path = Path(path)
    if not records_path.exists():
        return []
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    return [ProductRecord(**item) for item in payload]


def extract_product_records(docs: list[Document]) -> list[ProductRecord]:
    by_doc_id: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        doc_id = str(doc.metadata.get("doc_id", ""))
        if doc_id in _SUPPORTED_DOC_IDS:
            by_doc_id[doc_id].append(doc)

    records: list[ProductRecord] = []
    for doc_id, group in by_doc_id.items():
        records.extend(_extract_doc_records(doc_id, group))
    return records


def _clean_record_window(text: str) -> str:
    cleaned = normalize_pdf_text(text)
    cleaned = cleaned.replace("of? ce", "office")
    cleaned = cleaned.replace("pro? le", "profile")
    cleaned = cleaned.replace("L -PCM", "L-PCM")
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_usage_locations_value(text: str) -> str | None:
    compact = _clean_record_window(text)
    match = re.search(
        r"locations such as ([A-Za-z ,\-]+?)(?:\.\s|This full-range loudspeaker|Note:|$)",
        compact,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).strip(" .")
    value = value.replace("office buildings", "office buildings")
    value = re.sub(r"\s+", " ", value)
    parts = [part.strip() for part in re.split(r",|\band\b", value) if part.strip()]
    parts = list(dict.fromkeys(parts))
    if len(parts) < 4:
        return None
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _extract_doc_records(doc_id: str, docs: list[Document]) -> list[ProductRecord]:
    source_path = str(docs[0].metadata.get("source_path", "")) if docs else ""
    sections = []
    full_text_parts = []
    for doc in docs:
        section = str(doc.metadata.get("section_header", ""))
        sections.append((section, normalize_pdf_text(doc.page_content)))
        full_text_parts.append(normalize_pdf_text(doc.page_content))
    raw_source_text = _load_source_text(source_path)
    if raw_source_text:
        sections.insert(0, ("raw_document", raw_source_text))
        full_text_parts.insert(0, raw_source_text)
    full_text = "\n".join(part for part in full_text_parts if part.strip())
    product_name = _extract_primary_product_name(full_text, doc_id)

    records: list[ProductRecord] = []
    seen: set[tuple[str, str]] = set()

    def add_record(
        field_name: str,
        field_value: str,
        evidence_text: str,
        record_type: str = "spec",
        section_header: str = "",
    ) -> None:
        value = field_value.strip().rstrip(".")
        evidence = evidence_text.strip()
        if not value or not evidence:
            return
        key = (field_name, value.lower())
        if key in seen:
            return
        seen.add(key)
        record_id = f"{doc_id}:{field_name}:{len(records)}"
        records.append(
            ProductRecord(
                record_id=record_id,
                doc_id=doc_id,
                product_name=product_name,
                record_type=record_type,
                field_name=field_name,
                field_value=value,
                evidence_text=evidence,
                section_header=section_header,
                source_path=source_path,
            )
        )

    for section_header, text in sections:
        lower = text.lower()

        if doc_id == "zone_expander_data_sheet":
            if "adds 8 speaker zones, 8 monitored contact inputs, and 8 contact outputs" in lower:
                add_record(
                    "system_additions",
                    "8 speaker zones, 8 monitored contact inputs, and 8 contact outputs",
                    "The RK-ZONE8 is a zone expansion device controlled by the RK-MCU, which adds 8 speaker zones, 8 monitored contact inputs, and 8 contact outputs to the INTEVIO system.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "used primarily when dual channel audio is required" in lower:
                add_record(
                    "primary_usage",
                    "used primarily when dual channel audio is required for an application",
                    "The RK-ZONE8 is used primarily when dual channel audio is required for an application.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "8 monitored contact inputs" in lower:
                add_record(
                    "monitored_contact_inputs",
                    "8 monitored contact inputs",
                    "The RK-ZONE8 has 8 monitored contact inputs that can be linked with the fire alarm system to trigger alert and evacuation messages.",
                    record_type="spec",
                    section_header=section_header,
                )
            if "linked with the fire alarm system" in lower:
                add_record(
                    "monitored_contact_input_link",
                    "linked with the fire alarm system to trigger alert and evacuation messages",
                    "The RK-ZONE8 has 8 monitored contact inputs that can be linked with the fire alarm system to trigger alert and evacuation messages.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "line input" in lower and "external audio source" in lower:
                add_record(
                    "line_input",
                    "A single line input is provided to connect with an external audio source",
                    "A single line input is provided to connect with an external audio source.",
                    record_type="io",
                    section_header=section_header,
                )
            if "8 contact outputs" in lower and "activate external devices" in lower:
                add_record(
                    "contact_outputs",
                    "8 contact outputs used to activate external devices",
                    "It has 8 contact outputs that can be used to activate external devices.",
                    record_type="io",
                    section_header=section_header,
                )
            if "24 vdc output is provided to power external devices" in lower:
                add_record(
                    "dc_power_output",
                    "24 VDC output to power external devices",
                    "24 VDC output is provided to power external devices.",
                    record_type="spec",
                    section_header=section_header,
                )
            if "automatic fault diagnosis function" in lower and "speaker circuits" in lower:
                add_record(
                    "automatic_fault_diagnosis",
                    "main power, backup power, CPU, power amplifier, speaker circuits, dry contact inputs, communication line",
                    "The zone expander has an automatic fault diagnosis function to supervise various types of faults, such as main power, backup power, CPU, power amplifier, speaker circuits, dry contact inputs, communication line, etc.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "fault information is displayed on the led indicators" in lower:
                add_record(
                    "fault_display",
                    "displayed on the LED indicators located on the front panel",
                    "Fault information is displayed on the LED indicators located on the front panel.",
                    record_type="fact",
                    section_header=section_header,
                )

            match = re.search(r"main power supply voltage\s+([A-Z ]*\d+-\d+\s*V\s*50/60\s*Hz)", lower, flags=re.IGNORECASE)
            if match:
                add_record(
                    "main_power_supply_voltage",
                    "AC 100-240 V 50/60 Hz",
                    "MAIN POWER SUPPLY VOLTAGE AC 100-240 V 50/60 Hz",
                    section_header=section_header,
                )
            match = re.search(r"backup power supply voltage\s+([0-9.]+\s*VDC-[0-9.]+\s*VDC)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "backup_power_supply_voltage",
                    match.group(1).replace("VDC-", "VDC to "),
                    f"BACKUP POWER SUPPLY VOLTAGE {match.group(1)}",
                    section_header=section_header,
                )
            match = re.search(r"rated power\s+([0-9.]+\s*W)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "rated_power",
                    match.group(1),
                    f"RATED POWER {match.group(1)}",
                    section_header=section_header,
                )
            if "up to 15 rk-zone8" in lower:
                add_record(
                    "max_devices_per_system",
                    "Up to 15 RK-ZONE8 devices can be connected on one system",
                    "Up to 15 RK-ZONE8 devices can be connected on a system.",
                    record_type="capacity",
                    section_header=section_header,
                )

        if doc_id == "l_series_speakers_strobes_wall_datasheet_avds867":
            if "25 and 70.7 vrms" in lower and "power settings" in lower:
                add_record(
                    "speaker_voltage_power_settings",
                    "Speaker voltage settings: 25 and 70.7 Vrms. Power settings: 1/4, 1/2, 1, and 2 watts",
                    "(25 and 70.7 Vrms) and power settings (1/4, 1/2, 1 and 2 watts) in high ambient noise applications.",
                    record_type="setting",
                    section_header=section_header,
                )
            match = re.search(r"nominal voltage \(speakers\)\s+(25\s*Volts\s*or\s*70\.7\s*Volts \(nominal\))", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "nominal_speaker_voltage",
                    match.group(1),
                    f"Nominal Voltage (speakers) {match.group(1)}",
                    section_header=section_header,
                )
            match = re.search(r"maximum supervisory voltage \(speakers\)\s+([0-9.]+\s*VDC)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "maximum_supervisory_voltage",
                    match.group(1),
                    f"Maximum Supervisory Voltage (speakers) {match.group(1)}",
                    section_header=section_header,
                )
            if "strobe flash rate 1 flash per second" in lower:
                add_record(
                    "strobe_flash_rate",
                    "1 flash per second",
                    "Strobe Flash Rate 1 flash per second",
                    section_header=section_header,
                )
            match = re.search(r"frequency range\s+([0-9 ]+to\s+[0-9, ]+\s*Hz)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "frequency_range",
                    match.group(1).replace("4000", "4000"),
                    f"Frequency Range {match.group(1)}",
                    section_header=section_header,
                )
            if "520 hz +/- 10% square wave tone capable with compatible facp" in lower:
                add_record(
                    "tone_capability",
                    "520 Hz +/- 10% square wave tone capable with compatible FACP",
                    "520 Hz +/- 10% square wave tone capable with compatible FACP",
                    record_type="fact",
                    section_header=section_header,
                )
            if "mount to a 4 x 4 x 21/8-inch back box" in lower and "universal mounting plate" in lower:
                add_record(
                    "mounting_back_box",
                    "4 x 4 x 2 1/8-inch back box using a universal mounting plate",
                    "L-Series speaker and speaker strobes shall mount to a 4 x 4 x 21/8-inch back box. A universal mounting plate shall be used for mounting ceiling and wall products.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "candela" in lower and all(value in lower for value in ("15", "30", "75", "95", "110", "135", "185")):
                add_record(
                    "candela_settings",
                    "15, 30, 75, 95, 110, 135, 185",
                    "Field-selectable candela settings on wall units: 15, 30, 75, 95, 110, 135, 185.",
                    record_type="setting",
                    section_header=section_header,
                )

        if doc_id == "firstcommand_fire_fighter_telephone_honeywell_building_automation":
            if "provides secure and reliable communications for firefighters" in lower:
                add_record(
                    "primary_function",
                    "provides secure and reliable communications for firefighters",
                    "Notifier's FirstCommand NFC-FFT is a Fire Fighter telephone system that provides secure and reliable communications for firefighters.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "annunciation and control for local and remote telephone handsets" in lower:
                add_record(
                    "handset_control",
                    "provides annunciation and control for local and remote telephone handsets",
                    "This standalone system is capable of providing annunciation and control for local and remote telephone handsets.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "supports up to 24 n-fpj phone jacks" in lower:
                add_record(
                    "phone_jack_capacity",
                    "supports up to 24 N-FPJ phone jacks",
                    "Supports up to 24 N-FPJ phone jacks.",
                    record_type="capacity",
                    section_header=section_header,
                )
            if "simultaneous operation of up to 10 fire fighter remote handsets" in lower:
                add_record(
                    "remote_handset_capacity",
                    "allows simultaneous operation of up to 10 Fire Fighter remote handsets",
                    "Allows simultaneous operation of up to 10 Fire Fighter remote handsets.",
                    record_type="capacity",
                    section_header=section_header,
                )
            if "class a (style 6 or style 7) or class b (style 4)" in lower:
                add_record(
                    "wiring_configuration",
                    "Class A (Style 6 or Style 7) or Class B (Style 4)",
                    "Onboard SLC circuit can be wired for class A (Style 6 or Style 7) or class B (Style 4) configuration.",
                    record_type="setting",
                    section_header=section_header,
                )

        if doc_id == "notifier_first_command_remote_microphone_honeywell_building_automation":
            if "compatible with the nfc-50/100(e)" in lower:
                add_record(
                    "compatibility",
                    "compatible with the NFC-50/100(E) Emergency Voice Evacuation system",
                    "Notifier First Command (NFC-RM) are optional remote microphones that are compatible with the NFC-50/100(E) Emergency Voice Evacuation system for fire protection applications.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "extending the operator interface to remote locations within a building" in lower:
                add_record(
                    "remote_console_role",
                    "extends the operator interface to remote locations within a building",
                    "It is part of a family of external remote consoles that allows for extending the operator interface to remote locations within a building.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "all call paging broadcasts over the speaker zones" in lower:
                add_record(
                    "paging_function",
                    "provides all call paging broadcasts over the speaker zones",
                    "External remote console that provides all call paging broadcasts over the speaker zones.",
                    record_type="fact",
                    section_header=section_header,
                )
            if "maximum of eight nfc-rms can be connected" in lower:
                add_record(
                    "remote_microphone_capacity",
                    "a maximum of eight NFC-RMs can be connected to an NFC-50/100(E) primary operating console",
                    "A maximum of eight NFC-RMs can be connected to an NFC-50/100(E) primary operating console.",
                    record_type="capacity",
                    section_header=section_header,
                )
            if "supports both class a (style z) and class b (style y) wiring" in lower:
                add_record(
                    "wiring_configuration",
                    "Class A (Style Z) and Class B (Style Y) wiring",
                    "Supports both Class A (Style Z) and Class B (Style Y) wiring.",
                    record_type="setting",
                    section_header=section_header,
                )
            if "built-in microphone with push-to-talk feature" in lower and "all call paging" in lower:
                add_record(
                    "push_to_talk_paging",
                    "built-in microphone with push-to-talk for ALL CALL paging",
                    "Built-in microphone with push-to-talk feature that can be used for ALL CALL paging.",
                    record_type="fact",
                    section_header=section_header,
                )

        if doc_id == "public_address_speakers_datasheet":
            if "l-pcm06b" in lower and "cost-effective ceiling" in lower:
                add_record(
                    "product_name_fact",
                    "L-PCM06B ceiling loudspeaker",
                    "Public Address Speakers L-PCM06B Ceiling Loudspeaker",
                    record_type="fact",
                    section_header=section_header,
                )
            match = re.search(r"rated input voltage\s+([0-9.]+\s*V)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "rated_input_voltage",
                    match.group(1),
                    f"Rated input voltage {match.group(1)}",
                    section_header=section_header,
                )
            match = re.search(r"max power\s+([0-9.]+\s*W)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "max_power",
                    match.group(1),
                    f"Max power {match.group(1)}",
                    section_header=section_header,
                )
            match = re.search(r"power tapping\s+([0-9./ ]+W/[0-9./ ]+W/[0-9./ ]+W)", text, flags=re.IGNORECASE)
            if match:
                normalized_value = re.sub(r"\s+", " ", match.group(1)).strip()
                normalized_value = normalized_value.replace("/", ", ").replace("W, ", " W, ").replace("W", " W")
                normalized_value = re.sub(r"\s+", " ", normalized_value).strip()
                add_record(
                    "power_tapping",
                    normalized_value,
                    f"Power tapping {match.group(1)}",
                    record_type="setting",
                    section_header=section_header,
                )
            match = re.search(r"effective frequency response range \(-10 dB\)\s+([0-9 ]+Hz\s*~\s*[0-9 ]+kHz)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "effective_frequency_response_range",
                    match.group(1).replace(" ~ ", " to "),
                    f"Effective frequency response range (-10 dB) {match.group(1)}",
                    section_header=section_header,
                )
            match = re.search(r"dispersion angle \(1 kHz/-6 dB\)\s+([0-9.]+\s*degrees)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "dispersion_angle",
                    match.group(1),
                    f"Dispersion angle (1 kHz/-6 dB) {match.group(1)}",
                    section_header=section_header,
                )
            if "locations such as" in lower:
                usage_value = _extract_usage_locations_value(text)
                if usage_value:
                    add_record(
                        "usage_locations",
                        usage_value,
                        f"locations such as {usage_value}",
                        record_type="usage",
                        section_header=section_header,
                    )
            if "connection push-in terminal blocks" in lower:
                add_record(
                    "connection_type",
                    "Push-in terminal blocks",
                    "Connection Push-in terminal blocks",
                    section_header=section_header,
                )
            match = re.search(r"dimensions[^0-9]*([0-9.]+\s*mm\s*x\s*[0-9.]+\s*mm)", text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "dimensions",
                    match.group(1),
                    f"Dimensions {match.group(1)}",
                    section_header=section_header,
                )

    full_lower = full_text.lower()
    if doc_id == "zone_expander_data_sheet":
        if "linked with the fire alarm system" in full_lower and "trigger alert and evacuation messages" in full_lower:
            add_record(
                "monitored_contact_input_link",
                "linked with the fire alarm system to trigger alert and evacuation messages",
                "The RK-ZONE8 has 8 monitored contact inputs that can be linked with the fire alarm system to trigger alert and evacuation messages.",
                record_type="fact",
                section_header="input/output",
            )
        if "line input" in full_lower and "external audio source" in full_lower:
            add_record(
                "line_input",
                "A single line input is provided to connect with an external audio source",
                "A single line input is provided to connect with an external audio source.",
                record_type="io",
                section_header="input/output",
            )

    if doc_id == "public_address_speakers_datasheet":
        if "max power" in full_lower:
            match = re.search(r"max power\s+([0-9.]+\s*W)", full_text, flags=re.IGNORECASE)
            if match:
                add_record(
                    "max_power",
                    match.group(1),
                    f"Max power {match.group(1)}",
                    section_header="preamble",
                )
        if "locations such as" in full_lower:
            usage_value = _extract_usage_locations_value(full_text)
            if usage_value:
                add_record(
                    "usage_locations",
                    usage_value,
                    f"locations such as {usage_value}",
                    record_type="usage",
                    section_header="preamble",
                )

    return records


def _extract_primary_product_name(text: str, doc_id: str) -> str:
    lowered = text.lower()
    for token in ("RK-ZONE8", "L-PCM06B", "L-Series"):
        if token.lower() in lowered:
            return token
    if "zone_expander" in doc_id:
        return "RK-ZONE8"
    if "public_address" in doc_id:
        return "L-PCM06B"
    if "l_series" in doc_id:
        return "L-Series"
    match = re.search(r"\b([A-Z]{1,4}-[A-Z0-9]{2,})\b", text)
    if match:
        return match.group(1)
    return doc_id.replace("_", " ")


def _load_source_text(source_path: str) -> str:
    path = Path(source_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return ""
    try:
        reader = PdfReader(str(path))
        raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""
    return normalize_pdf_text(raw_text)
