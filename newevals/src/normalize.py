from __future__ import annotations

import re
import string


ALIASES = {
    "bacnet protocol": "bacnet",
    "bacnet/ip": "bacnet",
    "bacnet ip": "bacnet",
    "bacnet mstp": "bacnet",
    "bacnet ms tp": "bacnet",
    "modbus protocol": "modbus",
    "24 volts": "24v",
    "24 volt": "24v",
    "24 v": "24v",
    "24vdc": "24v",
    "24 vac": "24v",
    "24vac": "24v",
}


def normalize_entity(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""

    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(string.whitespace + string.punctuation)
    text = re.sub(r"\bprotocol\b$", "", text).strip()

    voltage_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:v|volt|volts|vac|vdc)(?:\s*(?:ac|dc))?", text)
    if voltage_match:
        number = voltage_match.group(1).rstrip("0").rstrip(".")
        return f"{number}v"

    compact_voltage = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s|-)*(?:v|volt|volts)(?:dc|ac)?", text)
    if compact_voltage:
        number = compact_voltage.group(1).rstrip("0").rstrip(".")
        return f"{number}v"

    text = ALIASES.get(text, text)
    text = text.translate(str.maketrans("", "", "\"'`"))
    text = re.sub(r"[^a-z0-9./#+-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(string.whitespace + string.punctuation)
    return ALIASES.get(text, text)


def normalize_entities(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = normalize_entity(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out

