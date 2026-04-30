from __future__ import annotations

import re


_CHAR_REPLACEMENTS = {
    "\u013d": "1/4",
    "\u02dd": "1/2",
    "\u00bc": "1/4",
    "\u00bd": "1/2",
    "\u00be": "3/4",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\uf057": " ohm",
    "\u2022": "- ",
    "\u25cf": "- ",
    "\u00b0": " degrees ",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00d7": " x ",
    "\u00d8": "",
    "\u03a6": "",
    "\u2122": "",
    "\u00ae": "",
}

_SPEC_LABEL_PATTERN = re.compile(
    r"^(?:"
    r"(?:rated|max(?:imum)?|nominal|operating|storage|effective|dispersion|frequency|power|voltage|"
    r"impedance|input|output|speaker|strobe|temperature|humidity|dimensions|weight|color|connection|"
    r"capacity|fuse|line input|contact inputs|contact outputs|main power|backup power)"
    r"[\w\s()/.-]*"
    r")$",
    flags=re.IGNORECASE,
)
_VALUE_LINE_PATTERN = re.compile(
    r"(?:\d|vdc|vrms|watt|watts|hz|db|ohm|zones?|inputs?|outputs?|degrees|f\b|c\b)",
    re.IGNORECASE,
)
_MEANINGFUL_SHORT_LINE = re.compile(
    r"(?:\d|rk-zone|l-series|l-pcm|candela|voltage|power|input|output|zones?|speaker|strobe|hotel|warehouse|school|stadium|restaurant)",
    re.IGNORECASE,
)


def normalize_pdf_text(text: str) -> str:
    normalized = text or ""
    for src, dst in _CHAR_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)

    normalized = normalized.replace("offi ce", "office")
    normalized = normalized.replace("profi le", "profile")
    normalized = normalized.replace("wi- fi", "wi-fi")
    normalized = normalized.replace("wi fi", "wi-fi")
    normalized = normalized.replace("of? ce", "office")
    normalized = normalized.replace("pro? le", "profile")
    normalized = normalized.replace("ampli? ers", "amplifiers")
    normalized = re.sub(r"\b([A-Z])\s*-\s*([A-Z0-9])", r"\1-\2", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\r\n?", "\n", normalized)
    lines = [normalize_pdf_line(line) for line in normalized.split("\n")]
    repaired = repair_structured_lines(lines)
    return "\n".join(line for line in repaired if line.strip())


def normalize_pdf_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace(" / ", "/")
    cleaned = cleaned.replace("( ", "(").replace(" )", ")")
    cleaned = re.sub(r"\b([0-9]+)\s*V\s*DC\b", r"\1 VDC", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([0-9]+(?:\.[0-9]+)?)\s*V\b", r"\1 V", cleaned)
    cleaned = re.sub(r"\b([0-9]+(?:\.[0-9]+)?)\s*W\b", r"\1 W", cleaned)
    cleaned = re.sub(r"\b([0-9]+(?:\.[0-9]+)?)\s*Hz\b", r"\1 Hz", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([0-9]+(?:\.[0-9]+)?)\s*degrees\s*F\b", r"\1F", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([0-9]+(?:\.[0-9]+)?)\s*degrees\s*C\b", r"\1C", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+-\s+", " - ", cleaned)
    cleaned = re.sub(r"\bDimensions\s+[^\w\s]?\s*([0-9])", r"Dimensions \1", cleaned, flags=re.IGNORECASE)

    lowered = cleaned.lower()
    if lowered.startswith("up to") and "connected on a system" in lowered:
        match = re.search(
            r"^(Up to\s+\d+\s+[A-Za-z0-9-]+).*?(?:devices?\s+can\s+be\s+)?connected on a system\.?$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if match:
            product_prefix = match.group(1).strip()
            cleaned = f"{product_prefix} devices can be connected on a system."

    if lowered.startswith("rated") and ":" not in cleaned:
        cleaned = re.sub(r"\s{2,}", " ", cleaned)

    if "locations such as" in lowered:
        cleaned = re.sub(r".*?(locations such as .*)", r"\1", cleaned, flags=re.IGNORECASE)

    return cleaned


def repair_structured_lines(lines: list[str]) -> list[str]:
    repaired: list[str] = []
    pending_label: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            pending_label = None
            continue

        if pending_label and _looks_like_value_line(line):
            repaired.append(f"{pending_label} {line}".strip())
            pending_label = None
            continue

        if _looks_like_label_line(line):
            pending_label = line
            continue

        if repaired and _should_merge_with_previous(repaired[-1], line):
            repaired[-1] = f"{repaired[-1]} {line}".strip()
            continue

        split_lines = _split_compound_structured_line(line)
        if len(split_lines) > 1:
            pending_label = None
            repaired.extend(split_lines)
            continue

        pending_label = None
        repaired.append(line)

    if pending_label:
        repaired.append(pending_label)

    return repaired


def is_meaningful_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    if _MEANINGFUL_SHORT_LINE.search(stripped):
        return True
    alpha_ratio = sum(c.isalpha() for c in stripped) / (len(stripped) + 1e-6)
    return alpha_ratio >= 0.25


def _looks_like_label_line(line: str) -> bool:
    return bool(_SPEC_LABEL_PATTERN.match(line)) and not _looks_like_value_line(line)


def _looks_like_value_line(line: str) -> bool:
    return bool(_VALUE_LINE_PATTERN.search(line))


def _should_merge_with_previous(previous: str, current: str) -> bool:
    if previous.endswith((".", ":", ";")):
        return False
    if _looks_like_label_line(previous):
        return True
    if current and current[0].islower():
        return True
    return False


def _split_compound_structured_line(line: str) -> list[str]:
    candidates = [line]
    cue_patterns = [
        r"(?=\bUp to\b)",
        r"(?=\bRated\b)",
        r"(?=\bField selectable\b)",
        r"(?=\bField-selectable\b)",
        r"(?=\bNominal Voltage\b)",
        r"(?=\blocations such as\b)",
    ]
    for pattern in cue_patterns:
        if len(candidates) != 1:
            break
        if re.search(pattern, line, flags=re.IGNORECASE):
            parts = [part.strip(" -") for part in re.split(pattern, line) if part.strip(" -")]
            if len(parts) > 1:
                candidates = [normalize_pdf_line(part) for part in parts if part.strip()]
    return candidates
