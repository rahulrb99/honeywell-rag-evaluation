"""LLM-based TTL ontology generation and validation for the GraphRAG domain builder."""

from __future__ import annotations

import re

from rdflib import Graph


_SYSTEM_PROMPT = """You are an ontology engineer specialising in industrial product documentation.
Given product datasheets, generate a Turtle (TTL) RDF ontology that captures:

1. Product types as OWL classes  (e.g. Speaker, ZoneExpander, Amplifier, RemoteMicrophone)
2. Specification values as OWL data properties (e.g. hasVoltage, hasPower, hasFrequencyRange)
3. Inter-product relationships as OWL object properties (e.g. compatibleWith, mountsWith, connectsTo)
4. Application contexts as subclasses of Application

Rules:
- Output ONLY valid Turtle syntax — no markdown fences, no prose.
- Start with @prefix declarations for at least: owl, rdf, rdfs, and a local namespace.
- Use CamelCase for class names and lowerCamelCase for property names."""

_USER_TEMPLATE = """Generate a Turtle ontology for the following Honeywell product documents.
Focus on fire safety, public address, and building automation products.

{document_block}

Output only valid TTL."""


def call_prompt_1a(client, model_name: str, document_block: str) -> str:
    """Call the LLM to generate a TTL ontology from the document block."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(document_block=document_block[:6000])},
        ],
        temperature=0,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


_STANDARD_PREFIXES = """\
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dc:   <http://purl.org/dc/elements/1.1/> .
"""


def validate_ttl(ttl: str) -> tuple[bool, str]:
    """Parse TTL with rdflib.

    Returns (True, cleaned_ttl) on success, (False, error_message) on failure.
    Strips markdown code fences if the LLM wrapped the output in them.
    Auto-injects standard prefix declarations that the LLM may have omitted.
    """
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", ttl.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    patched = _inject_missing_prefixes(cleaned)
    try:
        g = Graph()
        g.parse(data=patched, format="turtle")
        return True, patched
    except Exception as exc:
        return False, str(exc)


def _inject_missing_prefixes(ttl: str) -> str:
    """Prepend any standard @prefix lines that are used in the TTL but not declared."""
    needed = []
    for line in _STANDARD_PREFIXES.strip().splitlines():
        # Extract the prefix token, e.g. "owl" from "@prefix owl: ..."
        m = re.match(r"@prefix\s+(\w+):", line)
        if not m:
            continue
        prefix = m.group(1)
        # Only inject if the prefix is actually referenced and not already declared
        if re.search(rf"\b{re.escape(prefix)}:", ttl) and not re.search(
            rf"@prefix\s+{re.escape(prefix)}\s*:", ttl
        ):
            needed.append(line)
    if not needed:
        return ttl
    return "\n".join(needed) + "\n" + ttl
