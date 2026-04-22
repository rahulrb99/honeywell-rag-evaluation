from __future__ import annotations

from datetime import UTC, datetime
import logging
import re
import time
from uuid import uuid4

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from src.config import get_settings, require_groq_api_key


PROMPT = ChatPromptTemplate.from_template(
    """You are a Honeywell product assistant.
You must answer using ONLY the provided context.

Do NOT use prior knowledge.
Do NOT infer beyond the context.

If the answer is clearly supported by the context, provide it concisely.

Do NOT say "not found" if relevant information is present but phrased differently.

If the context truly does not contain the answer, respond with:
"Not found in provided context."

If the question asks for specs (power, temperature, dimensions, voltage, ranges),
quote exact numeric values and units exactly as written in context.

Important:
Answers are often found after phrases like:
- "Space Use:"
- "Land Applications:"
- "Specifications:"

Retrieved context:
{context}

Question:
{question}

Answer in 3-6 lines with concise, factual wording."""
)

logger = logging.getLogger(__name__)

_VECTORSTORE: FAISS | None = None
_CROSS_ENCODER: CrossEncoder | None = None


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for l in lines:
        l_strip = l.strip()

        # Remove empty or very short lines
        if len(l_strip) < 3:
            continue

        # Remove PDF garbage patterns
        if any(
            x in l_strip.lower()
            for x in [
                "asan",
                "photograph",
                "courtesy",
                "international mars introduction",
            ]
        ):
            continue

        # Remove mostly non-alphanumeric lines
        alpha_ratio = sum(c.isalpha() for c in l_strip) / (len(l_strip) + 1e-6)
        if alpha_ratio < 0.4:
            continue

        cleaned.append(l_strip)

    return "\n".join(cleaned)


def _query_terms(query: str) -> set[str]:
    generic_terms = {
        "what", "are", "the", "and", "for", "with", "from", "that", "this",
        "land", "applications", "application", "space", "use", "uses", "used",
    }
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", query.lower())
        if len(tok) > 2 and tok not in generic_terms
    }


def _is_section_header(line: str) -> bool:
    normalized = line.strip().lower().rstrip(":")
    return (
        normalized in {"space use", "land applications", "specifications"}
        or normalized.startswith("space use:")
        or normalized.startswith("land applications:")
        or normalized.startswith("specifications:")
    )


def _is_product_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("•"):
        return False
    words = re.findall(r"[A-Za-z0-9]+", stripped)
    if len(words) < 2 or len(words) > 7:
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    return sum(ch.isupper() for ch in letters) / len(letters) > 0.75


def _requested_section(query: str) -> str | None:
    q = query.lower()
    if "land application" in q or "land applications" in q:
        return "land applications"
    if "space use" in q:
        return "space use"
    if "spec" in q or "rating" in q or "voltage" in q:
        return "specifications"
    return None


def _extract_section_block(lines: list[str], start_idx: int, section: str | None) -> str:
    if section is None:
        return ""

    section_idx = None
    for idx in range(start_idx, len(lines)):
        normalized = lines[idx].strip().lower().rstrip(":")
        if normalized == section or normalized.startswith(f"{section}:"):
            section_idx = idx
            break
        if idx > start_idx and _is_product_heading(lines[idx]):
            break

    if section_idx is None:
        return ""

    selected = [lines[start_idx], lines[section_idx]] if start_idx != section_idx else [lines[section_idx]]
    for idx in range(section_idx + 1, len(lines)):
        line = lines[idx]
        if _is_section_header(line) or _is_product_heading(line):
            break
        selected.append(line)
        if len(selected) >= 7:
            break
    return "\n".join(line for line in selected if line.strip())


def extract_relevant_lines(query: str, text: str, max_lines: int = 7) -> str:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    terms = _query_terms(query)
    section = _requested_section(query)

    # Prefer the block for a matching product heading, preserving bullets that
    # may not repeat the product name or question words.
    for idx, line in enumerate(lines):
        line_terms = set(re.findall(r"[a-z0-9]+", line.lower()))
        if len(terms & line_terms) >= 2:
            block = _extract_section_block(lines, idx, section)
            if block:
                return block

    # If the chunk is a continuation of the matching product from the prior
    # chunk, keep the requested section rather than dropping answer bullets.
    block = _extract_section_block(lines, 0, section)
    if block:
        return block

    q_words = set(query.lower().split())
    scored = []
    for l in lines:
        l_lower = l.lower()
        score = sum(1 for w in q_words if w in l_lower)
        if score > 0:
            scored.append((score, l))

    scored.sort(reverse=True)

    return "\n".join([l for _, l in scored[:max_lines]])

# ---------------------------------------------------------------------------
# Platform auto-detection for metadata filtering
# ---------------------------------------------------------------------------

_PLATFORM_SIGNALS: dict[str, list[str]] = {
    "hvac": [
        "thermostat", "ventilation", "inncom", "e7", "hvac",
    ],
    "fire_alarm": [
        "alarm", "speaker", "strobe", "communicator", "ltem",
        "zone expander", "l-series", "siren", "public address",
    ],
    "space": [
        "mars rover", "nasa", "spacecraft", "satellite", "missile",
        "telescope", "clipper", "capsule", "proximity sensor",
        "load cell", "rtd", "resolver", "potentiometer",
        "thermostat on the", "series thermostat on",
    ],
}

_DOC_TYPE_SIGNALS: dict[str, list[str]] = {
    "installation": [
        "install", "installation", "setup", "mount", "wiring", "code", "technician"
    ],
    "manual": [
        "instruction", "instructions", "how to", "where should", "can i", "exit mode"
    ],
    "datasheet": [
        "spec", "specs", "specification", "temperature", "voltage", "power",
        "frequency", "rating", "range", "battery life", "dimensions"
    ],
}


def infer_platform_filter(question: str) -> dict[str, str] | None:
    """Return a FAISS metadata filter dict inferred from question keywords.

    Returns None when no signal matches (retrieval searches all documents).
    """
    q = question.lower()
    for platform, signals in _PLATFORM_SIGNALS.items():
        if any(sig in q for sig in signals):
            return {"platform": platform}
    return None


def infer_doc_type_filter(question: str) -> dict[str, str] | None:
    """Infer doc_type filter from question intent (doc_type-only first pass)."""
    q = question.lower()
    for doc_type, signals in _DOC_TYPE_SIGNALS.items():
        if any(sig in q for sig in signals):
            return {"doc_type": doc_type}
    return None


def infer_query_doc_type(question: str) -> str | None:
    q = question.lower()

    if "install" in q or "how to" in q:
        return "installation"

    if "spec" in q or "rating" in q or "voltage" in q:
        return "datasheet"

    return None


def infer_metadata_filter(question: str) -> dict[str, str] | None:
    """Backward-compatible inferred metadata payload (not hard-applied)."""
    doc_type = infer_query_doc_type(question)
    if doc_type:
        return {"doc_type": doc_type}
    return infer_doc_type_filter(question)


# ---------------------------------------------------------------------------
# Vectorstore cache
# ---------------------------------------------------------------------------

def _get_vectorstore() -> FAISS:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    _VECTORSTORE = FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return _VECTORSTORE


# ---------------------------------------------------------------------------
# Cross-encoder reranker (lazy-loaded)
# ---------------------------------------------------------------------------

def _get_cross_encoder() -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def _rerank(question: str, docs: list, top_k: int) -> list:
    if not docs:
        return docs
    ce = _get_cross_encoder()
    scores = ce.predict([(question, doc.page_content) for doc in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def _keyword_score(question: str, text: str) -> float:
    q = question.lower()
    t = text.lower()
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", q) if len(tok) > 2]
    overlap = sum(1 for tok in set(tokens) if tok in t)

    boosts = 0.0
    for phrase in ("input power", "operating temperature", "storage temperature", "voltage"):
        if phrase in q and phrase in t:
            boosts += 5.0
    return float(overlap) + boosts


def doc_type_boost(doc, query_doc_type: str | None) -> float:
    if not query_doc_type:
        return 0.0

    doc_type = doc.metadata.get("doc_type")

    if doc_type == query_doc_type:
        return 1.0  # strong boost

    return 0.0


def final_score(doc, query: str, query_doc_type: str | None) -> float:
    sim_score = doc.metadata.get("score", 0) or 0
    keyword_score = _keyword_score(query, doc.page_content)
    dt_score = doc_type_boost(doc, query_doc_type)

    return (
        sim_score * 1.0 +
        keyword_score * 1.5 +
        dt_score * 1.5
    )


# ---------------------------------------------------------------------------
# Retrieval (staged: similarity → MMR dedupe → cross-encoder rerank)
# ---------------------------------------------------------------------------

def _dedupe(docs: list) -> list:
    seen: set[tuple] = set()
    out = []
    for doc in docs:
        key = (
            str(doc.metadata.get("chunk_id", "")),
            str(doc.metadata.get("source_path", "")),
            doc.page_content[:120],
        )
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def _doc_key(doc) -> tuple[str, str, str]:
    return (
        str(doc.metadata.get("chunk_id", "")),
        str(doc.metadata.get("source_path", "")),
        doc.page_content[:120],
    )


def _faiss_score_to_similarity(score: float | int | None) -> float | None:
    """Convert FAISS distance-like scores to a higher-is-better similarity."""
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return 1.0 / (1.0 + value)


def _annotate_similarity_scores(scored_docs: list[tuple]) -> dict[tuple[str, str, str], float]:
    score_by_key: dict[tuple[str, str, str], float] = {}
    for doc, raw_score in scored_docs:
        similarity = _faiss_score_to_similarity(raw_score)
        if similarity is None:
            continue
        doc.metadata["score"] = similarity
        score_by_key[_doc_key(doc)] = similarity
    return score_by_key


def _apply_known_scores(docs: list, score_by_key: dict[tuple[str, str, str], float]) -> None:
    for doc in docs:
        score = score_by_key.get(_doc_key(doc))
        if score is not None:
            doc.metadata["score"] = score


def _retrieve_docs(
    question: str,
    metadata_filter: dict[str, str] | None = None,
) -> tuple[list, dict[str, str] | None]:
    """Return (ranked_docs, applied_filter).

    Stage 1: wide similarity_search (fetch_k=20) to cast a broad net.
    Stage 2: MMR search (k=10) to reduce redundancy.
    Stage 3: merge, dedupe, then cross-encoder rerank to top_k.
    """
    settings = get_settings()
    vectorstore = _get_vectorstore()

    query_doc_type = infer_query_doc_type(question)
    applied_filter = metadata_filter if metadata_filter is not None else (
        {"doc_type": query_doc_type} if query_doc_type else None
    )
    search_filter = applied_filter or None
    search_kwargs = {"filter": search_filter} if search_filter else {}

    # Broad retrieval first, honoring any explicit or inferred metadata filter.
    scored_sim_docs = vectorstore.similarity_search_with_score(
        question,
        k=settings.fetch_k,
        **search_kwargs,
    )
    sim_docs = [doc for doc, _ in scored_sim_docs]
    score_by_key = _annotate_similarity_scores(scored_sim_docs)
    mmr_docs = vectorstore.max_marginal_relevance_search(
        question,
        k=max(settings.top_k * 2, 10),
        fetch_k=settings.fetch_k,
        lambda_mult=settings.lambda_mult,
        **search_kwargs,
    )
    _apply_known_scores(mmr_docs, score_by_key)

    merged = _dedupe(mmr_docs + sim_docs)
    # Keep semantic ordering from cross-encoder, then apply lightweight final scoring.
    ce_ranked = _rerank(question, merged, top_k=max(settings.top_k * 3, 10))
    docs = sorted(
        ce_ranked,
        key=lambda d: final_score(d, question, query_doc_type),
        reverse=True,
    )

    def is_weak_context(candidates: list) -> bool:
        if not candidates:
            return True
        if not query_doc_type:
            return False
        top = candidates[:3]
        return all(d.metadata.get("doc_type") != query_doc_type for d in top)

    if is_weak_context(docs):
        # fallback: pure similarity search (no bias)
        scored_fallback_docs = vectorstore.similarity_search_with_score(
            question,
            k=settings.top_k,
            **search_kwargs,
        )
        fallback_docs = [doc for doc, _ in scored_fallback_docs]
        _annotate_similarity_scores(scored_fallback_docs)
        docs = fallback_docs + docs

    seen = set()
    deduped = []
    for d in docs:
        key = (d.metadata.get("doc_id"), d.metadata.get("chunk_id"))
        if key not in seen:
            deduped.append(d)
            seen.add(key)

    docs = deduped[: settings.top_k]
    return docs, applied_filter


# ---------------------------------------------------------------------------
# Public retriever (used by LangChain chains if needed)
# ---------------------------------------------------------------------------

def load_retriever():
    settings = get_settings()
    vectorstore = _get_vectorstore()
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "lambda_mult": settings.lambda_mult,
        },
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    metadata_filter: dict[str, str] | None = None,
) -> dict:
    """Run the full RAG pipeline and return a structured response dict.

    Args:
        question: The user question.
        metadata_filter: Optional explicit FAISS filter. When None, the filter
            is auto-inferred from the question via infer_platform_filter().
            Pass an empty dict {} to explicitly disable filtering.
    """
    settings = get_settings()
    api_key = require_groq_api_key(settings)
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=api_key,
        temperature=settings.temperature,
    )

    started_at = time.perf_counter()

    explicit_filter = metadata_filter  # None means "auto-detect"
    retrieved_docs, applied_filter = _retrieve_docs(question, explicit_filter)

    if settings.log_retrieved_contexts:
        logger.info("Retrieved %s contexts for query '%s'", len(retrieved_docs), question)
        for idx, doc in enumerate(retrieved_docs, start=1):
            logger.info(
                "ctx_%s doc_id=%s chunk_id=%s platform=%s source=%s",
                idx,
                doc.metadata.get("doc_id", ""),
                doc.metadata.get("chunk_id", ""),
                doc.metadata.get("platform", ""),
                doc.metadata.get("source_path", ""),
            )
            logger.info("ctx_%s_text=%s", idx, doc.page_content)

    context_docs = []
    for doc in retrieved_docs:
        cleaned_text = clean_text(doc.page_content)
        focused = extract_relevant_lines(question, cleaned_text)
        context_text = focused if focused.strip() else cleaned_text
        context_docs.append((doc, context_text))

    context_str = ""
    for i, (_, context_text) in enumerate(context_docs, 1):
        context_str += f"\nChunk {i}:\n{context_text}\n"

    prompt_value = PROMPT.format_prompt(question=question, context=context_str)
    answer = llm.invoke(prompt_value.to_messages()).content
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "query_id": str(uuid4()),
        "question": question,
        "answer": answer,
        "metadata_filter_used": applied_filter,
        "retrieved_contexts": [
            {
                "doc_id": doc.metadata.get("doc_id", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "text": context_text,
                "score": doc.metadata.get("score"),
                "source_type": doc.metadata.get("source_type", ""),
                "source_path": doc.metadata.get("source_path", ""),
                "platform": doc.metadata.get("platform", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
            }
            for doc, context_text in context_docs
        ],
        "latency_ms": latency_ms,
        "model_name": settings.groq_model,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
    }
