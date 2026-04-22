from __future__ import annotations

from datetime import UTC, datetime
import logging
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
Use only the retrieved context to answer.
If the answer is not in context, say you do not have enough information.
If the question asks for specs (power, temperature, dimensions, voltage, ranges),
quote exact numeric values and units exactly as written in context.

Question:
{question}

Retrieved context:
{context}

Answer in 3-6 lines with concise, factual wording."""
)

logger = logging.getLogger(__name__)

_VECTORSTORE: FAISS | None = None
_CROSS_ENCODER: CrossEncoder | None = None

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


def infer_platform_filter(question: str) -> dict[str, str] | None:
    """Return a FAISS metadata filter dict inferred from question keywords.

    Returns None when no signal matches (retrieval searches all documents).
    """
    q = question.lower()
    for platform, signals in _PLATFORM_SIGNALS.items():
        if any(sig in q for sig in signals):
            return {"platform": platform}
    return None


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

    filter_ = metadata_filter if metadata_filter is not None else infer_platform_filter(question)

    try:
        sim_docs = vectorstore.similarity_search(
            question, k=20, filter=filter_
        )
        mmr_docs = vectorstore.max_marginal_relevance_search(
            question,
            k=10,
            fetch_k=20,
            lambda_mult=settings.lambda_mult,
            filter=filter_,
        )
    except Exception:
        # FAISS filter may fail if the field doesn't exist in index metadata.
        # Fall back to unfiltered retrieval.
        logger.warning("Metadata filter %s failed, retrying without filter.", filter_)
        filter_ = None
        sim_docs = vectorstore.similarity_search(question, k=20)
        mmr_docs = vectorstore.max_marginal_relevance_search(
            question,
            k=10,
            fetch_k=20,
            lambda_mult=settings.lambda_mult,
        )

    merged = _dedupe(mmr_docs + sim_docs)
    ranked = _rerank(question, merged, top_k=settings.top_k)
    return ranked, filter_


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

    joined_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt_value = PROMPT.format_prompt(question=question, context=joined_context)
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
                "text": doc.page_content,
                "score": doc.metadata.get("score"),
                "source_type": doc.metadata.get("source_type", ""),
                "source_path": doc.metadata.get("source_path", ""),
                "platform": doc.metadata.get("platform", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
            }
            for doc in retrieved_docs
        ],
        "latency_ms": latency_ms,
        "model_name": settings.groq_model,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
    }
