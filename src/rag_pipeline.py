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


def load_retriever():
    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "lambda_mult": settings.lambda_mult,
        },
    )


def _keyword_score(question: str, text: str) -> int:
    q = question.lower()
    t = text.lower()
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", q) if len(tok) > 2]
    overlap = sum(1 for tok in set(tokens) if tok in t)

    # Spec-oriented phrase boosts for better exact-value retrieval.
    boosts = 0
    for phrase in ("input power", "operating temperature", "storage temperature", "voltage"):
        if phrase in q and phrase in t:
            boosts += 5
    return overlap + boosts


def _retrieve_docs(question: str):
    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    mmr_docs = vectorstore.max_marginal_relevance_search(
        question,
        k=settings.top_k,
        fetch_k=settings.fetch_k,
        lambda_mult=settings.lambda_mult,
    )
    sim_docs = vectorstore.similarity_search(question, k=settings.fetch_k)

    merged = {}
    for doc in mmr_docs + sim_docs:
        key = (
            str(doc.metadata.get("chunk_id", "")),
            str(doc.metadata.get("source_path", "")),
            doc.page_content[:120],
        )
        if key not in merged:
            merged[key] = doc

    ranked = sorted(
        merged.values(),
        key=lambda d: _keyword_score(question, d.page_content),
        reverse=True,
    )
    return ranked[: settings.top_k]


def answer_question(question: str) -> dict:
    settings = get_settings()
    api_key = require_groq_api_key(settings)
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=api_key,
        temperature=settings.temperature,
    )

    started_at = time.perf_counter()
    retrieved_docs = _retrieve_docs(question)
    if settings.log_retrieved_contexts:
        logger.info("Retrieved %s contexts for query '%s'", len(retrieved_docs), question)
        for idx, doc in enumerate(retrieved_docs, start=1):
            logger.info(
                "ctx_%s doc_id=%s chunk_id=%s source=%s",
                idx,
                doc.metadata.get("doc_id", ""),
                doc.metadata.get("chunk_id", ""),
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
        "retrieved_contexts": [
            {
                "doc_id": doc.metadata.get("doc_id", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "text": doc.page_content,
                "score": doc.metadata.get("score"),
                "source_type": doc.metadata.get("source_type", ""),
                "source_path": doc.metadata.get("source_path", ""),
            }
            for doc in retrieved_docs
        ],
        "latency_ms": latency_ms,
        "model_name": settings.groq_model,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
    }
