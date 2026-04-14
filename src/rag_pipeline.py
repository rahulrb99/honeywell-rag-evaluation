from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import get_settings


PROMPT = ChatPromptTemplate.from_template(
    """You are a Honeywell product assistant.
Use only the retrieved context to answer.
If the answer is not in context, say you do not have enough information.

Question:
{question}

Retrieved context:
{context}

Answer in 3-6 lines with concise, factual wording."""
)


def load_retriever():
    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})


def answer_question(question: str) -> dict:
    settings = get_settings()
    retriever = load_retriever()
    llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)

    retrieved_docs = retriever.get_relevant_documents(question)
    joined_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt_value = PROMPT.format_prompt(question=question, context=joined_context)
    answer = llm.invoke(prompt_value.to_messages()).content

    return {
        "question": question,
        "answer": answer,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
        "retrieved_sources": [doc.metadata.get("source_path", "") for doc in retrieved_docs],
    }
