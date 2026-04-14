from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import get_settings


def load_documents(raw_dir: str) -> list:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {raw_dir}. Add .txt/.md files under data/raw."
        )

    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_path"] = str(path)
        docs.extend(loaded)
    return docs


def main() -> None:
    settings = get_settings()
    docs = load_documents(settings.raw_data_dir)
    if not docs:
        raise ValueError("No source documents found. Add files to data/raw first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(settings.vectorstore_dir)

    print(f"Indexed {len(chunks)} chunks from {len(docs)} source documents.")
    print(f"Saved vector store at: {settings.vectorstore_dir}")


if __name__ == "__main__":
    main()
