from __future__ import annotations

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import get_settings

# Maps lowercased filename-stem substrings to doc_type and platform metadata.
# First matching key wins. Fallback: {"doc_type": "unknown", "platform": "general"}.
_DOC_META: dict[str, dict[str, str]] = {
    "sps-ast-space-products-ebook": {"doc_type": "ebook",     "platform": "space"},
    "ventilation system manual":    {"doc_type": "manual",    "platform": "hvac"},
    "communicator_datasheet":       {"doc_type": "datasheet", "platform": "fire_alarm"},
    "hbt-bms-e7":                   {"doc_type": "manual",    "platform": "hvac"},
    "l-series_installation_guide":  {"doc_type": "manual",    "platform": "fire_alarm"},
    "l-series_speakers_strobes":    {"doc_type": "datasheet", "platform": "fire_alarm"},
    "zone_expander":                {"doc_type": "datasheet", "platform": "fire_alarm"},
    "public address speakers":      {"doc_type": "datasheet", "platform": "fire_alarm"},
    "hbt-fire":                     {"doc_type": "datasheet", "platform": "fire_alarm"},
}

_FALLBACK_META: dict[str, str] = {"doc_type": "unknown", "platform": "general"}


def _resolve_doc_meta(path: Path) -> dict[str, str]:
    stem = path.stem.lower()
    for key, meta in _DOC_META.items():
        if key in stem:
            return meta
    return _FALLBACK_META


def load_documents(raw_dir: str) -> list:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {raw_dir}. Add .txt/.md files under data/raw."
        )

    docs = []
    for path in sorted(root.rglob("*")):
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            continue
        if suffix == ".pdf":
            loader = PDFPlumberLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        doc_id = _build_doc_id(path, root)
        extra_meta = _resolve_doc_meta(path)
        for d in loaded:
            d.metadata["doc_id"] = doc_id
            d.metadata["source_path"] = str(path)
            d.metadata["source_type"] = suffix.lstrip(".")
            d.metadata["doc_type"] = extra_meta["doc_type"]
            d.metadata["platform"] = extra_meta["platform"]
        docs.extend(loaded)
    return docs


def _build_doc_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    normalized = re.sub(r"[^a-z0-9]+", "_", relative.as_posix().lower()).strip("_")
    return normalized or "document"


def _annotate_chunks(chunks: list) -> list:
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id", "document"))
        chunk_index = chunk_counts.get(doc_id, 0)
        chunk.metadata["chunk_id"] = f"{doc_id}#chunk_{chunk_index}"
        chunk_counts[doc_id] = chunk_index + 1
    return chunks


def main() -> None:
    settings = get_settings()
    docs = load_documents(settings.raw_data_dir)
    if not docs:
        raise ValueError("No source documents found. Add files to data/raw first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = _annotate_chunks(splitter.split_documents(docs))

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(settings.vectorstore_dir)

    print(f"Indexed {len(chunks)} chunks from {len(docs)} source documents.")
    print(f"Saved vector store at: {settings.vectorstore_dir}")


if __name__ == "__main__":
    main()
