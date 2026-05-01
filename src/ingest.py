from __future__ import annotations

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import get_settings
from src.product_records import extract_product_records, save_product_records
from src.text_normalization import normalize_pdf_text

# Maps lowercased filename-stem substrings to doc_type and platform metadata.
# First matching key wins. Fallback: {"doc_type": "unknown", "platform": "general"}.
_DOC_META: dict[str, dict[str, str]] = {
    "sps-ast-space-products-ebook": {"doc_type": "ebook",     "platform": "space"},
    "ventilation system manual":    {"doc_type": "manual",    "platform": "hvac"},
    "communicator_datasheet":       {"doc_type": "datasheet", "platform": "fire_alarm"},
    "hbt-bms-e7":                   {"doc_type": "installation", "platform": "hvac"},
    "l-series_installation_guide":  {"doc_type": "installation", "platform": "fire_alarm"},
    "l-series_speakers_strobes":    {"doc_type": "datasheet", "platform": "fire_alarm"},
    "zone_expander":                {"doc_type": "datasheet", "platform": "fire_alarm"},
    "public address speakers":      {"doc_type": "datasheet", "platform": "fire_alarm"},
    "firstcommand fire fighter telephone": {"doc_type": "datasheet", "platform": "fire_alarm"},
    "notifier first command remote microphone": {"doc_type": "datasheet", "platform": "fire_alarm"},
    "lsecs info guide":             {"doc_type": "manual", "platform": "fire_alarm"},
    "slc wiring manual":            {"doc_type": "manual", "platform": "fire_alarm"},
    "daa2 amplifier manual":        {"doc_type": "manual", "platform": "fire_alarm"},
    "afp-3030 installation manual": {"doc_type": "installation", "platform": "fire_alarm"},
    "hbt-fire":                     {"doc_type": "datasheet", "platform": "fire_alarm"},
}

_FALLBACK_META: dict[str, str] = {"doc_type": "unknown", "platform": "general"}
_PDF_SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(installation|specifications?|technical specifications|features(?: and benefits)?|functions|description|general|electrical|mechanical|environmental|ordering information|parameters|input/output|approvals?)\s*$",
    flags=re.IGNORECASE,
)


def _resolve_doc_meta(path: Path) -> dict[str, str]:
    stem = path.stem.lower()
    for key, meta in _DOC_META.items():
        if key in stem:
            return meta
    return _FALLBACK_META


def infer_doc_type(path: str) -> str:
    p = path.lower()

    if "installation" in p or "install" in p:
        return "installation"
    elif "datasheet" in p or "data_sheet" in p:
        return "datasheet"
    elif "manual" in p:
        return "manual"
    else:
        return "other"


def is_noise_chunk(text: str) -> bool:
    t = text.lower()

    # Filter table-like or useless chunks
    if "table" in t:
        return True
    if "ul listed" in t:
        return True
    if "figure" in t and len(t.split()) < 30:
        return True

    # Too short to be meaningful, unless it contains dense structured specs
    if len(t.split()) < 15 and not re.search(
        r"(voltage|power|candela|zones?|inputs?|outputs?|rated|temperature|capacity|impedance|l-pcm|rk-zone|l-series|\d)",
        t,
    ):
        return True

    return False


def get_text_splitter(doc_type: str):
    settings = get_settings()
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    if doc_type == "installation":
        return RecursiveCharacterTextSplitter(
            chunk_size=max(chunk_size, 1000),
            chunk_overlap=max(chunk_overlap, 200),
            separators=["\n\n", "\n", "1.", "2.", "3.", ".", " "],
        )
    if doc_type == "datasheet":
        return RecursiveCharacterTextSplitter(
            chunk_size=max(chunk_size, 700),
            chunk_overlap=max(chunk_overlap, 120),
            separators=[
                "\nTechnical Specifications\n",
                "\nSpecifications\n",
                "\nFeatures and Benefits\n",
                "\nFeatures\n",
                "\nElectrical\n",
                "\nMechanical\n",
                "\nEnvironmental\n",
                "\nDescription\n",
                "\nGeneral\n",
                "\n\n",
                "\n",
            ],
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=max(chunk_size, 800),
            chunk_overlap=max(chunk_overlap, 150),
            separators=["\n\n", "\n", ".", " "],
        )


def _split_pdf_into_sections(doc: Document) -> list[Document]:
    """Split a PDF page/chunk into section-aware sub-docs when headings exist.

    Applies only to PDF content to repair common structure loss in OCR/extraction.
    If no known headings are found, returns the original document unchanged.
    """
    text = normalize_pdf_text(str(doc.page_content or ""))
    if not text.strip():
        return [doc]

    lines = text.splitlines()
    hits = [
        idx for idx, line in enumerate(lines)
        if _PDF_SECTION_HEADER_PATTERN.match(line.strip())
    ]
    if not hits:
        return [doc]

    spans: list[tuple[int, int]] = []
    if hits[0] > 0:
        spans.append((0, hits[0]))
    for i, start in enumerate(hits):
        end = hits[i + 1] if i + 1 < len(hits) else len(lines)
        spans.append((start, end))

    out: list[Document] = []
    for section_idx, (start, end) in enumerate(spans):
        section_lines = lines[start:end]
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        if start < hits[0]:
            header = "preamble"
        else:
            header = section_lines[0].strip().lower()
        meta = dict(doc.metadata)
        meta["section_header"] = header
        meta["pdf_section_index"] = section_idx
        out.append(Document(page_content=section_text, metadata=meta))

    return out or [doc]


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
        if suffix == ".pdf":
            sectioned: list[Document] = []
            for d in loaded:
                sectioned.extend(_split_pdf_into_sections(d))
            loaded = sectioned
        cleaned_loaded: list[Document] = []
        for d in loaded:
            d.page_content = normalize_pdf_text(str(d.page_content or ""))
            if not d.page_content.strip():
                continue
            cleaned_loaded.append(d)
        loaded = cleaned_loaded
        if not loaded:
            continue
        doc_id = _build_doc_id(path, root)
        extra_meta = _resolve_doc_meta(path)
        inferred_doc_type = infer_doc_type(str(path))
        for d in loaded:
            d.metadata["doc_id"] = doc_id
            d.metadata["source_path"] = str(path)
            d.metadata["source_type"] = suffix.lstrip(".")
            # Keep existing metadata fields; override doc_type with explicit infer_doc_type.
            d.metadata["doc_type"] = inferred_doc_type or extra_meta["doc_type"]
            d.metadata["platform"] = extra_meta["platform"]
            section_header = str(d.metadata.get("section_header", "")).lower()
            if "spec" in section_header or "electrical" in section_header or "mechanical" in section_header:
                d.metadata["block_type"] = "specification"
            elif "feature" in section_header or "benefit" in section_header:
                d.metadata["block_type"] = "features"
            elif "install" in section_header or "instruction" in section_header:
                d.metadata["block_type"] = "procedure"
            else:
                d.metadata["block_type"] = "general"
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

    product_records = extract_product_records(docs)
    save_product_records(settings.product_records_path, product_records)

    chunks: list[Document] = []
    for doc in docs:
        doc_type = str(doc.metadata.get("doc_type", "other"))
        text_splitter = get_text_splitter(doc_type)
        split_docs = text_splitter.split_documents([doc])
        filtered_docs = [
            chunk for chunk in split_docs
            if not is_noise_chunk(chunk.page_content)
        ]
        chunks.extend(filtered_docs)

    chunks = _annotate_chunks(chunks)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(settings.vectorstore_dir)

    print(f"Indexed {len(chunks)} chunks from {len(docs)} source documents.")
    print(f"Saved vector store at: {settings.vectorstore_dir}")
    print(f"Saved {len(product_records)} product records to: {settings.product_records_path}")


if __name__ == "__main__":
    main()
