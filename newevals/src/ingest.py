from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .config import LEGACY_RAW_PDF_DIR, PAGE_RECORDS_PATH, RAW_PDF_DIR, ensure_directories, seed_everything
from .io_utils import write_jsonl


def build_doc_id(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "document"


def resolve_raw_pdf_dir(raw_pdf_dir: Path = RAW_PDF_DIR) -> Path:
    if raw_pdf_dir.exists() and any(raw_pdf_dir.glob("*.pdf")):
        return raw_pdf_dir
    return LEGACY_RAW_PDF_DIR


def load_pdf_folder(raw_pdf_dir: Path = RAW_PDF_DIR) -> list[dict[str, object]]:
    seed_everything()
    raw_pdf_dir = resolve_raw_pdf_dir(raw_pdf_dir)
    if not raw_pdf_dir.exists():
        raise FileNotFoundError(f"PDF folder not found: {raw_pdf_dir}")

    records: list[dict[str, object]] = []
    for pdf_path in sorted(raw_pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower()):
        reader = PdfReader(str(pdf_path))
        doc_id = build_doc_id(pdf_path)
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            records.append(
                {
                    "doc_id": doc_id,
                    "source_document": pdf_path.name,
                    "page": page_index,
                    "text": text,
                }
            )
    return records


def main() -> None:
    ensure_directories()
    records = load_pdf_folder()
    write_jsonl(PAGE_RECORDS_PATH, records)
    print(f"Wrote {len(records)} page records to {PAGE_RECORDS_PATH}")


if __name__ == "__main__":
    main()
