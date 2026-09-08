from __future__ import annotations

import re

from .config import CHUNK_OVERLAP, CHUNK_TOKENS, CHUNKS_PATH, PAGE_RECORDS_PATH, seed_everything
from .io_utils import read_jsonl, write_jsonl


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_page_records(
    page_records: list[dict[str, object]],
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, object]]:
    seed_everything()
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("overlap must be >= 0 and smaller than chunk_tokens")

    rows = sorted(
        page_records,
        key=lambda r: (str(r.get("doc_id", "")), int(r.get("page", 0)), str(r.get("text", ""))[:40]),
    )
    chunks: list[dict[str, object]] = []
    per_doc_counts: dict[str, int] = {}

    for row in rows:
        doc_id = str(row["doc_id"])
        words = _tokens(str(row.get("text", "")))
        if not words:
            continue
        step = chunk_tokens - overlap
        for start in range(0, len(words), step):
            window = words[start : start + chunk_tokens]
            if not window:
                continue
            chunk_index = per_doc_counts.get(doc_id, 0)
            per_doc_counts[doc_id] = chunk_index + 1
            chunks.append(
                {
                    "chunk_id": f"{doc_id}#chunk_{chunk_index:04d}",
                    "doc_id": doc_id,
                    "source_document": row.get("source_document", ""),
                    "page_start": int(row.get("page", 0)),
                    "page_end": int(row.get("page", 0)),
                    "text": " ".join(window),
                }
            )
            if start + chunk_tokens >= len(words):
                break
    return chunks


def main() -> None:
    pages = read_jsonl(PAGE_RECORDS_PATH)
    chunks = chunk_page_records(pages)
    write_jsonl(CHUNKS_PATH, chunks)
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()

