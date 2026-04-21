from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.ingest import main as ingest_main
from src.run_eval_graph_rag import main as eval_main


def main() -> None:
    settings = get_settings()
    vectorstore_path = Path(settings.vectorstore_dir)

    if not vectorstore_path.exists():
        print(f"Vector store not found at {settings.vectorstore_dir}. Building index first...")
        ingest_main()
    else:
        print(f"Using existing vector store at {settings.vectorstore_dir}")

    eval_main()


if __name__ == "__main__":
    main()
