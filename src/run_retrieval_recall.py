"""Compatibility wrapper for retrieval metrics.

Use `python -m src.run_eval_retrieval_only` for the canonical evaluator.
"""

from __future__ import annotations

from src.run_eval_retrieval_only import main


if __name__ == "__main__":
    main()
