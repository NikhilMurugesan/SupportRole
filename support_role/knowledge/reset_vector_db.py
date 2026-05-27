"""Reset and rebuild the local RAG vector database.

Usage:
    python -m support_role.knowledge.reset_vector_db --rebuild
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import CONFIG
from .extractors import SUPPORTED_SUFFIXES
from .indexer import DocumentIndexer
from .store import reset_vector_store

log = logging.getLogger(__name__)


DEFAULT_CLEAN_NAME_TERMS = (
    "avathon",
    "resource_allocation",
    "gts_agentic",
    "agenticai",
    "agentic ai",
    "jd_genai",
    "resume",
)

DEFAULT_SKIP_NAME_TERMS = (
    "complete cheat sheet",
    "completeml",
    "faang dsa",
    "machine learning interview book",
    "phase 1_",
    "phase 2_",
    "phase 3_",
    "phase 4_",
    "phase 5_",
    "phase 6_",
    "phase 7_",
    "pyspark",
    "system design",
    "data_science",
    "data science",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset/rebuild the local Chroma vector DB.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from clean source docs after reset.")
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Also embed broad generic interview/reference books.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected source files without deleting or embedding anything.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    indexer = DocumentIndexer()
    source_files = _select_source_files(indexer, include_generic=args.include_generic)

    log.info("Selected %d clean source file(s):", len(source_files))
    for path in source_files:
        log.info("  - %s", path)

    if args.dry_run:
        return

    if not args.rebuild:
        reset_vector_store(CONFIG.knowledge)
        log.info("Vector DB reset complete. Re-run with --rebuild to embed clean sources.")
        return

    indexer.rebuild_files(source_files, reset=True)


def _select_source_files(indexer: DocumentIndexer, *, include_generic: bool) -> list[Path]:
    roots = [indexer.processed, indexer.inbox]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            name = path.name.lower()
            if include_generic:
                files.append(path)
                continue
            if any(term in name for term in DEFAULT_SKIP_NAME_TERMS):
                continue
            if any(term in name for term in DEFAULT_CLEAN_NAME_TERMS):
                files.append(path)
    return files


if __name__ == "__main__":
    main()
