"""Background document indexer.

Polls `knowledge/inbox_docs/` every `scan_interval_s` seconds. Each new
file is:

  1. hashed (sha256) — if the hash already exists in the collection the
     file is treated as a duplicate and moved straight to processed_docs.
  2. extracted -> chunked -> embedded -> upserted into ChromaDB.
  3. moved to `knowledge/processed_docs/`.

Errors are logged but never crash the loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from ..config import CONFIG, KnowledgeConfig
from .extractors import SUPPORTED_SUFFIXES, chunk_text, extract_text
from .store import get_collection, resolve_knowledge_dirs

log = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(
        self,
        cfg: KnowledgeConfig = CONFIG.knowledge,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.base_dir = base_dir
        self.inbox, self.processed, self.vector_dir = resolve_knowledge_dirs(
            cfg, base_dir
        )

    # ---------------------------------------------------------- one-shot pass
    def index_pending(self) -> int:
        """Index every supported file currently in the inbox. Returns count."""
        files = [
            p for p in sorted(self.inbox.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not files:
            return 0
        log.info("Indexer: %d file(s) waiting in %s", len(files), self.inbox)
        collection = get_collection(self.cfg, self.base_dir)
        ok = 0
        for path in files:
            try:
                if self._index_one(path, collection):
                    ok += 1
            except Exception:
                log.exception("Indexer: failed on %s", path.name)
        return ok

    # -------------------------------------------------------------- one file
    def _index_one(self, path: Path, collection) -> bool:
        digest = _sha256_file(path)

        # Dedup: have we already ingested this exact file?
        existing = collection.get(where={"file_hash": digest}, limit=1)
        if existing and existing.get("ids"):
            log.info("Indexer: '%s' already indexed (hash match) — moving on",
                     path.name)
            self._move_to_processed(path)
            return True

        t0 = time.monotonic()
        text = extract_text(path)
        if not text.strip():
            log.warning("Indexer: no extractable text in %s", path.name)
            self._move_to_processed(path)
            return False

        chunks = chunk_text(
            text,
            chunk_chars=self.cfg.chunk_chars,
            overlap_chars=self.cfg.chunk_overlap_chars,
        )
        if not chunks:
            log.warning("Indexer: extractor produced 0 chunks for %s", path.name)
            self._move_to_processed(path)
            return False

        ids = [f"{digest}:{i}" for i in range(len(chunks))]
        metas = [
            {
                "file_hash": digest,
                "source": path.name,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        # ChromaDB will call the OllamaEmbeddingFunction internally.
        collection.upsert(documents=chunks, metadatas=metas, ids=ids)

        log.info(
            "Indexer: '%s' -> %d chunks in %.1fs (hash=%s)",
            path.name, len(chunks), time.monotonic() - t0, digest[:10],
        )
        self._move_to_processed(path)
        return True

    def _move_to_processed(self, path: Path) -> None:
        dest = self.processed / path.name
        if dest.exists():
            # Avoid collisions: append a short hash suffix.
            stem, suffix = path.stem, path.suffix
            dest = self.processed / f"{stem}.{int(time.time())}{suffix}"
        shutil.move(str(path), str(dest))
        log.debug("Indexer: moved %s -> %s", path.name, dest)

    # ---------------------------------------------------- async watcher loop
    async def run(self, stop: asyncio.Event) -> None:
        if not self.cfg.enabled:
            log.info("Indexer disabled via config — not watching %s", self.inbox)
            return
        loop = asyncio.get_running_loop()
        log.info(
            "Indexer watching %s (interval=%.1fs)",
            self.inbox, self.cfg.scan_interval_s,
        )
        # First pass at startup.
        await loop.run_in_executor(None, self.index_pending)
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self.cfg.scan_interval_s
                )
                break  # stop fired
            except asyncio.TimeoutError:
                pass
            try:
                await loop.run_in_executor(None, self.index_pending)
            except Exception:
                log.exception("Indexer scan crashed (continuing)")


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
