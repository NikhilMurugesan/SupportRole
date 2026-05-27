"""Background document indexer.

Polls ``knowledge/inbox_docs/`` for new documents, extracts clean text,
creates semantic chunks with metadata, embeds them in ChromaDB, and moves
new inbox files to ``knowledge/processed_docs/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import CONFIG, KnowledgeConfig
from .extractors import SUPPORTED_SUFFIXES, SemanticChunk, extract_text, semantic_chunk_text
from .rag_policy import infer_source_metadata, is_probably_noisy_chat_log
from .store import get_collection, reset_vector_store, resolve_knowledge_dirs

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
                if self._index_one(path, collection, move_after=True):
                    ok += 1
            except Exception:
                log.exception("Indexer: failed on %s", path.name)
        return ok

    def rebuild_from_sources(
        self,
        source_dirs: Optional[list[Path]] = None,
        *,
        reset: bool = True,
    ) -> int:
        """Rebuild the vector DB from clean source documents without moving them."""
        if reset:
            reset_vector_store(self.cfg, self.base_dir)
        collection = get_collection(self.cfg, self.base_dir)
        roots = source_dirs or [self.processed, self.inbox]
        files: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            files.extend(
                p for p in sorted(root.iterdir())
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        return self._index_files(files, collection)

    def rebuild_files(self, files: list[Path], *, reset: bool = True) -> int:
        """Rebuild the vector DB from an explicit source-file list."""
        if reset:
            reset_vector_store(self.cfg, self.base_dir)
        collection = get_collection(self.cfg, self.base_dir)
        return self._index_files(files, collection)

    def _index_files(self, files: list[Path], collection) -> int:
        ok = 0
        supported = [
            path for path in files
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        for path in supported:
            try:
                if self._index_one(path, collection, move_after=False):
                    ok += 1
            except Exception:
                log.exception("Indexer: rebuild failed on %s", path)
        log.info("Indexer: rebuild complete (%d/%d source files indexed)", ok, len(supported))
        return ok

    # -------------------------------------------------------------- one file
    def _index_one(self, path: Path, collection, *, move_after: bool) -> bool:
        digest = _sha256_file(path)

        existing = collection.get(where={"file_hash": digest}, limit=1)
        if existing and existing.get("ids"):
            log.info("Indexer: '%s' already indexed (hash match) - moving on", path.name)
            if move_after:
                self._move_to_processed(path)
            return True

        t0 = time.monotonic()
        text = extract_text(path)
        if not text.strip():
            log.warning("Indexer: no extractable text in %s", path.name)
            if move_after:
                self._move_to_processed(path)
            return False

        if is_probably_noisy_chat_log(path, text):
            log.warning("Indexer: skipping noisy chat/transcript source %s", path.name)
            if move_after:
                self._move_to_processed(path)
            return False

        chunks = semantic_chunk_text(
            text,
            chunk_chars=self.cfg.chunk_chars,
            overlap_chars=self.cfg.chunk_overlap_chars,
        )
        if not chunks:
            log.warning("Indexer: extractor produced 0 chunks for %s", path.name)
            if move_after:
                self._move_to_processed(path)
            return False

        source_meta = infer_source_metadata(path, text)
        stat = path.stat()
        created_at = _iso_from_timestamp(getattr(stat, "st_ctime", 0.0))
        updated_at = _iso_from_timestamp(getattr(stat, "st_mtime", 0.0))
        unique_chunks = _dedupe_against_collection(collection, chunks)
        if not unique_chunks:
            log.info("Indexer: '%s' has no new unique chunks", path.name)
            if move_after:
                self._move_to_processed(path)
            return True

        ids = [
            f"{digest}:{_chunk_hash(chunk.text)}:{i}"
            for i, chunk in enumerate(unique_chunks)
        ]
        metas = [
            {
                "file_hash": digest,
                "content_hash": _chunk_hash(chunk.text),
                "source": path.name,
                "source_file": path.name,
                "document_type": source_meta["document_type"],
                "topic": source_meta["topic"],
                "project_name": source_meta["project_name"],
                "created_at": created_at,
                "updated_at": updated_at,
                "chunk_id": ids[i],
                "chunk_index": i,
                "chunk_title": chunk.title,
            }
            for i, chunk in enumerate(unique_chunks)
        ]
        collection.upsert(
            documents=[chunk.text for chunk in unique_chunks],
            metadatas=metas,
            ids=ids,
        )

        log.info(
            "Indexer: '%s' -> %d chunks in %.1fs (topic=%s, hash=%s)",
            path.name,
            len(unique_chunks),
            time.monotonic() - t0,
            source_meta["topic"],
            digest[:10],
        )
        if move_after:
            self._move_to_processed(path)
        return True

    def _move_to_processed(self, path: Path) -> None:
        dest = self.processed / path.name
        if dest.exists():
            stem, suffix = path.stem, path.suffix
            dest = self.processed / f"{stem}.{int(time.time())}{suffix}"
        shutil.move(str(path), str(dest))
        log.debug("Indexer: moved %s -> %s", path.name, dest)

    # ---------------------------------------------------- async watcher loop
    async def run(self, stop: asyncio.Event) -> None:
        if not self.cfg.enabled:
            log.info("Indexer disabled via config - not watching %s", self.inbox)
            return
        loop = asyncio.get_running_loop()
        log.info(
            "Indexer watching %s (interval=%.1fs)",
            self.inbox,
            self.cfg.scan_interval_s,
        )
        await loop.run_in_executor(None, self.index_pending)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.cfg.scan_interval_s)
                break
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


def _chunk_hash(text: str) -> str:
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _iso_from_timestamp(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _dedupe_against_collection(collection, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
    out: list[SemanticChunk] = []
    for chunk in chunks:
        content_hash = _chunk_hash(chunk.text)
        try:
            existing = collection.get(where={"content_hash": content_hash}, limit=1)
        except Exception:
            existing = None
        if existing and existing.get("ids"):
            continue
        out.append(chunk)
    return out
