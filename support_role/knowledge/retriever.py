"""Query-time retrieval from the document knowledge base."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import CONFIG, KnowledgeConfig
from .store import get_collection

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float


class KnowledgeRetriever:
    def __init__(
        self,
        cfg: KnowledgeConfig = CONFIG.knowledge,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.base_dir = base_dir
        self._collection = None  # lazy

    def _ensure(self):
        if self._collection is None:
            self._collection = get_collection(self.cfg, self.base_dir)
        return self._collection

    def is_empty(self) -> bool:
        try:
            return self._ensure().count() == 0
        except Exception:
            log.exception("Retriever: count() failed")
            return True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        if not self.cfg.enabled:
            return []
        q = (query or "").strip()
        if not q:
            return []
        k = top_k or self.cfg.top_k
        try:
            collection = self._ensure()
            if collection.count() == 0:
                return []
            res = collection.query(
                query_texts=[q],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            log.exception("Retriever: query failed")
            return []

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is not None and dist > self.cfg.max_distance:
                continue
            out.append(
                RetrievedChunk(
                    text=doc or "",
                    source=(meta or {}).get("source", "?"),
                    distance=float(dist if dist is not None else 0.0),
                )
            )
        return out

    @staticmethod
    def format_for_prompt(chunks: list[RetrievedChunk], max_chars: int) -> str:
        """Render chunks as a compact context block for the LLM prompt."""
        if not chunks:
            return ""
        parts: list[str] = []
        used = 0
        for i, c in enumerate(chunks, start=1):
            # NOTE: deliberately omit the source filename so the LLM cannot
            # quote it back to the user ("according to file X.pdf...").
            # The snippet is presented as plain reference text only.
            header = f"[{i}]"
            block = f"{header}\n{c.text.strip()}"
            if used + len(block) > max_chars:
                remaining = max_chars - used - len(header) - 1
                if remaining > 80:
                    parts.append(f"{header}\n{c.text.strip()[:remaining]}…")
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)
