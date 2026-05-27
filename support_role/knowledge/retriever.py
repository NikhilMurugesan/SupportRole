"""Query-time retrieval from the document knowledge base."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import CONFIG, KnowledgeConfig
from .rag_policy import topic_allowed
from .store import get_collection

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float
    metadata: dict = field(default_factory=dict)
    rerank_score: float = 0.0
    accepted: bool = True
    reject_reason: str = ""

    @property
    def similarity(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.distance))

    @property
    def topic(self) -> str:
        return str(self.metadata.get("topic") or "")


@dataclass
class RetrievalResult:
    query: str
    candidates: list[RetrievedChunk]
    accepted_chunks: list[RetrievedChunk]
    top_similarity_score: float
    rejected_reason: str = ""

    def debug_top_results(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for chunk in self.candidates:
            rows.append(
                {
                    "source": chunk.source,
                    "topic": chunk.topic,
                    "similarity": round(chunk.similarity, 4),
                    "distance": round(chunk.distance, 4),
                    "accepted": chunk.accepted,
                    "reject_reason": chunk.reject_reason,
                }
            )
        return rows


class KnowledgeRetriever:
    def __init__(
        self,
        cfg: KnowledgeConfig = CONFIG.knowledge,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.base_dir = base_dir
        self._collection = None

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

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        allowed_topics: tuple[str, ...] = (),
        blocked_topics: tuple[str, ...] = (),
    ) -> list[RetrievedChunk]:
        return self.retrieve_with_debug(
            query,
            top_k,
            allowed_topics=allowed_topics,
            blocked_topics=blocked_topics,
        ).accepted_chunks

    def retrieve_with_debug(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        allowed_topics: tuple[str, ...] = (),
        blocked_topics: tuple[str, ...] = (),
    ) -> RetrievalResult:
        if not self.cfg.enabled:
            return RetrievalResult(query=query, candidates=[], accepted_chunks=[], top_similarity_score=0.0)
        q = " ".join((query or "").strip().split())
        if not q:
            return RetrievalResult(query=q, candidates=[], accepted_chunks=[], top_similarity_score=0.0)
        k = top_k or self.cfg.top_k
        candidate_k = max(k * self.cfg.candidate_multiplier, k)
        try:
            collection = self._ensure()
            if collection.count() == 0:
                return RetrievalResult(query=q, candidates=[], accepted_chunks=[], top_similarity_score=0.0)
            res = collection.query(
                query_texts=[q],
                n_results=candidate_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            log.exception("Retriever: query failed")
            return RetrievalResult(query=q, candidates=[], accepted_chunks=[], top_similarity_score=0.0)

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        candidates: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            metadata = dict(meta or {})
            distance = float(dist if dist is not None else 1.0)
            chunk = RetrievedChunk(
                text=doc or "",
                source=str(metadata.get("source_file") or metadata.get("source") or "?"),
                distance=distance,
                metadata=metadata,
            )
            candidates.append(chunk)

        return filter_and_rerank_candidates(
            q,
            candidates,
            top_k=k,
            min_similarity=self.cfg.min_similarity,
            max_distance=self.cfg.max_distance,
            allowed_topics=allowed_topics,
            blocked_topics=blocked_topics,
        )

    @staticmethod
    def format_for_prompt(chunks: list[RetrievedChunk], max_chars: int) -> str:
        """Render accepted chunks as a compact context block for the LLM prompt."""
        if not chunks:
            return ""
        parts: list[str] = []
        used = 0
        for i, chunk in enumerate(chunks, start=1):
            header = f"[{i}]"
            text = chunk.text.strip()
            block = f"{header}\n{text}"
            if used + len(block) > max_chars:
                remaining = max_chars - used - len(header) - 1
                if remaining > 80:
                    parts.append(f"{header}\n{text[:remaining]}...")
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)


def filter_and_rerank_candidates(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int,
    min_similarity: float,
    max_distance: float,
    allowed_topics: tuple[str, ...] = (),
    blocked_topics: tuple[str, ...] = (),
) -> RetrievalResult:
    query_tokens = _keyword_tokens(query)
    top_similarity = max((chunk.similarity for chunk in candidates), default=0.0)

    accepted: list[RetrievedChunk] = []
    for chunk in candidates:
        if chunk.distance > max_distance or chunk.similarity < min_similarity:
            chunk.accepted = False
            chunk.reject_reason = "below similarity threshold"
        elif not topic_allowed(chunk.topic, allowed_topics, blocked_topics):
            chunk.accepted = False
            chunk.reject_reason = "topic mismatch"
        else:
            overlap = _overlap_score(query_tokens, _keyword_tokens(chunk.text))
            topic_boost = 0.08 if allowed_topics and chunk.topic in allowed_topics else 0.0
            chunk.rerank_score = (0.78 * chunk.similarity) + (0.22 * overlap) + topic_boost
            accepted.append(chunk)

    accepted.sort(key=lambda c: c.rerank_score, reverse=True)
    accepted = accepted[:top_k]
    if not accepted:
        rejected = "no candidates returned"
        if candidates:
            rejected = "all candidates rejected by similarity/topic gates"
    else:
        rejected = ""
    return RetrievalResult(
        query=query,
        candidates=candidates,
        accepted_chunks=accepted,
        top_similarity_score=top_similarity,
        rejected_reason=rejected,
    )


def _keyword_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "you",
        "your",
        "with",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "how",
        "why",
        "did",
        "does",
        "are",
        "was",
        "were",
        "from",
        "into",
        "about",
        "between",
    }
    return {
        token
        for token in re.findall(r"\b[a-z0-9][a-z0-9_-]{2,}\b", text.lower())
        if token not in stop
    }


def _overlap_score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)
