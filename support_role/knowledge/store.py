"""Shared ChromaDB collection accessor."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import httpx

from ..config import CONFIG, KnowledgeConfig

log = logging.getLogger(__name__)

_collection_cache: dict[str, object] = {}


class _OllamaHttpxEmbeddingFunction:
    """ChromaDB-compatible embedding function backed by Ollama HTTP."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._timeout = timeout

    def __call__(self, input):  # noqa: A002 - chroma's required name
        return self._embed(input)

    def embed_query(self, input):  # noqa: A002
        embs = self._embed(input)
        if isinstance(input, str):
            return embs[0]
        return embs

    def embed_documents(self, input):  # noqa: A002
        return self._embed(input)

    def _embed(self, input):  # noqa: A002
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = list(input)
        if not inputs:
            return []
        resp = httpx.post(
            self._url,
            json={"model": self._model, "input": inputs},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            single = data.get("embedding")
            if single is None:
                raise RuntimeError(f"Ollama embed response missing 'embeddings': {data}")
            embeddings = [single]
        return embeddings

    def name(self) -> str:  # type: ignore[override]
        return f"ollama-httpx:{self._model}"

    @staticmethod
    def is_legacy() -> bool:
        return False


def get_collection(
    cfg: KnowledgeConfig = CONFIG.knowledge,
    base_dir: Optional[Path] = None,
):
    """Open or create the persistent ChromaDB collection."""
    import chromadb

    vector_dir = _resolve_vector_dir(cfg, base_dir)
    key = str(vector_dir.resolve())
    cached = _collection_cache.get(key)
    if cached is not None:
        return cached

    vector_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Opening ChromaDB at %s (collection=%s, embed=%s)",
        vector_dir,
        cfg.collection_name,
        cfg.embed_model,
    )

    client = chromadb.PersistentClient(path=str(vector_dir))
    embed_fn = _OllamaHttpxEmbeddingFunction(
        base_url=CONFIG.llm.base_url,
        model=cfg.embed_model,
    )
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    _collection_cache[key] = collection
    return collection


def reset_vector_store(
    cfg: KnowledgeConfig = CONFIG.knowledge,
    base_dir: Optional[Path] = None,
) -> Path:
    """Delete the local vector DB directory after verifying the target path."""
    vector_dir = _resolve_vector_dir(cfg, base_dir).resolve()
    root_dir = (base_dir if base_dir is not None else Path(cfg.root_dir)).resolve()
    if vector_dir == root_dir or root_dir not in vector_dir.parents:
        raise RuntimeError(f"Refusing to delete unexpected vector path: {vector_dir}")
    if vector_dir.name != cfg.vector_subdir:
        raise RuntimeError(f"Refusing to delete non-vector directory: {vector_dir}")

    _collection_cache.pop(str(vector_dir), None)
    if vector_dir.exists():
        log.warning("Resetting ChromaDB vector store at %s", vector_dir)
        shutil.rmtree(vector_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)
    return vector_dir


def resolve_knowledge_dirs(
    cfg: KnowledgeConfig = CONFIG.knowledge,
    base_dir: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """Return inbox, processed, and vector directories and ensure they exist."""
    base = base_dir if base_dir is not None else Path(cfg.root_dir)
    inbox = base / cfg.inbox_subdir
    processed = base / cfg.processed_subdir
    vector = base / cfg.vector_subdir
    for directory in (inbox, processed, vector):
        directory.mkdir(parents=True, exist_ok=True)
    return inbox, processed, vector


def _resolve_vector_dir(cfg: KnowledgeConfig, base_dir: Optional[Path]) -> Path:
    base = base_dir if base_dir is not None else Path(cfg.root_dir)
    return base / cfg.vector_subdir
