"""Shared ChromaDB collection accessor.

Both `DocumentIndexer` and `KnowledgeRetriever` need the same persistent
collection with the same embedding function, so we centralise it here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx

from ..config import CONFIG, KnowledgeConfig

log = logging.getLogger(__name__)

_collection_cache: dict[str, object] = {}


class _OllamaHttpxEmbeddingFunction:
    """ChromaDB-compatible embedding function backed by raw httpx calls.

    Uses Ollama's `/api/embed` endpoint (batch-capable). We avoid the
    `chromadb.utils.embedding_functions.OllamaEmbeddingFunction` shipped
    with ChromaDB because it requires the separate `ollama` PyPI package.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._timeout = timeout

    # ChromaDB calls this as a function; signature `(input) -> embeddings`.
    def __call__(self, input):  # noqa: A002 - chroma's required name
        return self._embed(input)

    # ChromaDB >=0.5 query path goes through embed_query / embed_documents
    # instead of __call__. Provide both so add(), upsert(), and query() all
    # work without falling back to the missing-attribute error.
    def embed_query(self, input):  # noqa: A002
        embs = self._embed(input)
        # When chroma passes a single string it expects a single embedding.
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
        # `/api/embed` accepts either a string or a list under "input".
        resp = httpx.post(
            self._url,
            json={"model": self._model, "input": inputs},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            # Older Ollama (<0.2) returned a single "embedding".
            single = data.get("embedding")
            if single is None:
                raise RuntimeError(f"Ollama embed response missing 'embeddings': {data}")
            embeddings = [single]
        return embeddings

    # Chroma >=0.5 uses these for telemetry / cache keys.
    def name(self) -> str:  # type: ignore[override]
        return f"ollama-httpx:{self._model}"

    @staticmethod
    def is_legacy() -> bool:
        return False


def get_collection(
    cfg: KnowledgeConfig = CONFIG.knowledge,
    base_dir: Optional[Path] = None,
):
    """Open (or create) the persistent ChromaDB collection.

    Caches per `vector_dir` so repeated calls return the same client.
    """
    import chromadb

    vector_dir = _resolve_vector_dir(cfg, base_dir)
    key = str(vector_dir.resolve())
    cached = _collection_cache.get(key)
    if cached is not None:
        return cached

    vector_dir.mkdir(parents=True, exist_ok=True)
    log.info("Opening ChromaDB at %s (collection=%s, embed=%s)",
             vector_dir, cfg.collection_name, cfg.embed_model)

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


def _resolve_vector_dir(cfg: KnowledgeConfig, base_dir: Optional[Path]) -> Path:
    base = base_dir if base_dir is not None else Path(cfg.root_dir)
    return base / cfg.vector_subdir


def resolve_knowledge_dirs(
    cfg: KnowledgeConfig = CONFIG.knowledge,
    base_dir: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """Return (inbox_dir, processed_dir, vector_dir) and ensure they exist."""
    base = base_dir if base_dir is not None else Path(cfg.root_dir)
    inbox = base / cfg.inbox_subdir
    processed = base / cfg.processed_subdir
    vector = base / cfg.vector_subdir
    for d in (inbox, processed, vector):
        d.mkdir(parents=True, exist_ok=True)
    return inbox, processed, vector



def _resolve_vector_dir(cfg: KnowledgeConfig, base_dir: Optional[Path]) -> Path:
    base = base_dir if base_dir is not None else Path(cfg.root_dir)
    return base / cfg.vector_subdir


def resolve_knowledge_dirs(
    cfg: KnowledgeConfig = CONFIG.knowledge,
    base_dir: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """Return (inbox_dir, processed_dir, vector_dir) and ensure they exist."""
    base = base_dir if base_dir is not None else Path(cfg.root_dir)
    inbox = base / cfg.inbox_subdir
    processed = base / cfg.processed_subdir
    vector = base / cfg.vector_subdir
    for d in (inbox, processed, vector):
        d.mkdir(parents=True, exist_ok=True)
    return inbox, processed, vector
