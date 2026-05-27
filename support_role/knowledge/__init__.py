"""Document knowledge system (RAG) for SupportRole.

Public surface:
    * `DocumentIndexer`  — watches `knowledge/inbox_docs/`, extracts text,
      chunks, embeds, stores into a persistent ChromaDB, and moves the
      source file to `knowledge/processed_docs/`.
    * `KnowledgeRetriever` — embeds a query and returns top-K matching
      chunks from the same ChromaDB collection.

Both share the embedding model and collection defined in
`config.KnowledgeConfig`. Embeddings are produced by the local Ollama
server (model `nomic-embed-text` by default).
"""

from .indexer import DocumentIndexer
from .retriever import KnowledgeRetriever, RetrievalResult, RetrievedChunk

__all__ = ["DocumentIndexer", "KnowledgeRetriever", "RetrievalResult", "RetrievedChunk"]
