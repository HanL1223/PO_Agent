"""
Index package - OFFLINE store construction.

This package builds and persists the dense and sparse stores. It is
called by scripts/ingest_and_index.py and by tests; it is NOT called
at request time. The runtime retrieval layer lives in src.retrieval.

Layering rule of thumb:
    src.ingestion  -> raw Jira data -> JiraIssue
    src.index      -> JiraIssue -> persisted dense + sparse stores  (OFFLINE)
    src.retrieval  -> stores -> RetrievedChunk for the agent        (ONLINE)
"""

from src.index.bm25_store import BM25Store, bm25_path_for
from src.index.builder import BuildResult, HybridIndexBuilder
from src.index.chunking import issues_to_nodes
from src.index.dense_store import (
    collection_name_for,
    create_dense_store,
    delete_dense_store,
    load_dense_store,
)
from src.index.embeddings import get_embedding_model

__all__ = [
    # embeddings
    "get_embedding_model",
    # chunking
    "issues_to_nodes",
    # dense store
    "collection_name_for",
    "create_dense_store",
    "load_dense_store",
    "delete_dense_store",
    # sparse store
    "BM25Store",
    "bm25_path_for",
    # builder
    "HybridIndexBuilder",
    "BuildResult",
]
