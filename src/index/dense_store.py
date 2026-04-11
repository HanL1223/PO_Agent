"""
Creat  persist and load the desnes half for hybrid index

- ChromaDB runs in-process (no separate server). Persistence is a
  local directory; in AWS deployment that becomes an EFS mount so the
  index survives container restarts.
- One ChromaDB collection per Jira space. Naming convention:
      {settings.chroma.collection_prefix}_{space_key.lower()}
  e.g. "jira_csci", "jira_edp". This enables both targeted ("CSCI only")
  and cross-project retrieval - the retriever just queries multiple
  collections and re-merges.
- Cosine similarity (hnsw:space=cosine) is configured at collection
  creation time. This matches what BGE embeddings are trained for;
  using L2 with cosine-trained vectors degrades retrieval quality.
- Idempotent upsert: nodes carry stable content-derived IDs (set in
  src/index/chunking.py), so re-running indexing updates rows in place
  rather than duplicating them.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import settings
from src.index.embeddings import get_embedding_model

logger = logging.getLogger(__name__)



def collection_name_for(space_key: str) -> str:
    """Stable, lowercased ChromaDB collection name for a Jira space."""
    return f"{settings.chroma.collection_prefix}_{space_key.lower()}"


def _get_chroma_client(persist_dir: Optional[str] = None) -> chromadb.PersistentClient:
    """Construct a ChromaDB persistent client rooted at persist_dir."""
    persist_dir = persist_dir or settings.chroma.persist_dir
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def create_dense_store(
        nodes:list[TextNode],
        collection_name:str,
        persist_dir:Optional[str] = None
) -> VectorStoreIndex:
    """
    Build or upsert into a chormadb backe  desne vector store

    Args:
        nodes: TextNodes to index. Already chunked, with stable IDs.
        collection_name: ChromaDB collection name. Use collection_name_for().
        persist_dir: Override settings.chroma.persist_dir for tests.

    Returns:
        The VectorStoreIndex object. The caller can discard it once
        the on-disk store is written; the retrieval layer reloads it
        from disk via load_dense_store().
    """
    chroma_client = _get_chroma_client(persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(
        name = collection_name,
        metadata = {'hnsw:space':'cosine'},
    )

    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    embed_model = get_embedding_model()

    index = VectorStoreIndex(
        nodes = nodes,
        storage_context = storage_context,
        embed_model = embed_model,
        show_progress = False
    )

    logger.info(
        "dense_store_created",
        extra = {
            "collection":collection_name,
            "nodes": len(nodes),
            "persist_dir":persist_dir or settings.chroma.persist_dir
        },
    )
    return index


def load_dense_store(
        collection_name:str,
        persist_dir:Optional[str] = None,
) -> VectorStoreIndex:
    """
    Load a previously-built dense store from disk for read-only serving.

    In production the store is built once (offline, by the indexing job
    in CI/CD or a cron) and then loaded at server startup. This avoids
    re-embedding all documents on every container restart.

    Raises:
        chromadb errors if the collection does not exist. The retrieval
        layer is responsible for catching these and degrading gracefully
        (e.g. logging "space not loaded" and falling back to BM25-only).
    """
    chroma_client = _get_chroma_client(persist_dir)
    chroma_collection = chroma_client.get_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embedding_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    logger.info(
        "dense_store_loaded",
        extra={
            "collection": collection_name,
            "persist_dir": persist_dir or settings.chroma.persist_dir,
        },
    )
    return index

def delete_dense_store(
    collection_name: str,
    persist_dir: Optional[str] = None,
) -> None:
    """
    Delete a ChromaDB collection. Used by tests and full-rebuild scripts.

    Silent if the collection does not exist - we treat deletion as
    idempotent for ergonomics.
    """
    chroma_client = _get_chroma_client(persist_dir)
    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info("dense_store_deleted", extra={"collection": collection_name})
    except Exception as e:
        logger.warning(
            "dense_store_delete_failed",
            extra={"collection": collection_name, "error": str(e)},
        )