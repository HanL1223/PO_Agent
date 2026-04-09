"""
Embedding model Factory

To construct the embedding model used by both the
indexing pipeline (offline) and the retrieval layer (online).

We use a local HuggingFace model rather than a cloud
embedding API:

  1. Cost: zero per-request charges. Critical when re-indexing thousands
     of tickets, and when retrieval load grows.
  2. Latency: no network round-trip. Embeddings are computed in-process.
  3. Determinism: identical input always produces identical vectors -
     evaluation and debugging are reproducible.
  4. Privacy: ticket text never leaves the deployment boundary, which
     matters for enterprise Jira data.

The default model is BAAI/bge-small-en-v1.5 (384 dims). It is small,
fast, and competitive with much larger models on the MTEB benchmark.
For higher quality at the cost of speed, switch the env var
EMBEDDING_MODEL to bge-base (768) or bge-large (1024) and update
EMBEDDING_DIMENSION accordingly.

Both the indexer and the retriever MUST use the same embedding model.
Mixing models means the query vector lives in a different space from
the document vectors and retrieval becomes random. Always go through
this factory.
"""

from __future__ import annotations

import logging

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import settings

logger = logging.getLogger(__name__)

def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Construct the HF embedding model from settings
     The model is downloaded on first use and cached locally by the
    HuggingFace transformers library; subsequent calls reuse the cache.
    """

    logger.info(

        "embedding model loading",
        extra= {
        "model": settings.embedding.model,
        "dim":settings.embedding.dimension,
        },
    )
    return HuggingFaceEmbedding(model_name = settings.embedding.model)

