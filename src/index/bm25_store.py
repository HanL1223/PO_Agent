"""
BM25 Sparse Store

build p

"""


from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from llama_index.core.schema import TextNode
from rank_bm25 import BM25Okapi

from src.config import settings
from src.core.models import RetrievedChunk

logger = logging.getLogger(__name__)



#Tokenisation
_TOKEN_RE = re.compile(r"\w+",re.UNICODE)

def _tokenise(text:str) -> list[str]:
    """Lowercased alphanumeric word tokenisation suitable for BM25."""
    return _TOKEN_RE.findall(text.lower())


#BM25 Store
@dataclass
class BM25Store:
    """
    In-memory BM25 sparse store with disk persistence.

    Holds three parallel arrays plus the fitted BM25 model:
        - corpus_texts:    the original chunk text (returned to caller)
        - corpus_metadata: the per-chunk metadata dicts
        - node_ids:        stable IDs matching the dense store
        - bm25:            the fitted BM25Okapi instance

    The arrays are kept in lockstep: index `i` in any of them refers to
    the same chunk.
    """
    corpus_texts: list[str] = field(default_factory=list)
    corpus_metadata: list[dict[str, Any]] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    bm25: Optional[BM25Okapi] = None

    #Build
    def build(self,nodes:list[TextNode]) -> None:
        """Fit BM25 over a list of LlamaIndex TextNode"""
        self.corpus_text = [node.get_content() for node in nodes]
        self.corpus_metatdata = [dict(node.metadata) for node in nodes]
        self.node_id = [node.id_ for node in nodes]

        tokenised_corpus = [_tokenise(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenised_corpus)
        logger.info("bm25_store_built", extra={"docs": len(self.corpus_texts)})

    #Search
    def search(self,query:str , top_k:int = 5) -> list[RetrievedChunk]:
        """
            Score the query against the corpus and return the top_k chunks.

            Returns RetrievedChunk objects with source="sparse" so callers
            can attribute results back to BM25 for debugging and metrics.
            Documents with zero query-term overlap are skipped to keep them
            out of the fused result list.
            """
        if self.bm25 is None or not self.corpus_texts:
            return []
        
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        top_k = min(top_k,len(scores))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[RetrievedChunk] = []
        for i in top_indices:
            if scores[i] <= 0.0:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=self.node_ids[i],
                    text=self.corpus_texts[i],
                    metadata=self.corpus_metadata[i],
                    score=float(scores[i]),
                    source="sparse",
                )
            )
        return results
    

    def save(self, path: Path) -> None:
        """Serialise the BM25 store to disk via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "corpus_texts": self.corpus_texts,
                    "corpus_metadata": self.corpus_metadata,
                    "node_ids": self.node_ids,
                    "bm25": self.bm25,
                },
                f,
            )
        logger.info("bm25_store_saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        """Reconstruct a BM25 store from a pickle file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls(
            corpus_texts=data["corpus_texts"],
            corpus_metadata=data["corpus_metadata"],
            node_ids=data["node_ids"],
            bm25=data["bm25"],
        )
        logger.info(
            "bm25_store_loaded",
            extra={"path": str(path), "docs": len(store.corpus_texts)},
        )
        return store
    

def bm25_path_for(space_key: str) -> Path:
    """
    Standard on-disk path for a space's BM25 pickle.

    Both the indexer (offline) and the retriever (online) use this
    helper so the two layers stay in sync about where the file lives.
    """
    return Path(settings.chroma.persist_dir) / f"bm25_{space_key.lower()}.pkl"



