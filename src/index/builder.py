"""
Hybrid Index Builder

orchestrate the offline build of both the dense
store and the sparse store for one or more Jira spaces.

This is the OFFLINE side of the system. Its only callers are:
  - scripts/ingest_and_index.py (CLI / cron / CI build job)
  - tests that need a built index from scratch

The ONLINE side (HybridRetriever) lives in src/retrieval/. It never
calls anything in this module - it only loads what was built here.

PIPELINE
--------
    For each space:
        ingestion -> JiraIssue list
            |
            v
        chunking.issues_to_nodes -> TextNode list
            |
            +--> dense_store.create_dense_store -> ChromaDB collection
            |       (one per space: jira_csci, jira_edp, ...)
            |
            +--> bm25_store.BM25Store.build / .save -> pickle file
                    (one per space: bm25_csci.pkl, bm25_edp.pkl, ...)

The dense store and the BM25 store are independent artifacts. We could
build them with two function calls in a script. The builder exists
because:

  1. The chunking step is shared. Both stores need the same TextNodes
     with the same IDs. Building them separately risks drift if one
     side regenerates the nodes with different settings.
  2. The lifecycle is shared. Building one space means building both
     stores; loading one space means loading both. Centralising the
     orchestration makes that contract explicit.
  3. It gives the indexing CLI a single entry point per space, with
     consistent logging and error handling.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.config import settings
from src.core.models import JiraIssue
from src.index.bm25_store import BM25Store, bm25_path_for
from src.index.chunking import issues_to_nodes
from src.index.dense_store import collection_name_for, create_dense_store

logger = logging.getLogger(__name__)


@dataclass

class BuildResult:
    """
    Statistics from building one psace hybrid index
    """
    space_key: str
    issues_in: int = 0
    nodes_built: int = 0
    dense_collection: str = ""
    bm25_path: str = ""
    success: bool = False
    error: Optional[str] = None

class HybridIndexBuilder:
    """
    Offline builder for the hybrid index

    statelessL every call to buildspace is independent,
    holds no in-memory references to built stores; the dense store
    persists to ChromaDB and the BM25 store persists to a pickle file,
    and the retrieval layer loads them back from disk.

    Usage:
        builder = HybridIndexBuilder()
        result = builder.build_space("CSCI", csci_issues)
        if not result.success:
    """

    def build_space(
            self,
            space_key:str,
            issues:list[JiraIssue],
            chunk_size:Optional[int] = None,
            chunk_overlap:Optional[int] = None,
    ) -> BuildResult:
        """
        Build (or upsert) both the dense and sparse stores for one space.

        Args:
            space_key: Jira project key, e.g. "CSCI" or "EDP".
            issues: Normalised JiraIssue objects from the ingestion layer.
            chunk_size: Override settings.rag.chunk_size for this build.
            chunk_overlap: Override settings.rag.chunk_overlap for this build.

        Returns:
            A BuildResult with statistics. On failure, success=False and
            error contains the message; the exception is also logged.
        """
        result = BuildResult(
            space_key=space_key,
            issues_in=len(issues)
        )

        if not issues:
            result.error = "No Issues provided"
            logger.warning("build_space_no_issues",extra = {"space":space_key})
            return result
        
        try:
            nodes = issues_to_nodes(
                issues,
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap
            )
            result.nodes_built = len(nodes)
            if not nodes:
                result.error = "Chunking produced 0 nodes"
                logger.warning("build_space_no_nodes", extra={"space": space_key})
                return result
            
            #Dense Store
            collection = collection_name_for(space_key)
            create_dense_store(nodes,collection_name=collection)
            result.dense_collection - collection

            #BM25 Store
            bm25 = BM25Store()
            bm25.build(nodes)
            bm25_path = bm25_path_for(space_key)
            bm25.save(bm25_path)
            result.bm25_path = str(bm25_path)

            result.success = True
            logger.info(
                "space_indexed",
                extra={
                    "space": space_key,
                    "issues": len(issues),
                    "nodes": len(nodes),
                    "collection": collection,
                    "bm25_path": str(bm25_path),
                },
            )

            return result
        
        except Exception as e:
            result.error = str(e)
            logger.exception(
                "build_space_failed",
                extra = {'space':space_key,"error":str(e)},
            )
            return result
        
    def build_all(
            self,
            issues_by_space:dict[str,list[JiraIssue]],
            chunk_size:Optional[int] = None,
            chunk_overlap:Optional[int] = None
    ) -> list[BuildResult]:
        """
         Build every space in a dict {space_key: issues}.

        This is the entry point used by scripts/ingest_and_index.py
        after the ingestion step returns its per-space dictionary.
        """

        results = list[BuildResult] = []
        for space_key,issues in issues_by_space.items():
            results.append(
                self.build_space(
                    space_key,
                    issues,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        # Summary log line for the CI / cron job to grep on
        ok = sum(1 for r in results if r.success)
        total_nodes = sum(r.nodes_built for r in results)
        logger.info(
            "all_spaces_indexed",
            extra={
                "spaces_ok": ok,
                "spaces_total": len(results),
                "total_nodes": total_nodes,
                "persist_dir": settings.chroma.persist_dir,
            },
        )
