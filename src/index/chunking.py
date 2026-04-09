"""
Document Chunking

Turn JiraIssue domain objects into LLamaIndex
TextNode chunks ready for indexing

PIPELINE
    JiraIssue
      -> structured text (Summary / Description / AC / Comments)
         (built by src.ingestion.loader.build_structured_text)
      -> LlamaDocument with per-issue metadata
      -> SentenceSplitter chunks at sentence boundaries, respecting
         the "---" section separator we inserted in the structured text
      -> stable, content-derived chunk IDs assigned for idempotent indexing

STRUCTURED TEXT WITH SECTION HEADER
- Gives the chunker natural break points via the "---" separator.
- Preserves semantic structure inside each chunk so the LLM can
  distinguish summary from acceptance criteria during generation.
- Keeps related content (e.g. an AC and its parent description)
  in the same chunk when the chunk_size budget allows it.

The IDs are derived from a SHA-256 of the chunk text and the parent
issue key:  "{ISSUE_KEY}__{sha256[:16]}".

- Re-indexing the same issue with unchanged text produces identical
  IDs, so ChromaDB upserts in place rather than creating duplicates.
- When an issue *does* change, only the affected chunks get new IDs;
  the rest stay put. This is what makes incremental re-indexing safe.

CHUNK SIZE CHOICE

Defaults come from settings.rag (chunk_size=1024, overlap=128). For
Jira tickets these defaults work well: most tickets fit in 1-3 chunks.
Tune up if your tickets are very long (multi-page descriptions); tune
down if you have a lot of very short tickets and want finer-grained
retrieval.

"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from src.config import settings
from src.core.models import JiraIssue
from src.ingestion.loader import build_structured_text

logger = logging.getLogger(__name__)

def issues_to_node(
        issues: list[JiraIssue],
        chunk_size:Optional[int] = None,
        chunk_overlap : Optional[int] = None
) -> list[TextNode]:
    """
    Convert a list of JiraIssue objects into LlamaIndex TextNodes.

    Args:
        issues: Normalised JiraIssue objects from the ingestion pipeline.
        chunk_size: Characters per chunk. Defaults to settings.rag.chunk_size.
        chunk_overlap: Overlap between consecutive chunks. Defaults to
            settings.rag.chunk_overlap.

    Returns:
        A flat list of TextNode objects ready to hand to a dense or
        sparse store builder.
    """
    chunk_size = chunk_size or settings.rag.chunk_size
    chunk_overlap = chunk_overlap or settings.rag.chunk_overlap

    #Build LlamaDocuments from JiraIssue
    documents:list[LlamaDocument] = []
    for issue in issues:
        text = build_structured_text(
            summary= issue.summary,
            description= issue.description,
            acceptance_criteria=issue.acceptance_criteria,
            comments="\n".join(issue.comments) if issue.comments else None,
        )
        if not text.strip():
            continue
        #Poplulate metadata to every chunk
        # It enables filtering at query time (e.g. "only CSCI tickets")
        # and gives the generation step structured context about the source.
        metadata: dict[str, Any] = {
            "issue_key": issue.issue_key,
            "project_key": issue.project_key,
            "issue_type": issue.issue_type,
            "status": issue.status,
            "summary": issue.summary,
            "labels": ",".join(issue.labels) if issue.labels else "",
            "priority": issue.priority or "",
        }
        documents.append(
            LlamaDocument(
                text = text,
                metadata = metadata,
                doc_id = issue.content_hash
            )
        )

    logger.info("documents_prepared",extra = {"count": len(documents)})

    if not documents:
        return []
    
    #Split into chunks
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n---\n\n",
    )

    nodes = splitter.get_nodes_from_documents(documents,show_progress=False)

    #Assign stable content-derived IDs

    for node in nodes:
        content_hash = hashlib.sha256(node.get_content().encode("utf-8")).hdexdigest()[:16]
        issue_key = node.metadata.get("issue_key","UNK")
        node.id_ = {f"issue_key__{content_hash}"}
    logger.info(
        "nodes_created",
        extra={
            "nodes": len(nodes),
            "documents": len(documents),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    return nodes


