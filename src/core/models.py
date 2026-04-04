"""
Domain model that represent business concepts

Shared across all modules and serve as the "Contract" between different parts of the system

KEY PRINCIPLES:
    1. Immutability: Use frozen=True on dataclasses to prevent accidental mutation
    2. Type safety: Every field has a type annotation for IDE support
    3. Validation: Pydantic models validate data at construction time
    4. Serialisation: All models can convert to/from dictionaries and JSON


"""


from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field



#ENUMS
#Prevents typos in string comparisons and provide
#IDE autocompletions


class TicketStyle(str, Enum):
    """Detected style of a Jira ticket."""

    BRIEF = "brief"
    VERBOSE = "verbose"
    AUTO = "auto"


class AgentAction(str, Enum):
    """Actions the LangGraph agent can take."""

    SEARCH = "search"
    GENERATE = "generate"
    VALIDATE = "validate"
    REFINE = "refine"
    CREATE_JIRA = "create_jira"
    BATCH_CREATE = "batch_create"
    COMPLETE = "complete"


#Jira Issue Model - Internal

@dataclass(frozen=True)
class JiraIssue:
    """
    Represnets a normalisd Jira issue from any project

    Project_key is used to identifies which Jira space this issue belongs to, and criticals for per space indexing and
    cross space retrieval filtering (e.g. I have a BAU task on another board hence reduced time in this one etc)

    Fields:
        issue_key: Unique identifier like "CSCI-123" or "EDP-456"
        project_key: The Jira project prefix (e.g., "CSCI", "EDP")
        issue_type: Task, Story, Bug, Epic, etc.
        status: Current workflow state (To Do, In Progress, Done, etc.)
        summary: One-line title of the issue
        description: Full description (may contain Jira wiki markup)
        acceptance_criteria: Definition of done (optional)
        comments: List of comment bodies (optional)
        labels: Tags/categories (optional)
        priority: Urgency level (optional)
        created: When the issue was created (optional)
        updated: When the issue was last updated (optional)
        raw: Original data dictionary for debugging (optional)
    """

    issue_key: str
    project_key: str
    issue_type: str
    status: str
    summary: str
    description: Optional[str] = None
    acceptance_criteria: Optional[str] = None
    comments: Optional[list[str]] = None
    labels: Optional[list[str]] = None
    priority: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    raw: Optional[dict[str, Any]] = None

    @property
    def content_hash(self) -> str:
        """
        Generate a stable hash for dedup
        """
        content = f"{self.issue_key}:{self.summary}:{self.description or ''}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


#Chunk models

@dataclass
class RetrievedChunk:
    """
    Document chunk retrieved from the hybrid index

    This wraps the raw retrieval result with with domain-sepecific metadata, the source field indicates whether this
    chunk came from the dense/embedding index or the sparse/BM25 index, useful
    for debuging retrieval quality
    """
    chunk_id:str
    text:str
    metadata:dict[str,Any] = Field(default_factory = dict)
    score:float  = 0.0
    source: str = 'dense'

    @property
    def issue_key(self):
        return self.metadata.get("issue_key","UNKNOWN")
    @property
    def project_key(self):
        return self.metadata.get("project_key","UNKNOWN")
    

#Generation Models
@dataclass
class TicketDraft:
    """
    Generated Jira ticket draft, before or after refinement

    Separating the draft from the final results allows the agent to track the evolution of a ticket through multiple refinement steps
    
    use 'version' field increments with each refinement pass
    """

    content: str
    style: TicketStyle = TicketStyle.AUTO
    version: int = 1 
    validation_score: Optional[float] = None
    validation_feedback: Optional[str] = None


@dataclass
class GenerationResult:
    """
    Complete result from the Langgraph agent pipeline

    This capture everything generateion process
    - Final ticket content
    - Reaonsing trace
    - Performancemetric(timing, token usage)
    - Toll usage statistics

    Use for 
     Debugging
     Evaluation
     Cost tracking
    """

    ticket_text: str
    draft_text: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    style_detected: TicketStyle = TicketStyle.AUTO
    refinement_applied: bool = False
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tools_used: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str,Any]:
        """Serialise for API responses"""
        return {
            "ticket_text": self.ticket_text,
            "draft_text": self.draft_text,
            "retrieved_chunks": [
                {
                    "text": c.text[:300],
                    "issue_key": c.issue_key,
                    "score":c.score,
                    "source":c.source,
                }
                for c in self.retrieved_chunks
            ],
            "style_detected": self.style_detected.value,
            "refinement_applied": self.refinement_applied,
            "reasoning_trace": self.reasoning_trace,
            "iterations": self.iterations,
            "tools_used": self.tools_used,
            "metadata": self.metadata,
        }
    


#Batch Operation Model

class BatchTicketRequest(BaseModel):
    """
    Request for batch ticket creation with human in the loop
    """

    descriptions: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of ticket descriptions to generate.",
    )

    target_space:str = Field(
        default = "CSCI",
        pattern = r"^[A-Z]{2,10}$",
        description = "Target JIra project Key"
    )
    auto_create: bool = Field(
        default = False,
        description = "If True, create in Jira without human review"
    )


class BatchTicketItem(BaseModel):
    """Single ticket within a batch with approval status"""
    index: int
    draft: str
    status: str = Field(default="pending", pattern="^(pending|approved|rejected|edited)$")
    edited_content: Optional[str] = None
    jira_key: Optional[str] = None
    jira_url: Optional[str] = None

class BatchTicketResponse(BaseModel):
    """Response for a batch operation."""

    batch_id: str
    items: list[BatchTicketItem]
    total: int
    approved: int = 0
    created: int = 0

@dataclass
class AgentStep:
    """
    A single step in the agent's reasoning trace.

    Recording each step allows post-hoc analysis of
    agent behaviour. In production, these are logged to the monitoring
    system for debugging and quality improvement.
    """

    thought: str
    action: AgentAction
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    timestamp: Optional[datetime] = None