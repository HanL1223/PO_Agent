"""
Multi-Space(Project) Ingestion Pipeline

Loading Jira CSV exports from multiple prioject and normalise them inot a unified format for indexing

DESIGN PATTERN: Strategy Pattern for Loaders
    Different CSV exports may have different column names. The ColumnMapping
    dataclass allows each space to define its own column mapping without
    changing the core loading logic.


PIPELINE:
    For each space:
        1. Load CSV with encoding handling
        2. Map columns to unified schema
        3. Clean text (normalise whitespace, redact PII)
        4. Build structured text for embedding
        5. Emit list of JiraIssue objects

should be replace my Jira JQL loading via API in production
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.core.models import JiraIssue

logger = logging.getLogger(__name__)


#Column Mapping
@dataclass
class ColumnMapping:
    """
    Maps CSV column names to unified JiraIssue field

    Sample:
    mapping = ColumnMapping.default()  # Works for most Jira exports
        mapping = ColumnMapping(
            issue_key="Key",
            summary="Title",
            description="Body",
        )
    """
    issue_key: str = "Issue key"
    project_key: str = "Project key"
    issue_type: str = "Issue Type"
    status: str = "Status"
    summary: str = "Summary"
    description: str = "Description"
    acceptance_criteria: str = "Custom field (Acceptance Criteria)"
    labels: str = "Labels"
    priority: str = "Priority"
    created: str = "Created"
    updated: str = "Updated"
    comments: str = "Comment"

    @classmethod
    def default(cls) -> ColumnMapping:
        """Default mapping for standard Jira CSV export"""
        return cls()

    @classmethod
    def auto_detect(cls,columns: list[str]) -> ColumnMapping:
        """
        Attempt to auto detect column mapping from csv headre
        """
        normalised = {col.lower().replace(" ","").replace("_",""): col for col in columns}

        def find_col(candidates:list[str],default:str) -> str:
            for c in candidates:
                key = c.lower().replace(" ","").replace("_","")
                if key in normalised:
                    return normalised[key]
            return default
        return cls(
            issue_key=find_col(["Issue key", "issuekey", "Key"], "Issue key"),
            project_key=find_col(["Project key", "projectkey", "Project"], "Project key"),
            issue_type=find_col(["Issue Type", "issuetype", "Type"], "Issue Type"),
            status=find_col(["Status"], "Status"),
            summary=find_col(["Summary", "Title"], "Summary"),
            description=find_col(["Description", "Body"], "Description"),
            acceptance_criteria=find_col(
                ["Custom field (Acceptance Criteria)", "Acceptance Criteria", "AC"],
                "Custom field (Acceptance Criteria)",
            ),
            labels=find_col(["Labels", "Tags"], "Labels"),
            priority=find_col(["Priority"], "Priority"),
            created=find_col(["Created", "Created Date"], "Created"),
            updated=find_col(["Updated", "Updated Date"], "Updated"),
            comments=find_col(["Comment", "Comments"], "Comment"),
        )
    

#Text rpocessing
def normalise_whitespace(text:str) -> str:
    """Collapse multiple whitespace characters into single spaces"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def redact_pii(text:str) -> str:
    """
    Remove PI from jira text like accoutn id,
    """
    if not text:
        return ""
    # Remove Jira account references: [~accountid:xxx]
    text = re.sub(r"\[~accountid:[^\]]+\]", "[user]", text)
    # Remove embedded images: !image.png|...!
    text = re.sub(r"![\w\-./]+(\|[^!]*)?\!", "[image]", text)
    # Remove Atlassian internal IDs
    text = re.sub(r"accountid:[a-f0-9]{24}", "[redacted]", text)
    return text

def build_structured_text(
    summary: str,
    description: Optional[str] = None,
    acceptance_criteria: Optional[str] = None,
    comments: Optional[str] = None,
    ) -> str:
    """
    build a structured text block for embedding,

    Structured text with section headers helps
    Chunking
    Retrieval
    Generation
    """
    sections: list[str] = []

    if summary:
        sections.append(f"Summary\n{normalise_whitespace(summary)}")

    if description:
        #Add 2 layer in case on desciption contain pii or whitespace only
        cleaned = redact_pii(normalise_whitespace(description))
        if cleaned:
            sections.append(f"Description\n{cleaned}")

    if acceptance_criteria:
        cleaned = redact_pii(normalise_whitespace(acceptance_criteria))
        if cleaned:
            sections.append(f"Acceptance Criteria\n{cleaned}")

    if comments:
        cleaned = redact_pii(normalise_whitespace(comments))
        if cleaned:
            sections.append(f"Comments\n{cleaned}")

    return "\n\n---\n\n".join(sections)


#CSV Loader

def load_csv(csv_path: Path, encoding:str = 'uft-8-sig') -> pd.DataFrame:
    """
    Loading Jira CSV with encoding handling

    """
    logger.info("loading_csv",extra={"path":str(csv_path)})

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found in {csv_path}")
    
    df = pd.read_csv(csv_path, encoding = encoding,encoding_errors="replace" )
    logger.info("csv_loaded",extra = {"row":len(df),"columns":len(df.columns)})
    return df


#Ingestion pipeline
class IngestionConfig:
    """
    Configuration for a signle space ingestion
        space_key: Jira project key (e.g., "CSCI")
        csv_path: Path to the CSV export file
        column_mapping: How CSV columns map to JiraIssue fields
        auto_detect_columns: Whether to auto-detect column mapping
    """

    space_key:str
    csv_path: Path
    Column_mapping: ColumnMapping = field(default_factory=ColumnMapping.default)
    auto_detect_columns: bool = True

@dataclass
class IngestionResult:
    """Statistics from an ingestion run"""
    space_key: str
    issues_loaded: int = 0
    issues_valid: int = 0
    issues_skipped: int = 0
    errors: list[str] = field(default_factory=list)

def ingest_space(config: IngestionConfig) -> tuple[list[JiraIssue],IngestionResult]:
    """
    Ingest a signle Jira space from CSV
    """

    result = IngestionResult(space_key=config.space_key)

    #Step 1 load csv
    df = load_csv(config.csv_path)

    #Auto detect columns
    mapping = config.Column_mapping
    if config.auto_detect_columns:
        mapping = ColumnMapping.auto_detect(list(df.columns))
        logger.info(
            "columns_auto_detected",
            extra={"space": config.space_key, "mapping_summary": mapping.issue_key},
        )

    #Process each row

    issues:list[JiraIssue] = []

    for idx,row in df.iterrows():
        try:
            #Helper to get value with fallback
            def get_val(primary:str, *fallbacks:str) -> Optional[str]:
                for col_name in [primary,*fallbacks]:
                    if col_name in row and pd.notna(row[col_name]):
                        return str(row[col_name]).strip()
                return None
            
            #Extract and validate require field
            issue_key = get_val(mapping.issue_key) or ""
            summary = get_val(mapping.summary) or ""

            if not issue_key or not summary:
                result.issues_skipped += 1
                continue
            #Parse labels
            raw_labels = get_val(mapping.labels)
            labels: Optional[str] = None
            if raw_labels:
                labels = [l.strip() for l in raw_labels.split(",") if l.strip()]
            
            # Parse comments
            raw_comments = get_val(mapping.comments)
            comments_list: Optional[list[str]] = None
            if raw_comments:
                comments_list = [raw_comments]

            # Build Issue
            issue = JiraIssue(
                issue_key=issue_key,
                project_key=config.space_key,
                issue_type=get_val(mapping.issue_type) or "Task",
                status=get_val(mapping.status) or "Unknown",
                summary=summary,
                description= get_val(mapping.description),
                acceptance_criteria=get_val(mapping.acceptance_criteria),
                comments = comments_list,
                labels = labels,
                priority=get_val(mapping.priority),
                raw=row.to_dict() if hasattr(row, "to_dict") else None,
            )
            issues.append(issue)
            result.issues_valid += 1 
        except Exception as e:
            result.errors.append(f"Row {idx}: {e}")
            result.issues_skipped += 1


    logger.info(
        "ingestion_completed",
        extra = {
            "space": config.space_key,
            "valid": result.issues_valid,
            "skipped": result.issues_skipped,
            "errors": len(result.errors),
        }
    )

def ingest_all_spaces(
    data_dir: Path,
    spaces: Optional[list[str]] = None,
) -> tuple[dict[str, list[JiraIssue]], list[IngestionResult]]:
    """
    Ingest all configured Jira spaces.

     iterates over all configured spaces
    and ingests each one. The result is a dictionary keyed by space name,
    which is then used by the indexing pipeline to create per-space indices.

    Args:
        data_dir: Directory containing CSV files (named {SPACE}.csv)
        spaces: List of space keys. If None, uses settings.jira.space_list

    Returns:
        Tuple of (dict of space->issues, list of ingestion results)
    """
    from src.config import settings
    if spaces is None:
        spaces = settings.jira.space_list

    all_issues: dict[str, list[JiraIssue]] = {}
    all_results: list[IngestionResult] = []

    for space in spaces:
        csv_path = data_dir / f"{space}.csv"

        if not csv_path.exists():
            logger.warning("csv_not_found", extra={"space": space, "path": str(csv_path)})
            all_results.append(
                IngestionResult(space_key=space, errors=[f"File not found: {csv_path}"])
            )
            continue

        config = IngestionConfig(space_key=space, csv_path=csv_path)
        issues, result = ingest_space(config)
        all_issues[space] = issues
        all_results.append(result)

    total = sum(len(v) for v in all_issues.values())
    logger.info(
        "all_spaces_ingested",
        extra={"spaces": len(all_issues), "total_issues": total},
    )

    return all_issues, all_results