"""
Jira Ingestion via REST API

Using Jira REST API with JQL for
    1. Fully automated -- runs on a schedule or on-demand
    2. Always current -- queries live data
    3. Incremental -- query by updated date to get only changes
    4. Scriptable -- no human intervention needed

JQL EXAMPLES:
    - All issues in a project:  project = CSCI
    - Updated since yesterday:  project = CSCI AND updated >= -1d
    - Specific types:           project = CSCI AND issuetype in (Story, Task)
    - Exclude done:             project = CSCI AND status != Done

INCREMENTAL SYNC PATTERN:
    1. Store last_sync_timestamp per space
    2. Query: project = {SPACE} AND updated >= "{last_sync}"
    3. Upsert returned issues into the index
    4. Update last_sync_timestamp
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any,Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.core.models import JiraIssue
from src.ingestion.loader import normalise_whitespace, redact_pii

logger = logging.getLogger(__name__)



#Jira API Client
class JiraAPIClient:
    def __init__(
            self,
            base_url:Optional[str] = None,
            email:Optional[str] = None,
            api_token:Optional[str] = None,       
    ) -> None:
        self.base_url = (base_url or settings.jira.base_url).rstrip("/")
        self.email = email or settings.jira.email
        self.api_token = api_token or settings.jira.api_token

        if not all([self.base_url,self.email,self.api_token]):
            raise ValueError(
                "Jira API requires JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN. " \
                "Set in .env or pass as argument"
            )
        auth_str = f"{self.email}:{self.api_token}"

        b64 = base64.b64encode(auth_str.encode()).decode()

        self._client = httpx.Client(
            base_url=f"{self.base_url}/rest/api/3",
            headers={
                "Authorization": f"Basic {b64}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def close(self) -> None:
        "Close the underlying HTTP Client"
        self._client.close()

    def __enter__(self) -> JiraAPIClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    #Health Check

    def health_check(self) -> bool:
        """Verify API connectivity and authentication"""
        try:
            resp = self._client.get("/myself")
            return resp.status_code ==200
        except httpx.HTTPError:
            return False
    
    #JQL search with Pagination
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )

    def _search_page(
        self,
        jql:str,
        start_at:int = 0,
        max_results :int = 0,
        fields:Optional[list[str]] = None,
    ) -> dict[str,Any]:
        """
        Execute a signle JQL search page

        Each response includes 'total' (total matching issues) and 'startAt'
        (current offset). loop over pages until we have all results.

        The 'fields' parameter controls which fields are returned. Requesting
        only the fields we need reduces response size and API processing time.
        """
        default_fields = [
            "summary",
            "description",
            "issuetype",
            "status",
            "labels",
            "priority",
            "created",
            "updated",
            "comment",
            "customfield_10037",  # Acceptance Criteria for F06
        ]

        payload = {
            "jql":jql,
            "startAt":start_at,
            "maxResults":max_results,
            "fields": fields or default_fields
        }

        resp = self._client.post("/search",json = payload)
        resp.raise_for_status()
        return resp.json()
    
    def search_all(
        self,
        jql: str,
        fields: Optional[list[str]] = None,
        max_total: int = 5000,
    ) -> list[dict[str, Any]]:
        """
        Execute JQL return all matching issues

        iterates over pages of 100 issues each until
        all results are fetched or max_total is reached. The max_total limit
        prevents accidentally fetching too much tickets
        Args:
            jql: JQL query string
            fields: Which fields to return (None = defaults)
            max_total: Safety limit on total issues to fetch

        Returns:
            List of raw Jira issue dicts
        """
        all_issues :list[dict[str,Any]] = []
        start_at = 0
        page_size = 100

        while True:
            page = self._search_page(jql, start_at=start_at, max_results=page_size, fields=fields)
            issues = page.get("issues",[])
            all_issues.extend(issues)

            total = page.get("total",0)
            start_at += len(issues)

            logger.info(
                "jira_search_page",
                extra = {
                    "fetched": len(all_issues),
                    "total": total,
                    "page_size":len(issues)
                },
            )

            if start_at >= total or start_at >= max_total or len(issues) == 0:
                break

            time.sleep(0.5)

        logger.info(
            "jira_search_complete",
            extra={"jql": jql[:100], "total_fetched": len(all_issues)},
        )
        return all_issues

#Issue Parsing
def _extract_text_from_adf(adf_content: Any) -> str:
    """
    Extract plain text from Atlassisan Document Format(ADF)

    ADF structure:
        {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Hello world"}
            ]}
        ]}
    """
    if adf_content is None:
        return ""
    if isinstance(adf_content,str):
        return adf_content
    
    def _walk(node: Any) -> str:
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            if node.get("type") == "text":
                return node.get("text", "")
            children = node.get("content",[])
            parts = [_walk(child) for child in children]
            #add newline aftrer block-level elemetns
            if node.get("type") in ("paragraph", "heading", "listItem", "bulletList", "orderedList"):
                return " ".join(parts) + "\n"
            return " ".join("part")
        if isinstance(node,list):
            return " ".join(_walk(item) for item in node)
        return ""
    return normalise_whitespace(_walk(adf_content))

def parse_jira_issue(raw: dict[str, Any], space_key: str) -> Optional[JiraIssue]:
    """
    Parse a raw Jira API response into JiraIssue

    The Jira API returns a nested JSON structure:
        {
            "key": "CSCI-123",
            "fields": {
                "summary": "...",
                "description": { ... ADF ... },
                "issuetype": {"name": "Story"},
                ...
            }
        }

    We flatten this into domain model, extracting text from ADF
    and normalising all fields.
    """

    try:
        fields = raw.get("fields",{})
        issue_key = raw.get("key","")
        if not issue_key:
            return None
        summary = normalise_whitespace(fields.get("summary",""))
        if not summary:
            return None
        
        #Extract description
        description = _extract_text_from_adf(fields.get("description"))
        description = redact_pii(description) if description else None

        # Extract acceptacne criteria (custom field - used for F06)
        ac_raw = fields.get("customfield_10037")
        acceptance_criteria = _extract_text_from_adf(ac_raw) if ac_raw else None

        # Extract comments
        comments_data= fields.get("comment",{})
        comments_list = comments_data.get("comments", []) if isinstance(comments_data, dict) else []
        comments = [
            redact_pii(_extract_text_from_adf(c.get("body", "")))
            for c in comments_list
            if c.get("body")
        ] or None

        #Extract labels
        labels = fields.get("labels") or None

        # Extract priority
        priority_obj = fields.get("priority")
        priority = priority_obj.get("name") if isinstance(priority_obj, dict) else None

        # Extract issue type
        type_obj = fields.get("issuetype", {})
        issue_type = type_obj.get("name", "Task") if isinstance(type_obj, dict) else "Task"

        # Extract status
        status_obj = fields.get("status", {})
        status = status_obj.get("name", "Unknown") if isinstance(status_obj, dict) else "Unknown"


        return JiraIssue(
            issue_key=issue_key,
            project_key=space_key,
            issue_type=issue_type,
            status=status,
            summary = summary,
            description=description,
            acceptance_criteria=acceptance_criteria,
            comments = comments,
            labels = labels,
            priority = priority,
            raw = raw
        )
    except Exception as e:
        logger.warning("parse_issue_failed", extra = {"key":raw.get("key","?"),"error":str(e)})
        return None
    

#Live Ingestion Pipelien

@dataclass
class LiveIngestionConfig:
    """
    Configuration for live Jira ingestion.

    Attributes:
        space_key: Jira project key (e.g., "CSCI")
        jql_filter: Additional JQL filter (appended to project filter)
        since: Only fetch issues updated after this datetime (ISO format)
        max_issues: Safety limit on total issues to fetch
        ac_custom_field: Custom field ID for Acceptance Criteria
    """

    space_key: str
    jql_filter: str = ""
    since: Optional[str] = None
    max_issues: int = 5000
    ac_custom_field: str = "customfield_10037"


        
        
        
