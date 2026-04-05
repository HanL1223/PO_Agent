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

    
    """


