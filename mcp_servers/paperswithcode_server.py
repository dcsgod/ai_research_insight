"""
PapersWithCode MCP Server for AI Research Intelligence Platform.

Exposes four MCP tools:
  - get_latest_papers    – paginated list of newest papers
  - get_sota_results     – state-of-the-art benchmark results for a task
  - get_trending_methods – trending ML methods
  - get_paper_repos      – code repositories linked to a paper

All results are normalised into PaperSchema / NormalizedSignal objects.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base_server import (
    BaseServer,
    MCPFetchError,
    MCPParseError,
    MCPRateLimitError,
    NormalizedSignal,
    PaperSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PWC_API_BASE = "https://paperswithcode.com/api/v1"
DEFAULT_ITEMS_PER_PAGE = 20
MAX_ITEMS_PER_PAGE = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# PapersWithCode MCP Server
# ---------------------------------------------------------------------------


class PapersWithCodeMCPServer(BaseServer):
    """
    MCP server wrapping the PapersWithCode public REST API v1.

    No authentication is required for the public endpoints used here.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__(server_name="paperswithcode")
        self._timeout = timeout

        # Register tools
        self._register_tool("get_latest_papers", self._tool_get_latest_papers)
        self._register_tool("get_sota_results", self._tool_get_sota_results)
        self._register_tool("get_trending_methods", self._tool_get_trending_methods)
        self._register_tool("get_paper_repos", self._tool_get_paper_repos)

        self._logger.info("PapersWithCodeMCPServer initialised")

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    async def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """GET request with retry and error handling."""
        url = f"{PWC_API_BASE}{path}"
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        ) as client:
            for attempt in range(1, 4):
                try:
                    response = await client.get(url, params=params)

                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 30))
                        raise MCPRateLimitError(
                            f"PapersWithCode rate limit (429). Retry after {retry_after}s.",
                            server_name=self.server_name,
                            status_code=429,
                        )

                    if response.status_code == 404:
                        # Resource not found – return empty list rather than raising
                        self._logger.warning("PapersWithCode 404 for %s", url)
                        return []

                    if response.status_code >= 500:
                        if attempt < 3:
                            wait = 2.0 ** (attempt - 1)
                            self._logger.warning(
                                "PWC 5xx (attempt %d/3). Retrying in %.1fs…", attempt, wait
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise MCPFetchError(
                            f"PapersWithCode returned HTTP {response.status_code}",
                            server_name=self.server_name,
                            status_code=response.status_code,
                        )

                    response.raise_for_status()
                    return response.json()

                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    if attempt == 3:
                        raise MCPFetchError(
                            f"Failed to connect to PapersWithCode after 3 attempts: {exc}",
                            server_name=self.server_name,
                        ) from exc
                    await asyncio.sleep(2.0 ** (attempt - 1))

        raise MCPFetchError("Exhausted retries", server_name=self.server_name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    async def fetch_data(self, **kwargs: Any) -> Any:
        if "paper_id" in kwargs:
            return await self._tool_get_paper_repos(**kwargs)
        if "task_id" in kwargs:
            return await self._tool_get_sota_results(**kwargs)
        return await self._tool_get_latest_papers(**kwargs)

    def normalize(self, raw: Any) -> List[NormalizedSignal]:
        """
        Normalise PapersWithCode API responses into NormalizedSignal objects.

        ``raw`` may be:
          - a list of paper dicts
          - a paginated response dict with a "results" key
          - a list of repository dicts (from get_paper_repos)
          - a SOTA results dict
        """
        if raw is None:
            return []

        # Handle paginated wrapper
        if isinstance(raw, dict) and "results" in raw:
            items = raw["results"]
        elif isinstance(raw, list):
            items = raw
        else:
            self._logger.warning("Unexpected raw type from PapersWithCode: %s", type(raw))
            return []

        signals: List[NormalizedSignal] = []
        for item in items:
            try:
                paper = self._item_to_paper(item)
                if paper:
                    signals.append(NormalizedSignal.from_paper(paper))
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.warning("Skipping PWC item due to error: %s", exc)

        self._logger.debug("Normalised %d papers from PapersWithCode", len(signals))
        return signals

    # ------------------------------------------------------------------
    # Data mapping
    # ------------------------------------------------------------------

    def _item_to_paper(self, item: Dict[str, Any]) -> Optional[PaperSchema]:
        """Convert a PapersWithCode paper or method dict to PaperSchema."""
        if not isinstance(item, dict):
            return None

        paper_id = (
            item.get("id")
            or item.get("arxiv_id")
            or item.get("paper_id")
            or ""
        )
        title = item.get("title", "")

        if not paper_id and not title:
            return None

        # Fallback ID using title slug
        if not paper_id and title:
            import hashlib

            paper_id = hashlib.md5(title.encode()).hexdigest()[:12]

        # Authors
        authors_raw = item.get("authors", [])
        authors: List[str] = []
        for a in authors_raw:
            if isinstance(a, str):
                authors.append(a)
            elif isinstance(a, dict):
                authors.append(a.get("name", ""))

        # URLs
        arxiv_id = item.get("arxiv_id", "")
        url = item.get("url_abs") or item.get("url", "")
        if not url and arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = item.get("url_pdf", "")
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        # Code repository
        repositories = item.get("repositories", [])
        github_url = ""
        has_impl = False
        if repositories:
            has_impl = True
            first_repo = repositories[0]
            if isinstance(first_repo, dict):
                github_url = first_repo.get("url", "")
            elif isinstance(first_repo, str):
                github_url = first_repo

        # Task
        tasks = item.get("tasks", [])
        task = tasks[0] if tasks and isinstance(tasks[0], str) else (
            tasks[0].get("name", "") if tasks and isinstance(tasks[0], dict) else ""
        )

        return PaperSchema(
            id=str(paper_id),
            title=title,
            abstract=item.get("abstract", ""),
            authors=authors,
            url=url,
            pdf_url=pdf_url,
            published_date=_parse_dt(item.get("published") or item.get("date")),
            categories=item.get("categories") or [],
            citation_count=item.get("citations", 0),
            github_url=github_url,
            task=task,
            has_implementation=has_impl,
            raw_score=float(item.get("stars", 0)),
            source="paperswithcode",
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _tool_get_latest_papers(
        self,
        page: int = 1,
        items_per_page: int = DEFAULT_ITEMS_PER_PAGE,
    ) -> Any:
        """
        Retrieve the most recently added papers from PapersWithCode.

        Args:
            page:           Page number (1-indexed).
            items_per_page: Number of papers per page (max 50).

        Returns:
            Paginated API response with a "results" list.
        """
        items_per_page = min(items_per_page, MAX_ITEMS_PER_PAGE)
        params: Dict[str, Any] = {
            "page": page,
            "items_per_page": items_per_page,
            "ordering": "-published",
        }
        self._logger.info(
            "get_latest_papers: page=%d items_per_page=%d", page, items_per_page
        )
        return await self._get("/papers/", params=params)

    async def _tool_get_sota_results(self, task_id: str) -> Any:
        """
        Retrieve state-of-the-art benchmark results for a given task.

        Args:
            task_id: PapersWithCode task slug (e.g. "image-classification").

        Returns:
            SOTA results dict containing task metadata and leaderboard entries.
        """
        self._logger.info("get_sota_results: task_id=%r", task_id)
        # PWC SOTA endpoint
        result = await self._get(f"/tasks/{task_id}/")
        # Augment with evaluation table data if available
        sota_tables = await self._get(f"/tasks/{task_id}/results/")
        if isinstance(result, dict) and isinstance(sota_tables, (list, dict)):
            result["sota_tables"] = sota_tables
        return result

    async def _tool_get_trending_methods(self, limit: int = 20) -> Any:
        """
        Retrieve trending ML methods from PapersWithCode.

        Args:
            limit: Maximum number of methods to return.

        Returns:
            Paginated API response with a "results" list of method dicts.
        """
        limit = min(limit, MAX_ITEMS_PER_PAGE)
        params: Dict[str, Any] = {
            "items_per_page": limit,
            "ordering": "-paper_count",
        }
        self._logger.info("get_trending_methods: limit=%d", limit)
        return await self._get("/methods/", params=params)

    async def _tool_get_paper_repos(self, paper_id: str) -> Any:
        """
        Retrieve the code repositories linked to a specific paper.

        Args:
            paper_id: PapersWithCode paper ID or ArXiv ID slug.

        Returns:
            List of repository dicts associated with the paper.
        """
        self._logger.info("get_paper_repos: paper_id=%r", paper_id)
        return await self._get(f"/papers/{paper_id}/repositories/")
