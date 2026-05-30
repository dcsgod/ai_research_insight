"""
GitHub MCP Server for AI Research Intelligence Platform.

Provides three MCP tools:
  - get_trending_repos  – uses GitHub Search API with date filters to surface
                          repos that gained the most stars recently (mirrors
                          the github.com/trending experience)
  - get_repo_details    – fetches full metadata for a single owner/repo
  - search_repos        – full-text repository search

All results are normalised into RepoSchema / NormalizedSignal objects.

Authentication:
    Set the GITHUB_TOKEN environment variable to a GitHub Personal Access Token
    (classic or fine-grained) to avoid the 60 req/h unauthenticated rate limit.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base_server import (
    BaseServer,
    MCPAuthError,
    MCPFetchError,
    MCPParseError,
    MCPRateLimitError,
    NormalizedSignal,
    RepoSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
GITHUB_ACCEPT_HEADER = "application/vnd.github+json"
GITHUB_API_VERSION = "2022-11-28"

VALID_SINCE_VALUES = ("daily", "weekly", "monthly")
DEFAULT_LANGUAGE = "python"
MAX_RESULTS_CAP = 100

# Seconds to wait between requests when no token is configured
UNAUTHENTICATED_DELAY = 2.0


# ---------------------------------------------------------------------------
# Helper: parse ISO-8601 datetimes
# ---------------------------------------------------------------------------


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# GitHub MCP Server
# ---------------------------------------------------------------------------


class GitHubMCPServer(BaseServer):
    """
    MCP server that wraps the GitHub REST API v3.

    The *get_trending_repos* tool replicates the github.com/trending page by
    querying the Search API for repositories created within the requested
    time window and ordering by stars.  This is the recommended programmatic
    approach since GitHub does not expose a dedicated trending endpoint in
    their public API.
    """

    def __init__(
        self,
        github_token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(server_name="github")
        self._token: str = github_token or os.getenv("GITHUB_TOKEN", "")
        self._timeout = timeout
        self._authenticated = bool(self._token)

        if not self._authenticated:
            self._logger.warning(
                "No GITHUB_TOKEN found – operating in unauthenticated mode "
                "(60 req/h limit applies)."
            )

        # Register tools
        self._register_tool("get_trending_repos", self._tool_get_trending_repos)
        self._register_tool("get_repo_details", self._tool_get_repo_details)
        self._register_tool("search_repos", self._tool_search_repos)

        self._logger.info(
            "GitHubMCPServer initialised (authenticated=%s)", self._authenticated
        )

    # ------------------------------------------------------------------
    # HTTP client helper
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": GITHUB_ACCEPT_HEADER,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Perform a GET request to the GitHub API.

        Handles rate-limiting (HTTP 403 / 429), auth errors (401), and
        transient server errors (5xx) with exponential back-off.

        Returns the parsed JSON response body.
        """
        url = f"{GITHUB_API_BASE}{path}"
        headers = self._build_headers()

        if not self._authenticated:
            await asyncio.sleep(UNAUTHENTICATED_DELAY)

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            for attempt in range(1, 4):
                try:
                    response = await client.get(url, headers=headers, params=params)

                    if response.status_code == 401:
                        raise MCPAuthError(
                            "GitHub returned 401 Unauthorized – check GITHUB_TOKEN",
                            server_name=self.server_name,
                            status_code=401,
                        )

                    if response.status_code in (403, 429):
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise MCPRateLimitError(
                            f"GitHub rate limit hit (HTTP {response.status_code}). "
                            f"Retry after {retry_after}s.",
                            server_name=self.server_name,
                            status_code=response.status_code,
                        )

                    if response.status_code >= 500:
                        if attempt < 3:
                            wait = 2.0 ** (attempt - 1)
                            self._logger.warning(
                                "GitHub 5xx (attempt %d/3). Retrying in %.1fs…", attempt, wait
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise MCPFetchError(
                            f"GitHub returned HTTP {response.status_code}",
                            server_name=self.server_name,
                            status_code=response.status_code,
                        )

                    response.raise_for_status()
                    return response.json()

                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    if attempt == 3:
                        raise MCPFetchError(
                            f"Failed to connect to GitHub after 3 attempts: {exc}",
                            server_name=self.server_name,
                        ) from exc
                    wait = 2.0 ** (attempt - 1)
                    self._logger.warning(
                        "Connection error (attempt %d/3). Retrying in %.1fs…", attempt, wait
                    )
                    await asyncio.sleep(wait)

        raise MCPFetchError("Exhausted retries", server_name=self.server_name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    async def fetch_data(self, **kwargs: Any) -> Any:
        """Generic fetch – delegates to the appropriate tool."""
        if "owner" in kwargs and "repo" in kwargs:
            return await self._tool_get_repo_details(**kwargs)
        if "query" in kwargs:
            return await self._tool_search_repos(**kwargs)
        return await self._tool_get_trending_repos(**kwargs)

    def normalize(self, raw: Any) -> List[NormalizedSignal]:
        """
        Convert raw GitHub API response data into NormalizedSignal objects.

        ``raw`` may be:
          - a single repo dict (from get_repo_details)
          - a list of repo dicts
          - a Search API response dict with an "items" key
        """
        if raw is None:
            return []

        # Search API wrapper
        if isinstance(raw, dict) and "items" in raw:
            items = raw["items"]
        elif isinstance(raw, dict):
            items = [raw]
        elif isinstance(raw, list):
            items = raw
        else:
            self._logger.warning("Unexpected raw type from GitHub: %s", type(raw))
            return []

        signals: List[NormalizedSignal] = []
        for item in items:
            try:
                repo = self._item_to_repo(item)
                if repo:
                    signals.append(NormalizedSignal.from_repo(repo))
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.warning("Skipping repo item due to error: %s", exc)

        self._logger.debug("Normalised %d repos from GitHub", len(signals))
        return signals

    # ------------------------------------------------------------------
    # Data mapping
    # ------------------------------------------------------------------

    def _item_to_repo(self, item: Dict[str, Any]) -> Optional[RepoSchema]:
        if not isinstance(item, dict):
            return None

        repo_id = str(item.get("id", ""))
        name = item.get("name", "")
        full_name = item.get("full_name", "")

        if not repo_id or not name:
            return None

        owner_info = item.get("owner") or {}
        owner = owner_info.get("login", "") if isinstance(owner_info, dict) else ""

        return RepoSchema(
            id=repo_id,
            name=name,
            full_name=full_name,
            description=item.get("description") or "",
            url=item.get("html_url", ""),
            stars=item.get("stargazers_count", 0),
            forks=item.get("forks_count", 0),
            watchers=item.get("watchers_count", 0),
            language=item.get("language") or "",
            topics=item.get("topics") or [],
            owner=owner,
            created_at=_parse_dt(item.get("created_at")),
            updated_at=_parse_dt(item.get("updated_at") or item.get("pushed_at")),
            open_issues=item.get("open_issues_count", 0),
            stars_today=item.get("stars_today", 0),
            source="github",
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _tool_get_trending_repos(
        self,
        language: str = DEFAULT_LANGUAGE,
        since: str = "daily",
        limit: int = 25,
    ) -> Any:
        """
        Surface trending repositories by querying the GitHub Search API for
        repos created within the *since* window, ordered by stars descending.

        Args:
            language: Programming language filter (e.g. "python", "rust").
            since:    Trending window – "daily", "weekly", or "monthly".
            limit:    Maximum number of repositories to return.

        Returns:
            A Search API response dict with an "items" list.
        """
        since = since if since in VALID_SINCE_VALUES else "daily"
        limit = min(limit, MAX_RESULTS_CAP)

        days_map = {"daily": 1, "weekly": 7, "monthly": 30}
        days = days_map[since]
        since_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        query_parts = [f"created:>{since_date}"]
        if language:
            query_parts.append(f"language:{language}")
        query_parts.append("stars:>10")

        params: Dict[str, Any] = {
            "q": " ".join(query_parts),
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
            "page": 1,
        }
        self._logger.info(
            "get_trending_repos: language=%r since=%r limit=%d", language, since, limit
        )
        result = await self._get("/search/repositories", params=params)

        # Annotate with stars_today placeholder (GitHub API doesn't provide this
        # directly; real-world impl could diff against a cached baseline)
        if isinstance(result, dict) and "items" in result:
            for item in result["items"]:
                item["stars_today"] = 0  # sentinel – override with diff logic if desired

        return result

    async def _tool_get_repo_details(self, owner: str, repo: str) -> Any:
        """
        Fetch full metadata for a single repository.

        Args:
            owner: Repository owner / organisation login.
            repo:  Repository name.

        Returns:
            GitHub repo object dict.
        """
        self._logger.info("get_repo_details: owner=%r repo=%r", owner, repo)
        result = await self._get(f"/repos/{owner}/{repo}")

        # Also fetch topics (requires separate Accept header in some API versions)
        try:
            topics_result = await self._get(f"/repos/{owner}/{repo}/topics")
            if isinstance(topics_result, dict):
                result["topics"] = topics_result.get("names", [])
        except Exception:  # pylint: disable=broad-except
            pass  # topics are optional

        return result

    async def _tool_search_repos(
        self,
        query: str,
        sort: str = "stars",
        limit: int = 25,
    ) -> Any:
        """
        Search GitHub repositories using the Search API.

        Args:
            query: Full-text search query (supports GitHub search qualifiers).
            sort:  Sort field – "stars", "forks", "updated", or "best-match".
            limit: Maximum results to return.

        Returns:
            A Search API response dict with an "items" list.
        """
        valid_sorts = {"stars", "forks", "updated", "best-match"}
        sort = sort if sort in valid_sorts else "stars"
        limit = min(limit, MAX_RESULTS_CAP)

        params: Dict[str, Any] = {
            "q": query,
            "sort": sort if sort != "best-match" else "",
            "order": "desc",
            "per_page": limit,
            "page": 1,
        }
        self._logger.info(
            "search_repos: query=%r sort=%r limit=%d", query, sort, limit
        )
        return await self._get("/search/repositories", params=params)
