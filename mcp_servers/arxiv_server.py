"""
ArXiv MCP Server for AI Research Intelligence Platform.

Fetches papers from the ArXiv Atom API, normalises them into PaperSchema /
NormalizedSignal objects, and exposes three MCP tools:
  - search_papers
  - get_recent_papers
  - get_paper_by_id
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from .base_server import (
    BaseServer,
    MCPAuthError,
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

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

SUPPORTED_CATEGORIES: List[str] = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.NE",
    "stat.ML",
]

# ArXiv recommends a 3 second delay between consecutive requests.
RATE_LIMIT_DELAY: float = 3.0

# Max results per single API call (ArXiv caps at 2000 but 100 is sensible).
MAX_RESULTS_CAP = 100


# ---------------------------------------------------------------------------
# Helper: parse a datetime string that may or may not carry timezone info
# ---------------------------------------------------------------------------


def _parse_dt(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 / RFC-3339 datetime string tolerantly."""
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# ArXiv MCP Server
# ---------------------------------------------------------------------------


class ArXivMCPServer(BaseServer):
    """
    MCP server that wraps the ArXiv public Atom feed API.

    All three tools return raw XML strings which are then normalised by
    self.normalize() into NormalizedSignal objects.
    """

    def __init__(
        self,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(server_name="arxiv")
        self._rate_limit_delay = rate_limit_delay
        self._timeout = timeout
        self._last_request_time: float = 0.0

        # Register tools
        self._register_tool("search_papers", self._tool_search_papers)
        self._register_tool("get_recent_papers", self._tool_get_recent_papers)
        self._register_tool("get_paper_by_id", self._tool_get_paper_by_id)

        self._logger.info(
            "ArXivMCPServer initialised (rate_limit_delay=%.1fs)", rate_limit_delay
        )

    # ------------------------------------------------------------------
    # Rate-limit helper
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        """Enforce the minimum inter-request delay recommended by ArXiv."""
        import time  # local import to avoid shadowing

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            wait = self._rate_limit_delay - elapsed
            self._logger.debug("Rate limiting: sleeping %.2fs", wait)
            await asyncio.sleep(wait)
        import time as _time

        self._last_request_time = _time.monotonic()

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    async def _get(self, params: Dict[str, Any]) -> str:
        """
        Perform a GET request to the ArXiv API with retry on transient errors.

        Returns the raw response body as a string.
        """
        await self._rate_limit()
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            for attempt in range(1, 4):
                try:
                    response = await client.get(ARXIV_API_BASE, params=params)
                    if response.status_code == 429:
                        raise MCPRateLimitError(
                            "ArXiv returned 429 Too Many Requests",
                            server_name=self.server_name,
                            status_code=429,
                        )
                    if response.status_code >= 500:
                        raise MCPFetchError(
                            f"ArXiv returned HTTP {response.status_code}",
                            server_name=self.server_name,
                            status_code=response.status_code,
                        )
                    response.raise_for_status()
                    return response.text
                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    if attempt == 3:
                        raise MCPFetchError(
                            f"Failed to connect to ArXiv after 3 attempts: {exc}",
                            server_name=self.server_name,
                        ) from exc
                    wait = 2.0 ** (attempt - 1)
                    self._logger.warning(
                        "Connection error (attempt %d/3). Retrying in %.1fs…", attempt, wait
                    )
                    await asyncio.sleep(wait)
        raise MCPFetchError("Exhausted retries", server_name=self.server_name)

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    async def fetch_data(self, **kwargs: Any) -> Any:
        """Generic fetch – delegates to the appropriate tool based on kwargs."""
        if "arxiv_id" in kwargs:
            return await self._tool_get_paper_by_id(**kwargs)
        if "query" in kwargs:
            return await self._tool_search_papers(**kwargs)
        return await self._tool_get_recent_papers(**kwargs)

    def normalize(self, raw: Any) -> List[NormalizedSignal]:
        """
        Parse an ArXiv Atom XML string and return a list of NormalizedSignal.

        Each <entry> in the feed is converted to a PaperSchema then wrapped.
        """
        if not isinstance(raw, str) or not raw.strip():
            return []

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise MCPParseError(
                f"Failed to parse ArXiv XML: {exc}", server_name=self.server_name
            ) from exc

        signals: List[NormalizedSignal] = []
        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            try:
                paper = self._entry_to_paper(entry)
                if paper:
                    signals.append(NormalizedSignal.from_paper(paper))
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.warning("Skipping entry due to parse error: %s", exc)

        self._logger.debug("Normalised %d papers from ArXiv XML", len(signals))
        return signals

    # ------------------------------------------------------------------
    # XML parsing helper
    # ------------------------------------------------------------------

    def _entry_to_paper(self, entry: ET.Element) -> Optional[PaperSchema]:
        """Convert a single Atom <entry> element to a PaperSchema."""

        def _text(tag: str, ns: str = ATOM_NS) -> str:
            el = entry.find(f"{{{ns}}}{tag}")
            return (el.text or "").strip() if el is not None else ""

        # --- ID ---
        raw_id = _text("id")
        # ArXiv IDs look like http://arxiv.org/abs/2301.12345v1
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

        # --- Title ---
        title = " ".join(_text("title").split())  # collapse whitespace

        # --- Abstract ---
        abstract = " ".join(_text("summary").split())

        # --- Authors ---
        authors = [
            " ".join((a.find(f"{{{ATOM_NS}}}name") or ET.Element("_")).text.split())
            for a in entry.findall(f"{{{ATOM_NS}}}author")
        ]

        # --- URLs ---
        url = ""
        pdf_url = ""
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            rel = link.get("rel", "")
            href = link.get("href", "")
            title_attr = link.get("title", "")
            if rel == "alternate" or title_attr == "abs":
                url = href
            elif title_attr == "pdf" or link.get("type", "") == "application/pdf":
                pdf_url = href

        # Fallback URL construction
        if not url and arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        # --- Published date ---
        published_date = _parse_dt(_text("published"))

        # --- Categories ---
        primary_cat_el = entry.find(f"{{{ARXIV_NS}}}primary_category")
        categories: List[str] = []
        if primary_cat_el is not None:
            cat = primary_cat_el.get("term", "")
            if cat:
                categories.append(cat)
        for cat_el in entry.findall(f"{{{ATOM_NS}}}category"):
            term = cat_el.get("term", "")
            if term and term not in categories:
                categories.append(term)

        if not arxiv_id or not title:
            return None

        return PaperSchema(
            id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            url=url,
            pdf_url=pdf_url,
            published_date=published_date,
            categories=categories,
            citation_count=0,
            source="arxiv",
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _tool_search_papers(
        self,
        query: str,
        max_results: int = 20,
        categories: Optional[List[str]] = None,
    ) -> str:
        """
        Search ArXiv papers by free-text query, optionally restricted to
        one or more subject categories.

        Args:
            query: Search query string.
            max_results: Number of results to return (capped at 100).
            categories: ArXiv category codes to filter by (e.g. ["cs.AI"]).

        Returns:
            Raw Atom XML string from the ArXiv API.
        """
        max_results = min(max_results, MAX_RESULTS_CAP)

        # Build search query
        search_parts = [f"all:{quote_plus(query)}"]
        if categories:
            valid_cats = [c for c in categories if c in SUPPORTED_CATEGORIES]
            if valid_cats:
                cat_query = " OR ".join(f"cat:{c}" for c in valid_cats)
                search_parts.append(f"({cat_query})")

        search_query = " AND ".join(search_parts)
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        self._logger.info(
            "search_papers: query=%r categories=%s max_results=%d",
            query,
            categories,
            max_results,
        )
        return await self._get(params)

    async def _tool_get_recent_papers(
        self,
        categories: Optional[List[str]] = None,
        days_back: int = 7,
        max_results: int = 50,
    ) -> str:
        """
        Retrieve recently submitted papers for the given categories.

        ArXiv does not offer a direct date-range filter on the Atom API, so we
        fetch the most recent N papers sorted by submission date and let the
        caller filter further if needed.

        Args:
            categories: ArXiv category codes to include. Defaults to all supported.
            days_back: Used to build a submittedDate query (best-effort).
            max_results: Number of results to return.

        Returns:
            Raw Atom XML string.
        """
        if not categories:
            categories = SUPPORTED_CATEGORIES

        valid_cats = [c for c in categories if c in SUPPORTED_CATEGORIES]
        if not valid_cats:
            valid_cats = SUPPORTED_CATEGORIES

        cat_query = " OR ".join(f"cat:{c}" for c in valid_cats)

        # Build date range: ArXiv submittedDate uses YYYYMMDDHHMMSS format
        since_dt = datetime.utcnow() - timedelta(days=days_back)
        since_str = since_dt.strftime("%Y%m%d%H%M%S")
        now_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        search_query = f"({cat_query}) AND submittedDate:[{since_str} TO {now_str}]"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, MAX_RESULTS_CAP),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        self._logger.info(
            "get_recent_papers: categories=%s days_back=%d", valid_cats, days_back
        )
        return await self._get(params)

    async def _tool_get_paper_by_id(self, arxiv_id: str) -> str:
        """
        Retrieve a single paper by its ArXiv ID.

        Args:
            arxiv_id: ArXiv paper identifier, e.g. "2301.12345" or "2301.12345v2".

        Returns:
            Raw Atom XML string containing a single entry.
        """
        # Strip version suffix for the id_list parameter
        base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        params = {"id_list": base_id, "max_results": 1}
        self._logger.info("get_paper_by_id: arxiv_id=%r", arxiv_id)
        return await self._get(params)
