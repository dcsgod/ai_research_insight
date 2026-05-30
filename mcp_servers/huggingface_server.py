"""
HuggingFace MCP Server for AI Research Intelligence Platform.

Exposes four MCP tools:
  - get_trending_models  – models sorted by trending / downloads
  - get_trending_papers  – papers from the HF daily papers feed
  - get_datasets         – datasets filtered by ML task
  - get_model_details    – full metadata for a single model ID

Results are normalised to RepoSchema (models/datasets) or PaperSchema (papers)
and wrapped in NormalizedSignal envelopes.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base_server import (
    BaseServer,
    MCPAuthError,
    MCPFetchError,
    MCPParseError,
    MCPRateLimitError,
    NormalizedSignal,
    PaperSchema,
    RepoSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_API_BASE = "https://huggingface.co/api"
HF_TOKEN_ENV = "HF_TOKEN"
MAX_RESULTS_CAP = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string tolerantly."""
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
# HuggingFace MCP Server
# ---------------------------------------------------------------------------


class HuggingFaceMCPServer(BaseServer):
    """
    MCP server wrapping the public HuggingFace Hub API.

    An optional HF_TOKEN environment variable can be provided for higher
    rate limits and access to gated models.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(server_name="huggingface")
        self._token: str = hf_token or os.getenv(HF_TOKEN_ENV, "")
        self._timeout = timeout

        # Register tools
        self._register_tool("get_trending_models", self._tool_get_trending_models)
        self._register_tool("get_trending_papers", self._tool_get_trending_papers)
        self._register_tool("get_datasets", self._tool_get_datasets)
        self._register_tool("get_model_details", self._tool_get_model_details)

        self._logger.info(
            "HuggingFaceMCPServer initialised (authenticated=%s)", bool(self._token)
        )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """GET request with error handling and basic retry."""
        url = f"{HF_API_BASE}{path}"
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            import asyncio

            for attempt in range(1, 4):
                try:
                    response = await client.get(url, headers=headers, params=params)

                    if response.status_code == 401:
                        raise MCPAuthError(
                            "HuggingFace returned 401 – check HF_TOKEN",
                            server_name=self.server_name,
                            status_code=401,
                        )

                    if response.status_code == 429:
                        raise MCPRateLimitError(
                            "HuggingFace rate limit exceeded (429)",
                            server_name=self.server_name,
                            status_code=429,
                        )

                    if response.status_code >= 500:
                        if attempt < 3:
                            await asyncio.sleep(2.0 ** (attempt - 1))
                            continue
                        raise MCPFetchError(
                            f"HuggingFace returned HTTP {response.status_code}",
                            server_name=self.server_name,
                            status_code=response.status_code,
                        )

                    response.raise_for_status()
                    return response.json()

                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    if attempt == 3:
                        raise MCPFetchError(
                            f"Failed to connect to HuggingFace after 3 attempts: {exc}",
                            server_name=self.server_name,
                        ) from exc
                    await asyncio.sleep(2.0 ** (attempt - 1))

        raise MCPFetchError("Exhausted retries", server_name=self.server_name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    async def fetch_data(self, **kwargs: Any) -> Any:
        if "model_id" in kwargs:
            return await self._tool_get_model_details(**kwargs)
        if "task" in kwargs:
            return await self._tool_get_trending_models(**kwargs)
        return await self._tool_get_trending_papers(**kwargs)

    def normalize(self, raw: Any) -> List[NormalizedSignal]:
        """
        Dispatch normalisation based on the raw payload type.

        - list of model/dataset dicts → RepoSchema signals
        - list of paper dicts → PaperSchema signals
        - single dict → infer type from keys
        """
        if raw is None:
            return []

        if isinstance(raw, list):
            signals: List[NormalizedSignal] = []
            for item in raw:
                try:
                    sig = self._normalize_item(item)
                    if sig:
                        signals.append(sig)
                except Exception as exc:  # pylint: disable=broad-except
                    self._logger.warning("Skipping HF item: %s", exc)
            return signals

        if isinstance(raw, dict):
            sig = self._normalize_item(raw)
            return [sig] if sig else []

        self._logger.warning("Unexpected raw type from HuggingFace: %s", type(raw))
        return []

    # ------------------------------------------------------------------
    # Item normalisation helpers
    # ------------------------------------------------------------------

    def _normalize_item(self, item: Dict[str, Any]) -> Optional[NormalizedSignal]:
        """Detect whether item is a paper or a model/dataset and delegate."""
        if not isinstance(item, dict):
            return None
        # Papers have an "arxivId" or "paper" nested key
        if "arxivId" in item or ("id" in item and "authors" in item):
            paper = self._item_to_paper(item)
            return NormalizedSignal.from_paper(paper) if paper else None
        repo = self._item_to_repo(item)
        return NormalizedSignal.from_repo(repo) if repo else None

    def _item_to_paper(self, item: Dict[str, Any]) -> Optional[PaperSchema]:
        """Convert a HuggingFace paper dict to PaperSchema."""
        paper_id = item.get("arxivId") or item.get("id", "")
        title = item.get("title", "")
        if not paper_id or not title:
            return None

        authors_raw = item.get("authors", [])
        authors: List[str] = []
        for a in authors_raw:
            if isinstance(a, dict):
                authors.append(a.get("name", ""))
            elif isinstance(a, str):
                authors.append(a)

        return PaperSchema(
            id=str(paper_id),
            title=title,
            abstract=item.get("summary", ""),
            authors=authors,
            url=item.get("url") or f"https://arxiv.org/abs/{paper_id}",
            pdf_url=item.get("pdfUrl") or f"https://arxiv.org/pdf/{paper_id}",
            published_date=_parse_dt(item.get("publishedAt") or item.get("submittedAt")),
            categories=item.get("tags", []),
            citation_count=item.get("citationCount", 0),
            raw_score=float(item.get("upvotes", 0)),
            source="huggingface",
        )

    def _item_to_repo(self, item: Dict[str, Any]) -> Optional[RepoSchema]:
        """Convert a HuggingFace model/dataset dict to RepoSchema."""
        model_id = item.get("modelId") or item.get("id", "")
        if not model_id:
            return None

        # full_name is typically "owner/model-name"
        full_name = str(model_id)
        parts = full_name.split("/", 1)
        owner = parts[0] if len(parts) == 2 else ""
        name = parts[1] if len(parts) == 2 else full_name

        return RepoSchema(
            id=full_name,
            name=name,
            full_name=full_name,
            description=item.get("cardData", {}).get("description", "") if isinstance(item.get("cardData"), dict) else "",
            url=f"https://huggingface.co/{full_name}",
            stars=0,
            forks=0,
            watchers=0,
            language="Python",
            topics=item.get("tags", []),
            owner=owner,
            created_at=_parse_dt(item.get("createdAt")),
            updated_at=_parse_dt(item.get("lastModified")),
            open_issues=0,
            likes=item.get("likes", 0),
            downloads=item.get("downloads", 0),
            source="huggingface",
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _tool_get_trending_models(
        self,
        task: Optional[str] = None,
        limit: int = 20,
    ) -> Any:
        """
        Fetch trending / popular models from the HuggingFace Hub.

        Args:
            task:  Pipeline/task filter, e.g. "text-generation", "image-classification".
            limit: Maximum number of models to return.

        Returns:
            List of model dicts from the HF API.
        """
        limit = min(limit, MAX_RESULTS_CAP)
        params: Dict[str, Any] = {
            "sort": "trending",
            "direction": -1,
            "limit": limit,
            "full": "true",
        }
        if task:
            params["pipeline_tag"] = task

        self._logger.info("get_trending_models: task=%r limit=%d", task, limit)
        return await self._get("/models", params=params)

    async def _tool_get_trending_papers(self, limit: int = 20) -> Any:
        """
        Fetch trending papers from the HuggingFace daily papers feed.

        Args:
            limit: Maximum number of papers to return.

        Returns:
            List of paper dicts from the HF API.
        """
        limit = min(limit, MAX_RESULTS_CAP)
        params: Dict[str, Any] = {"limit": limit}
        self._logger.info("get_trending_papers: limit=%d", limit)
        result = await self._get("/papers", params=params)
        # Result may be a list or a dict with a "papers" key
        if isinstance(result, dict) and "papers" in result:
            return result["papers"]
        return result

    async def _tool_get_datasets(
        self,
        task: Optional[str] = None,
        limit: int = 20,
    ) -> Any:
        """
        Fetch datasets from the HuggingFace Hub, optionally filtered by task.

        Args:
            task:  Task/domain filter, e.g. "question-answering".
            limit: Maximum number of datasets to return.

        Returns:
            List of dataset dicts from the HF API.
        """
        limit = min(limit, MAX_RESULTS_CAP)
        params: Dict[str, Any] = {
            "sort": "downloads",
            "direction": -1,
            "limit": limit,
            "full": "true",
        }
        if task:
            params["task_categories"] = task

        self._logger.info("get_datasets: task=%r limit=%d", task, limit)
        return await self._get("/datasets", params=params)

    async def _tool_get_model_details(self, model_id: str) -> Any:
        """
        Retrieve full metadata for a single HuggingFace model.

        Args:
            model_id: Full model identifier in "owner/model-name" format.

        Returns:
            Single model dict from the HF API.
        """
        self._logger.info("get_model_details: model_id=%r", model_id)
        return await self._get(f"/models/{model_id}")
