"""
Base MCP Server module for AI Research Intelligence Platform.

Provides abstract base class, shared Pydantic schemas, and utilities
that all concrete MCP server implementations inherit from.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class MCPError(Exception):
    """Base exception for all MCP server errors."""

    def __init__(self, message: str, server_name: str = "", status_code: int = 0) -> None:
        super().__init__(message)
        self.server_name = server_name
        self.status_code = status_code
        self.message = message

    def __repr__(self) -> str:
        return (
            f"MCPError(server={self.server_name!r}, "
            f"status={self.status_code}, message={self.message!r})"
        )


class MCPFetchError(MCPError):
    """Raised when fetching data from an upstream source fails."""


class MCPParseError(MCPError):
    """Raised when parsing / normalising fetched data fails."""


class MCPRateLimitError(MCPError):
    """Raised when an upstream API enforces a rate limit."""


class MCPAuthError(MCPError):
    """Raised when authentication with an upstream API fails."""


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class PaperSchema(BaseModel):
    """Normalised representation of an academic paper."""

    id: str = Field(..., description="Unique identifier (e.g. arXiv ID or DOI)")
    title: str = Field(..., description="Full paper title")
    abstract: str = Field(default="", description="Paper abstract / summary")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    url: str = Field(default="", description="Canonical landing page URL")
    pdf_url: str = Field(default="", description="Direct PDF download URL")
    published_date: Optional[datetime] = Field(None, description="Publication or submission date")
    categories: List[str] = Field(default_factory=list, description="Topic / subject categories")
    citation_count: int = Field(default=0, description="Number of citations (if available)")
    github_url: str = Field(default="", description="Linked GitHub repository URL (if any)")
    task: str = Field(default="", description="Primary ML task the paper addresses")
    has_implementation: bool = Field(
        default=False, description="Whether a code implementation is publicly available"
    )
    source: str = Field(default="", description="Originating data source (arxiv, pwc, hf, …)")
    raw_score: float = Field(default=0.0, description="Source-specific popularity/trending score")

    # Content hash used for deduplication
    content_hash: str = Field(default="", description="SHA-256 hash of id+title for dedup")

    @validator("content_hash", always=True, pre=False)
    @classmethod
    def compute_hash(cls, v: str, values: Dict[str, Any]) -> str:  # noqa: N805
        if v:
            return v
        raw = f"{values.get('id', '')}{values.get('title', '')}".lower().encode()
        return hashlib.sha256(raw).hexdigest()

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class RepoSchema(BaseModel):
    """Normalised representation of a code repository (GitHub / HuggingFace)."""

    id: str = Field(..., description="Unique repository identifier")
    name: str = Field(..., description="Repository short name")
    full_name: str = Field(default="", description="owner/repo slug")
    description: str = Field(default="", description="Repository description")
    url: str = Field(default="", description="Web URL of the repository")
    stars: int = Field(default=0, description="Total star count")
    forks: int = Field(default=0, description="Total fork count")
    watchers: int = Field(default=0, description="Watcher / subscriber count")
    language: str = Field(default="", description="Primary programming language")
    topics: List[str] = Field(default_factory=list, description="Attached topic tags")
    owner: str = Field(default="", description="Owner username or organisation")
    created_at: Optional[datetime] = Field(None, description="Repository creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last push / update timestamp")
    open_issues: int = Field(default=0, description="Number of open issues")
    stars_today: int = Field(default=0, description="Stars gained in the most recent period")
    likes: int = Field(default=0, description="HuggingFace-style likes (where applicable)")
    downloads: int = Field(default=0, description="Download count (HuggingFace models/datasets)")
    source: str = Field(default="", description="Originating data source (github, huggingface, …)")

    content_hash: str = Field(default="", description="SHA-256 hash for dedup")

    @validator("content_hash", always=True, pre=False)
    @classmethod
    def compute_hash(cls, v: str, values: Dict[str, Any]) -> str:  # noqa: N805
        if v:
            return v
        raw = f"{values.get('id', '')}{values.get('full_name', '')}".lower().encode()
        return hashlib.sha256(raw).hexdigest()

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class TopicSignalSchema(BaseModel):
    """Normalised representation of a social/community signal (Reddit post, etc.)."""

    id: str = Field(..., description="Unique post / signal identifier")
    title: str = Field(..., description="Post title or headline")
    score: int = Field(default=0, description="Community score / upvotes")
    url: str = Field(default="", description="Link to the original post")
    subreddit: str = Field(default="", description="Source subreddit or community")
    created_utc: Optional[datetime] = Field(None, description="Creation timestamp (UTC)")
    num_comments: int = Field(default=0, description="Comment count")
    upvote_ratio: float = Field(default=0.0, description="Upvote ratio (0–1)")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    sentiment: str = Field(
        default="neutral", description="Sentiment label: positive | negative | neutral"
    )
    source: str = Field(default="reddit", description="Originating platform")

    content_hash: str = Field(default="", description="SHA-256 hash for dedup")

    @validator("content_hash", always=True, pre=False)
    @classmethod
    def compute_hash(cls, v: str, values: Dict[str, Any]) -> str:  # noqa: N805
        if v:
            return v
        raw = f"{values.get('id', '')}{values.get('title', '')}".lower().encode()
        return hashlib.sha256(raw).hexdigest()

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class NormalizedSignal(BaseModel):
    """
    Unified envelope that wraps any of the concrete signal types so that
    the orchestrator can pass heterogeneous signals through a single pipeline.
    """

    signal_type: str = Field(
        ..., description="Type discriminator: 'paper' | 'repo' | 'topic'"
    )
    source: str = Field(..., description="Server / API that produced this signal")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of ingestion",
    )
    content_hash: str = Field(..., description="Deduplication hash")
    data: Dict[str, Any] = Field(..., description="Serialised payload (PaperSchema / RepoSchema / …)")

    @classmethod
    def from_paper(cls, paper: PaperSchema) -> "NormalizedSignal":
        return cls(
            signal_type="paper",
            source=paper.source,
            content_hash=paper.content_hash,
            data=json.loads(paper.json()),
        )

    @classmethod
    def from_repo(cls, repo: RepoSchema) -> "NormalizedSignal":
        return cls(
            signal_type="repo",
            source=repo.source,
            content_hash=repo.content_hash,
            data=json.loads(repo.json()),
        )

    @classmethod
    def from_topic(cls, topic: TopicSignalSchema) -> "NormalizedSignal":
        return cls(
            signal_type="topic",
            source=topic.source,
            content_hash=topic.content_hash,
            data=json.loads(topic.json()),
        )

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


# ---------------------------------------------------------------------------
# Tool result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MCPToolResult:
    """Structured result returned from an MCP tool invocation."""

    tool_name: str
    server_name: str
    success: bool
    data: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "success": self.success,
            "data_count": len(self.data),
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract Base Server
# ---------------------------------------------------------------------------


class BaseServer(ABC):
    """
    Abstract base class for all MCP servers.

    Concrete subclasses must implement:
        - fetch_data(**kwargs) -> Any
        - normalize(raw: Any) -> List[NormalizedSignal]

    They may also register tools in self.tools during __init__ using
    the _register_tool() helper.
    """

    def __init__(self, server_name: str) -> None:
        self.server_name: str = server_name
        self.tools: Dict[str, Callable[..., Any]] = {}
        self._logger: logging.Logger = logging.getLogger(
            f"mcp.{server_name}"
        )
        self._request_count: int = 0
        self._error_count: int = 0
        self._total_signals: int = 0

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a callable as a named MCP tool."""
        self.tools[name] = fn
        self._logger.debug("Registered tool: %s", name)

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self.tools.keys())

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_data(self, **kwargs: Any) -> Any:
        """Fetch raw data from the upstream source."""

    @abstractmethod
    def normalize(self, raw: Any) -> List[NormalizedSignal]:
        """Normalise raw upstream data into NormalizedSignal objects."""

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    async def invoke_tool(
        self, tool_name: str, **kwargs: Any
    ) -> MCPToolResult:
        """
        Invoke a registered tool by name, wrapping it in timing and error
        handling so callers always receive an MCPToolResult.
        """
        if tool_name not in self.tools:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.server_name,
                success=False,
                error=f"Tool '{tool_name}' not found on server '{self.server_name}'",
            )

        start = time.perf_counter()
        try:
            self._logger.info("Invoking tool '%s' with kwargs=%s", tool_name, kwargs)
            self._request_count += 1
            raw = await self.tools[tool_name](**kwargs)
            signals = self.normalize(raw)
            self._total_signals += len(signals)
            duration_ms = (time.perf_counter() - start) * 1000
            self._logger.info(
                "Tool '%s' completed: %d signals in %.1f ms",
                tool_name,
                len(signals),
                duration_ms,
            )
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.server_name,
                success=True,
                data=signals,
                duration_ms=duration_ms,
            )
        except MCPRateLimitError as exc:
            self._error_count += 1
            self._logger.warning("Rate limit hit for tool '%s': %s", tool_name, exc)
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.server_name,
                success=False,
                error=f"Rate limit: {exc}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except MCPAuthError as exc:
            self._error_count += 1
            self._logger.error("Auth error for tool '%s': %s", tool_name, exc)
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.server_name,
                success=False,
                error=f"Auth error: {exc}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except MCPError as exc:
            self._error_count += 1
            self._logger.error("MCP error in tool '%s': %s", tool_name, exc)
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.server_name,
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._error_count += 1
            self._logger.exception("Unexpected error in tool '%s'", tool_name)
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.server_name,
                success=False,
                error=f"Unexpected error: {exc}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(
        self,
        tool_name: str,
        retries: int = 3,
        backoff_base: float = 2.0,
        **kwargs: Any,
    ) -> MCPToolResult:
        """
        Run a named tool with automatic retry/exponential back-off.

        Args:
            tool_name: Name of the registered tool to execute.
            retries: Maximum number of attempts (including the first).
            backoff_base: Base seconds for exponential back-off.
            **kwargs: Arguments forwarded to the tool.

        Returns:
            MCPToolResult with success status and data or error details.
        """
        last_result: Optional[MCPToolResult] = None
        for attempt in range(1, retries + 1):
            last_result = await self.invoke_tool(tool_name, **kwargs)
            if last_result.success:
                return last_result
            if attempt < retries:
                wait = backoff_base ** (attempt - 1)
                self._logger.warning(
                    "Attempt %d/%d failed for '%s'. Retrying in %.1fs…",
                    attempt,
                    retries,
                    tool_name,
                    wait,
                )
                await asyncio.sleep(wait)
        return last_result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return server-level statistics."""
        return {
            "server_name": self.server_name,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "total_signals": self._total_signals,
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} server_name={self.server_name!r} "
            f"tools={self.list_tools()}>"
        )
