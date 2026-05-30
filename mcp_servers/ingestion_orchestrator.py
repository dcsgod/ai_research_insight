"""
Ingestion Orchestrator for AI Research Intelligence Platform.

Coordinates all five MCP servers, runs full or incremental ingestion
concurrently, deduplicates signals via content hashing, tracks progress,
and returns a unified List[NormalizedSignal].

Configuration via environment variables:

    ARXIV_RATE_LIMIT_DELAY   – seconds between ArXiv requests (default: 3)
    GITHUB_TOKEN             – GitHub Personal Access Token
    HF_TOKEN                 – HuggingFace Hub API token
    REDDIT_CLIENT_ID         – Reddit OAuth2 client ID
    REDDIT_CLIENT_SECRET     – Reddit OAuth2 client secret
    REDDIT_USER_AGENT        – Reddit API user-agent string
    INGESTION_MAX_CONCURRENCY – max concurrent server calls (default: 5)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .arxiv_server import ArXivMCPServer
from .base_server import MCPToolResult, NormalizedSignal
from .github_server import GitHubMCPServer
from .huggingface_server import HuggingFaceMCPServer
from .paperswithcode_server import PapersWithCodeMCPServer
from .reddit_server import RedditMCPServer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source name constants (used by selective ingestion)
# ---------------------------------------------------------------------------

SOURCE_ARXIV = "arxiv"
SOURCE_GITHUB = "github"
SOURCE_HUGGINGFACE = "huggingface"
SOURCE_PWC = "paperswithcode"
SOURCE_REDDIT = "reddit"

ALL_SOURCES: List[str] = [
    SOURCE_ARXIV,
    SOURCE_GITHUB,
    SOURCE_HUGGINGFACE,
    SOURCE_PWC,
    SOURCE_REDDIT,
]

# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


@dataclass
class IngestionProgress:
    """Tracks signals fetched and errors per source during a single run."""

    source: str
    signals_fetched: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


@dataclass
class IngestionReport:
    """Aggregated report for a complete ingestion run."""

    run_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    total_signals: int = 0
    deduplicated_signals: int = 0
    duplicates_removed: int = 0
    per_source: Dict[str, IngestionProgress] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at

    def summary(self) -> str:
        lines = [
            f"IngestionReport run_id={self.run_id}",
            f"  Duration:       {self.duration_seconds:.2f}s",
            f"  Total signals:  {self.total_signals}",
            f"  After dedup:    {self.deduplicated_signals}",
            f"  Duplicates:     {self.duplicates_removed}",
            "  Per source:",
        ]
        for src, prog in self.per_source.items():
            status = "OK" if prog.success else f"ERRORS({len(prog.errors)})"
            lines.append(
                f"    {src:<20} {prog.signals_fetched:>5} signals  "
                f"{prog.duration_ms:>7.1f}ms  [{status}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class IngestionOrchestrator:
    """
    Coordinates concurrent ingestion across all five MCP servers.

    Usage::

        orchestrator = IngestionOrchestrator()
        signals, report = await orchestrator.run_full_ingestion()
        print(report.summary())
    """

    def __init__(
        self,
        arxiv_server: Optional[ArXivMCPServer] = None,
        github_server: Optional[GitHubMCPServer] = None,
        huggingface_server: Optional[HuggingFaceMCPServer] = None,
        pwc_server: Optional[PapersWithCodeMCPServer] = None,
        reddit_server: Optional[RedditMCPServer] = None,
        max_concurrency: Optional[int] = None,
    ) -> None:
        self._logger = logging.getLogger("mcp.orchestrator")

        # Initialise servers (use provided instances or create defaults from env)
        self.servers: Dict[str, Any] = {
            SOURCE_ARXIV: arxiv_server or ArXivMCPServer(
                rate_limit_delay=float(os.getenv("ARXIV_RATE_LIMIT_DELAY", "3"))
            ),
            SOURCE_GITHUB: github_server or GitHubMCPServer(),
            SOURCE_HUGGINGFACE: huggingface_server or HuggingFaceMCPServer(),
            SOURCE_PWC: pwc_server or PapersWithCodeMCPServer(),
            SOURCE_REDDIT: reddit_server or RedditMCPServer(),
        }

        self._max_concurrency = max_concurrency or int(
            os.getenv("INGESTION_MAX_CONCURRENCY", "5")
        )
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

        self._logger.info(
            "IngestionOrchestrator initialised with sources=%s, max_concurrency=%d",
            list(self.servers.keys()),
            self._max_concurrency,
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def deduplicate(signals: List[NormalizedSignal]) -> List[NormalizedSignal]:
        """
        Remove duplicate signals based on content_hash.

        The first occurrence of each hash is kept; subsequent duplicates
        (even from different sources) are discarded.
        """
        seen: Set[str] = set()
        unique: List[NormalizedSignal] = []
        for signal in signals:
            h = signal.content_hash
            if h and h not in seen:
                seen.add(h)
                unique.append(signal)
        return unique

    # ------------------------------------------------------------------
    # Per-source ingestion tasks
    # ------------------------------------------------------------------

    async def _ingest_arxiv(self) -> tuple[List[NormalizedSignal], IngestionProgress]:
        """Run ArXiv ingestion: search + recent papers for core categories."""
        source = SOURCE_ARXIV
        progress = IngestionProgress(source=source)
        server: ArXivMCPServer = self.servers[source]
        signals: List[NormalizedSignal] = []
        start = time.perf_counter()

        # Tool 1: search for trending AI topics
        searches = [
            {"query": "large language model", "max_results": 20, "categories": ["cs.AI", "cs.CL"]},
            {"query": "diffusion model", "max_results": 15, "categories": ["cs.CV", "cs.LG"]},
            {"query": "reinforcement learning", "max_results": 15, "categories": ["cs.AI", "cs.LG"]},
        ]
        for kwargs in searches:
            result = await server.run("search_papers", **kwargs)
            if result.success:
                signals.extend(result.data)
            else:
                progress.errors.append(f"search_papers({kwargs['query']}): {result.error}")

        # Tool 2: recent papers across all supported categories
        result = await server.run(
            "get_recent_papers",
            categories=["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "stat.ML"],
            days_back=3,
            max_results=50,
        )
        if result.success:
            signals.extend(result.data)
        else:
            progress.errors.append(f"get_recent_papers: {result.error}")

        progress.signals_fetched = len(signals)
        progress.duration_ms = (time.perf_counter() - start) * 1000
        return signals, progress

    async def _ingest_github(self) -> tuple[List[NormalizedSignal], IngestionProgress]:
        """Run GitHub ingestion: trending repos in Python and AI-relevant queries."""
        source = SOURCE_GITHUB
        progress = IngestionProgress(source=source)
        server: GitHubMCPServer = self.servers[source]
        signals: List[NormalizedSignal] = []
        start = time.perf_counter()

        trending_configs = [
            {"language": "python", "since": "daily", "limit": 25},
            {"language": "python", "since": "weekly", "limit": 25},
            {"language": "rust", "since": "weekly", "limit": 15},
        ]
        for cfg in trending_configs:
            result = await server.run("get_trending_repos", **cfg)
            if result.success:
                signals.extend(result.data)
            else:
                progress.errors.append(f"get_trending_repos({cfg}): {result.error}")

        # Targeted AI/ML searches
        ai_queries = [
            "machine learning framework stars:>100",
            "LLM inference topic:llm",
            "transformer model topic:deep-learning",
        ]
        for q in ai_queries:
            result = await server.run("search_repos", query=q, sort="stars", limit=20)
            if result.success:
                signals.extend(result.data)
            else:
                progress.errors.append(f"search_repos({q!r}): {result.error}")

        progress.signals_fetched = len(signals)
        progress.duration_ms = (time.perf_counter() - start) * 1000
        return signals, progress

    async def _ingest_huggingface(self) -> tuple[List[NormalizedSignal], IngestionProgress]:
        """Run HuggingFace ingestion: trending models, papers, and datasets."""
        source = SOURCE_HUGGINGFACE
        progress = IngestionProgress(source=source)
        server: HuggingFaceMCPServer = self.servers[source]
        signals: List[NormalizedSignal] = []
        start = time.perf_counter()

        # Trending models by task
        tasks = ["text-generation", "image-classification", "text-to-image", "automatic-speech-recognition"]
        for task in tasks:
            result = await server.run("get_trending_models", task=task, limit=20)
            if result.success:
                signals.extend(result.data)
            else:
                progress.errors.append(f"get_trending_models(task={task!r}): {result.error}")

        # Trending papers
        result = await server.run("get_trending_papers", limit=30)
        if result.success:
            signals.extend(result.data)
        else:
            progress.errors.append(f"get_trending_papers: {result.error}")

        # Trending datasets
        result = await server.run("get_datasets", limit=20)
        if result.success:
            signals.extend(result.data)
        else:
            progress.errors.append(f"get_datasets: {result.error}")

        progress.signals_fetched = len(signals)
        progress.duration_ms = (time.perf_counter() - start) * 1000
        return signals, progress

    async def _ingest_pwc(self) -> tuple[List[NormalizedSignal], IngestionProgress]:
        """Run PapersWithCode ingestion: latest papers and trending methods."""
        source = SOURCE_PWC
        progress = IngestionProgress(source=source)
        server: PapersWithCodeMCPServer = self.servers[source]
        signals: List[NormalizedSignal] = []
        start = time.perf_counter()

        # Latest papers across multiple pages
        for page in range(1, 4):
            result = await server.run("get_latest_papers", page=page, items_per_page=20)
            if result.success:
                signals.extend(result.data)
            else:
                progress.errors.append(f"get_latest_papers(page={page}): {result.error}")

        # Trending methods
        result = await server.run("get_trending_methods", limit=20)
        if result.success:
            signals.extend(result.data)
        else:
            progress.errors.append(f"get_trending_methods: {result.error}")

        progress.signals_fetched = len(signals)
        progress.duration_ms = (time.perf_counter() - start) * 1000
        return signals, progress

    async def _ingest_reddit(self) -> tuple[List[NormalizedSignal], IngestionProgress]:
        """Run Reddit ingestion: hot and trending posts from AI subreddits."""
        source = SOURCE_REDDIT
        progress = IngestionProgress(source=source)
        server: RedditMCPServer = self.servers[source]
        signals: List[NormalizedSignal] = []
        start = time.perf_counter()

        subreddits = ["MachineLearning", "LocalLLaMA", "artificial", "datascience", "deeplearning"]

        # Hot posts from each subreddit
        for sr in subreddits:
            result = await server.run("get_hot_posts", subreddit=sr, limit=25)
            if result.success:
                signals.extend(result.data)
            else:
                progress.errors.append(f"get_hot_posts(r/{sr}): {result.error}")

        # Cross-subreddit trending topics
        result = await server.run(
            "get_trending_topics",
            subreddits=subreddits,
            time_filter="week",
            limit=15,
        )
        if result.success:
            signals.extend(result.data)
        else:
            progress.errors.append(f"get_trending_topics: {result.error}")

        progress.signals_fetched = len(signals)
        progress.duration_ms = (time.perf_counter() - start) * 1000
        return signals, progress

    # ------------------------------------------------------------------
    # Source dispatcher
    # ------------------------------------------------------------------

    _SOURCE_TASKS = {
        SOURCE_ARXIV: "_ingest_arxiv",
        SOURCE_GITHUB: "_ingest_github",
        SOURCE_HUGGINGFACE: "_ingest_huggingface",
        SOURCE_PWC: "_ingest_pwc",
        SOURCE_REDDIT: "_ingest_reddit",
    }

    async def _run_source(
        self, source: str
    ) -> tuple[List[NormalizedSignal], IngestionProgress]:
        """
        Execute the ingestion coroutine for a named source, isolated inside
        the concurrency semaphore so that at most _max_concurrency sources
        run in parallel.

        Any unhandled exception is caught and reported via the IngestionProgress
        so that a single failing source does not abort the whole run.
        """
        method_name = self._SOURCE_TASKS.get(source)
        if not method_name:
            prog = IngestionProgress(source=source)
            prog.errors.append(f"Unknown source: {source!r}")
            return [], prog

        async with self._semaphore:
            self._logger.info("Starting ingestion for source: %s", source)
            try:
                method = getattr(self, method_name)
                signals, progress = await method()
                self._logger.info(
                    "Completed ingestion for %s: %d signals", source, len(signals)
                )
                return signals, progress
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.exception("Unhandled error ingesting source %s", source)
                prog = IngestionProgress(source=source)
                prog.errors.append(f"Unhandled exception: {exc}")
                return [], prog

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_full_ingestion(
        self,
    ) -> tuple[List[NormalizedSignal], IngestionReport]:
        """
        Run ingestion across all five sources concurrently.

        Returns:
            A tuple of (deduplicated_signals, IngestionReport).
        """
        import uuid

        run_id = uuid.uuid4().hex[:8]
        report = IngestionReport(run_id=run_id)
        self._logger.info("=== Full ingestion started (run_id=%s) ===", run_id)

        # Launch all source tasks concurrently
        tasks = [self._run_source(src) for src in ALL_SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_signals: List[NormalizedSignal] = []
        for source, outcome in zip(ALL_SOURCES, results):
            if isinstance(outcome, Exception):
                prog = IngestionProgress(source=source)
                prog.errors.append(str(outcome))
                report.per_source[source] = prog
            else:
                signals, progress = outcome  # type: ignore[misc]
                report.per_source[source] = progress
                all_signals.extend(signals)

        report.total_signals = len(all_signals)
        unique_signals = self.deduplicate(all_signals)
        report.deduplicated_signals = len(unique_signals)
        report.duplicates_removed = report.total_signals - report.deduplicated_signals
        report.finished_at = time.time()

        self._logger.info(
            "=== Full ingestion complete (run_id=%s) ===\n%s",
            run_id,
            report.summary(),
        )
        return unique_signals, report

    async def run_incremental_ingestion(
        self,
        sources: List[str],
    ) -> tuple[List[NormalizedSignal], IngestionReport]:
        """
        Run ingestion for a specific subset of sources.

        Args:
            sources: List of source names to ingest.
                     Valid values: "arxiv", "github", "huggingface",
                     "paperswithcode", "reddit".

        Returns:
            A tuple of (deduplicated_signals, IngestionReport).
        """
        import uuid

        # Validate source names
        unknown = [s for s in sources if s not in ALL_SOURCES]
        if unknown:
            self._logger.warning("Unknown sources requested: %s – skipping", unknown)
        valid_sources = [s for s in sources if s in ALL_SOURCES]

        if not valid_sources:
            self._logger.error("No valid sources to ingest.")
            return [], IngestionReport(run_id="empty")

        run_id = uuid.uuid4().hex[:8]
        report = IngestionReport(run_id=run_id)
        self._logger.info(
            "=== Incremental ingestion started (run_id=%s, sources=%s) ===",
            run_id,
            valid_sources,
        )

        tasks = [self._run_source(src) for src in valid_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_signals: List[NormalizedSignal] = []
        for source, outcome in zip(valid_sources, results):
            if isinstance(outcome, Exception):
                prog = IngestionProgress(source=source)
                prog.errors.append(str(outcome))
                report.per_source[source] = prog
            else:
                signals, progress = outcome  # type: ignore[misc]
                report.per_source[source] = progress
                all_signals.extend(signals)

        report.total_signals = len(all_signals)
        unique_signals = self.deduplicate(all_signals)
        report.deduplicated_signals = len(unique_signals)
        report.duplicates_removed = report.total_signals - report.deduplicated_signals
        report.finished_at = time.time()

        self._logger.info(
            "=== Incremental ingestion complete (run_id=%s) ===\n%s",
            run_id,
            report.summary(),
        )
        return unique_signals, report

    # ------------------------------------------------------------------
    # Stats & helpers
    # ------------------------------------------------------------------

    def get_server_stats(self) -> Dict[str, Any]:
        """Return aggregated statistics from all underlying MCP servers."""
        return {
            source: server.get_stats()
            for source, server in self.servers.items()
        }

    def __repr__(self) -> str:
        return (
            f"<IngestionOrchestrator sources={list(self.servers.keys())} "
            f"max_concurrency={self._max_concurrency}>"
        )
