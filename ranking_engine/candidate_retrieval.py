"""
Candidate retrieval module for the AI Research Intelligence Platform.

Fetches candidate entities (papers, repos, topics) from backing stores and
applies baseline quality filters and diversity sampling before handing the
list to the ranking pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CandidateItem:
    """A single candidate entity ready for the ranking pipeline.

    All optional numeric fields default to 0 / None so that the scorer can
    decide how to treat missing data.
    """

    entity_id: str
    entity_type: str  # "paper" | "repo" | "topic"
    title: str
    description: str = ""

    # Timestamps
    published_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None

    # GitHub signals
    github_stars: int = 0
    github_forks: int = 0
    github_watchers: int = 0
    github_stars_today: int = 0

    # Citation signals
    citation_count: int = 0
    citation_history: List[int] = field(default_factory=list)

    # Community signals
    upvotes: int = 0
    comments: int = 0
    shares: int = 0

    # Time-series signals
    signals_history: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Semantic embedding (sentence-transformer output)
    embedding: Optional[NDArray[np.float32]] = None

    # Computed quality flags
    quality_score: float = 1.0     # 0–1, set by quality filter
    is_spam: bool = False

    # Arbitrary extra metadata (authors, tags, URLs, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> float:
        """Age of the item in fractional days (based on published_at)."""
        if self.published_at is None:
            return 0.0
        now = datetime.now(timezone.utc)
        pub = self.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return max((now - pub).total_seconds() / 86_400.0, 0.0)

    @property
    def recency_score(self) -> float:
        """Exponential recency score (1.0 = brand new, decays over 30 days)."""
        age = self.age_days
        return float(np.exp(-age / 30.0))


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------


def _passes_quality_filter(
    item: CandidateItem,
    min_quality_score: float = 0.05,
    max_age_days: Optional[float] = None,
) -> bool:
    """Return True if the item clears minimum quality thresholds."""
    if item.is_spam:
        return False
    if item.quality_score < min_quality_score:
        return False
    if max_age_days is not None and item.age_days > max_age_days:
        return False
    return True


def _diversity_sample(
    items: List[CandidateItem],
    limit: int,
    diversity_factor: float = 0.3,
) -> List[CandidateItem]:
    """Return at most ``limit`` items with controlled diversity.

    A fraction (``diversity_factor``) of slots is filled by random sampling
    from the full pool to prevent highly-clustered items from dominating.
    The remaining slots are taken from the top of the list (assumed sorted by
    a pre-ranking criterion).

    Args:
        items: Pre-sorted candidate list.
        limit: Maximum number of items to return.
        diversity_factor: Fraction of returned items chosen randomly (0–1).

    Returns:
        Sampled list of length ≤ limit.
    """
    if len(items) <= limit:
        return items

    n_diverse = max(1, int(limit * diversity_factor))
    n_top = limit - n_diverse

    top_items = items[:n_top]
    remaining = items[n_top:]

    diverse_picks = random.sample(remaining, min(n_diverse, len(remaining)))
    combined = top_items + diverse_picks
    random.shuffle(combined)
    return combined[:limit]


# ---------------------------------------------------------------------------
# Retrieval filters dataclass
# ---------------------------------------------------------------------------


@dataclass
class RetrievalFilters:
    """Optional filters applied during candidate retrieval."""

    min_stars: int = 0
    min_citations: int = 0
    max_age_days: Optional[float] = None       # None = no age filter
    min_quality_score: float = 0.05
    required_tags: List[str] = field(default_factory=list)
    excluded_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------


class CandidateRetriever:
    """Retrieves ranking candidates from multiple data sources.

    In production this class would hold references to database connection
    pools and the vector store client.  For now it exposes a clean async API
    that downstream code can depend on without tight coupling to a specific
    storage backend.

    Args:
        db_pool: Async database connection pool (SQLAlchemy / asyncpg, etc.).
        vector_service: QdrantService instance for embedding-based retrieval.
        diversity_factor: Fraction of result slots filled by diversity sampling.
        default_max_age_days: Default age cap for all queries (None = unlimited).
    """

    def __init__(
        self,
        db_pool: Any = None,
        vector_service: Any = None,
        diversity_factor: float = 0.3,
        default_max_age_days: Optional[float] = 90.0,
    ) -> None:
        self._db = db_pool
        self._vector = vector_service
        self._diversity_factor = diversity_factor
        self._default_max_age_days = default_max_age_days
        logger.info(
            "CandidateRetriever initialized (diversity=%.2f, max_age_days=%s)",
            diversity_factor,
            default_max_age_days,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_from_db(
        self,
        entity_type: str,
        limit: int,
        filters: RetrievalFilters,
    ) -> List[CandidateItem]:
        """Execute a database query and return raw CandidateItem objects.

        This stub returns an empty list and must be replaced with real
        DB queries for production.  The interface is intentionally simple so
        that the rest of the pipeline is testable with mock data.

        Args:
            entity_type: "paper", "repo", or "topic".
            limit: Maximum rows to fetch.
            filters: RetrievalFilters instance.

        Returns:
            List of CandidateItem (may be empty if DB not configured).
        """
        if self._db is None:
            logger.debug(
                "No DB configured; returning empty candidates for %s.", entity_type
            )
            return []

        # ---- Production implementation placeholder ----
        # rows = await self._db.fetch(
        #     f"SELECT * FROM {entity_type}s WHERE ... LIMIT $1", limit
        # )
        # return [_row_to_candidate(r, entity_type) for r in rows]
        # -----------------------------------------------
        return []

    async def _apply_quality_filter(
        self,
        items: List[CandidateItem],
        filters: RetrievalFilters,
    ) -> List[CandidateItem]:
        """Remove low-quality and spam candidates."""
        max_age = filters.max_age_days or self._default_max_age_days
        kept = [
            item for item in items
            if _passes_quality_filter(
                item,
                min_quality_score=filters.min_quality_score,
                max_age_days=max_age,
            )
        ]
        logger.debug(
            "Quality filter: %d → %d items.", len(items), len(kept)
        )
        return kept

    async def _apply_metadata_filters(
        self,
        items: List[CandidateItem],
        filters: RetrievalFilters,
    ) -> List[CandidateItem]:
        """Apply star/citation thresholds and exclusion lists."""
        excluded = set(filters.excluded_ids)
        result = []
        for item in items:
            if item.entity_id in excluded:
                continue
            if item.github_stars < filters.min_stars:
                continue
            if item.citation_count < filters.min_citations:
                continue
            if filters.required_tags:
                item_tags = set(item.metadata.get("tags", []))
                if not item_tags.intersection(filters.required_tags):
                    continue
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    async def retrieve_papers(
        self,
        limit: int = 200,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[CandidateItem]:
        """Retrieve research paper candidates.

        Args:
            limit: Maximum number of candidates to return.
            filters: Optional quality and metadata filters.

        Returns:
            Filtered and diversity-sampled list of paper candidates.
        """
        filters = filters or RetrievalFilters(min_citations=0)
        logger.info("Retrieving up to %d paper candidates.", limit)

        raw = await self._fetch_from_db("paper", limit * 3, filters)
        filtered = await self._apply_quality_filter(raw, filters)
        filtered = await self._apply_metadata_filters(filtered, filters)

        # Sort by recency as a pre-ranking heuristic
        filtered.sort(key=lambda x: x.recency_score, reverse=True)

        sampled = _diversity_sample(filtered, limit, self._diversity_factor)
        logger.info("Retrieved %d paper candidates (from %d raw).", len(sampled), len(raw))
        return sampled

    async def retrieve_repos(
        self,
        limit: int = 200,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[CandidateItem]:
        """Retrieve GitHub repository candidates.

        Args:
            limit: Maximum number of candidates to return.
            filters: Optional quality and metadata filters.

        Returns:
            Filtered and diversity-sampled list of repo candidates.
        """
        filters = filters or RetrievalFilters(min_stars=10)
        logger.info("Retrieving up to %d repo candidates.", limit)

        raw = await self._fetch_from_db("repo", limit * 3, filters)
        filtered = await self._apply_quality_filter(raw, filters)
        filtered = await self._apply_metadata_filters(filtered, filters)

        # Sort by stars as a pre-ranking heuristic
        filtered.sort(key=lambda x: x.github_stars, reverse=True)

        sampled = _diversity_sample(filtered, limit, self._diversity_factor)
        logger.info("Retrieved %d repo candidates (from %d raw).", len(sampled), len(raw))
        return sampled

    async def retrieve_topics(
        self,
        limit: int = 50,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[CandidateItem]:
        """Retrieve trending topic candidates.

        Args:
            limit: Maximum number of topics to return.
            filters: Optional filters (tag-based or age-based).

        Returns:
            Filtered list of topic candidates.
        """
        filters = filters or RetrievalFilters()
        logger.info("Retrieving up to %d topic candidates.", limit)

        raw = await self._fetch_from_db("topic", limit * 2, filters)
        filtered = await self._apply_quality_filter(raw, filters)
        filtered = await self._apply_metadata_filters(filtered, filters)

        logger.info("Retrieved %d topic candidates.", len(filtered[:limit]))
        return filtered[:limit]

    async def retrieve_all(
        self,
        limit: int = 300,
        paper_filters: Optional[RetrievalFilters] = None,
        repo_filters: Optional[RetrievalFilters] = None,
        topic_filters: Optional[RetrievalFilters] = None,
    ) -> List[CandidateItem]:
        """Retrieve candidates of all types concurrently.

        Divides the ``limit`` budget proportionally: 50 % papers, 35 % repos,
        15 % topics.

        Args:
            limit: Total candidate budget.
            paper_filters: Filters specific to papers.
            repo_filters: Filters specific to repos.
            topic_filters: Filters specific to topics.

        Returns:
            Merged and diversity-sampled list of candidates.
        """
        n_papers = int(limit * 0.50)
        n_repos = int(limit * 0.35)
        n_topics = limit - n_papers - n_repos

        papers, repos, topics = await asyncio.gather(
            self.retrieve_papers(n_papers, paper_filters),
            self.retrieve_repos(n_repos, repo_filters),
            self.retrieve_topics(n_topics, topic_filters),
        )

        all_candidates = papers + repos + topics
        # Final diversity sampling across the merged pool
        sampled = _diversity_sample(all_candidates, limit, self._diversity_factor)
        logger.info(
            "retrieve_all: %d total candidates (papers=%d, repos=%d, topics=%d).",
            len(sampled),
            len(papers),
            len(repos),
            len(topics),
        )
        return sampled
