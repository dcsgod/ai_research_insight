"""
ranking_service.py — Ranking Service
======================================
FastAPI service layer that bridges the HTTP API with the ranking_engine
module. Provides cached trend rankings for papers, repositories, and topics.

All results are cached in Redis for 15 minutes to avoid repeated heavy
pipeline runs on hot API routes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger("ranking_service")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EntityType = Literal["paper", "repository", "topic"]
Timeframe = Literal["24h", "7d", "30d", "90d"]


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class ForecastSummary(BaseModel):
    """Condensed forecast data attached to a ranked item."""

    entity_id: str
    predicted_trend_score: float = Field(..., ge=0.0, le=1.0)
    momentum: float = Field(..., description="Rate of change over forecast horizon")
    confidence_interval_low: float
    confidence_interval_high: float
    forecast_horizon_days: int = 30


class RankedItem(BaseModel):
    """A single ranked entity with score, metadata, and forecast."""

    id: str
    entity_type: EntityType
    rank: int = Field(..., ge=1)
    title: str
    description: str | None = None
    trend_score: float = Field(..., ge=0.0, le=1.0)
    momentum_score: float = Field(
        default=0.0,
        description="Short-term growth velocity (positive = accelerating)",
    )
    citation_count: int | None = None
    github_stars: int | None = None
    weekly_growth: int | None = None
    tags: list[str] = Field(default_factory=list)
    external_url: str | None = None
    published_at: datetime | None = None
    forecast: ForecastSummary | None = None
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TopicRanking(BaseModel):
    """Ranking entry for a research topic cluster."""

    id: str
    name: str
    slug: str
    description: str | None = None
    trend_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)
    paper_count: int = 0
    repo_count: int = 0
    top_keywords: list[str] = Field(default_factory=list)
    weekly_paper_growth: int = 0
    forecast: ForecastSummary | None = None


# ---------------------------------------------------------------------------
# RankingService
# ---------------------------------------------------------------------------
class RankingService:
    """
    Service class for retrieving and refreshing entity trend rankings.

    Responsibilities:
    - Call ranking_engine.pipeline.RankingPipeline to compute scores
    - Cache results in Redis (TTL: 15 minutes)
    - Return enriched results with optional forecast data
    - Invalidate cache on manual refresh

    Dependencies (injected via constructor):
    - redis_client: async Redis client (redis.asyncio.Redis)
    - db_session:   SQLAlchemy AsyncSession
    - forecast_svc: ForecastService instance (for enrichment)
    """

    CACHE_TTL_SECONDS: int = 15 * 60  # 15 minutes
    CACHE_KEY_PREFIX: str = "rankings"

    def __init__(
        self,
        redis_client: Any,
        db_session: Any,
        forecast_svc: Any | None = None,
    ) -> None:
        self._redis = redis_client
        self._db = db_session
        self._forecast_svc = forecast_svc
        self._logger = structlog.get_logger("ranking_service")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    async def get_ranked_trends(
        self,
        entity_type: EntityType,
        limit: int = 20,
        timeframe: Timeframe = "7d",
        include_forecast: bool = False,
    ) -> list[RankedItem]:
        """
        Retrieve ranked entities of a given type for a given timeframe.

        Results are served from Redis cache when available. Cache misses
        trigger a live pipeline run and populate the cache.

        Args:
            entity_type:      Type of entity to rank ('paper', 'repository', 'topic').
            limit:            Maximum number of results to return (1-100).
            timeframe:        Time window for trend computation.
            include_forecast: If True, attach forecast summaries to each item.

        Returns:
            List of RankedItem sorted by trend_score descending.
        """
        limit = max(1, min(limit, 100))
        cache_key = self._build_cache_key(entity_type, timeframe, limit, include_forecast)

        # ── Try cache first ────────────────────────────────────────────────
        cached = await self._get_from_cache(cache_key)
        if cached is not None:
            self._logger.debug(
                "Cache hit",
                key=cache_key,
                count=len(cached),
            )
            return [RankedItem(**item) for item in cached]

        # ── Cache miss: run pipeline ───────────────────────────────────────
        self._logger.info(
            "Cache miss — running ranking pipeline",
            entity_type=entity_type,
            timeframe=timeframe,
            limit=limit,
        )

        items = await self._run_pipeline(entity_type, timeframe, limit)

        # ── Optionally enrich with forecast data ───────────────────────────
        if include_forecast and self._forecast_svc is not None:
            items = await self._enrich_with_forecasts(items)

        # ── Persist to cache ───────────────────────────────────────────────
        await self._set_cache(cache_key, [item.model_dump(mode="json") for item in items])

        return items

    async def refresh_rankings(self) -> None:
        """
        Force a live ranking pipeline run and invalidate all cached rankings.

        Called by Celery task `update_rankings` or via admin API endpoint.
        """
        self._logger.info("Refreshing rankings — invalidating cache")

        # Clear all ranking cache keys
        pattern = f"{self.CACHE_KEY_PREFIX}:*"
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
                self._logger.info("Cache invalidated", keys_deleted=len(keys))
        except Exception as exc:
            self._logger.warning("Cache invalidation failed", error=str(exc))

        # Run pipeline for all entity types and prime the cache
        for entity_type in ("paper", "repository", "topic"):
            for timeframe in ("24h", "7d", "30d"):
                try:
                    await self.get_ranked_trends(
                        entity_type=entity_type,  # type: ignore[arg-type]
                        limit=50,
                        timeframe=timeframe,  # type: ignore[arg-type]
                    )
                except Exception as exc:
                    self._logger.error(
                        "Failed to prime cache for entity type",
                        entity_type=entity_type,
                        timeframe=timeframe,
                        error=str(exc),
                    )

        self._logger.info("Rankings refreshed and cache primed")

    async def get_topic_rankings(
        self,
        limit: int = 20,
        timeframe: Timeframe = "7d",
        include_forecast: bool = False,
    ) -> list[TopicRanking]:
        """
        Retrieve ranked research topic clusters.

        Args:
            limit:            Maximum number of topics to return.
            timeframe:        Time window for trend computation.
            include_forecast: If True, attach forecast data to each topic.

        Returns:
            List of TopicRanking sorted by trend_score descending.
        """
        cache_key = f"{self.CACHE_KEY_PREFIX}:topics:{timeframe}:{limit}"
        if include_forecast:
            cache_key += ":with_forecast"

        cached = await self._get_from_cache(cache_key)
        if cached is not None:
            return [TopicRanking(**item) for item in cached]

        self._logger.info(
            "Cache miss — fetching topic rankings",
            timeframe=timeframe,
            limit=limit,
        )

        topics = await self._fetch_topic_rankings(timeframe, limit)

        if include_forecast and self._forecast_svc is not None:
            topics = await self._enrich_topics_with_forecasts(topics)

        await self._set_cache(cache_key, [t.model_dump(mode="json") for t in topics])
        return topics

    # -----------------------------------------------------------------------
    # Private: Pipeline invocation
    # -----------------------------------------------------------------------
    async def _run_pipeline(
        self,
        entity_type: EntityType,
        timeframe: Timeframe,
        limit: int,
    ) -> list[RankedItem]:
        """Invoke ranking_engine pipeline and map output to RankedItem models."""
        try:
            from ranking_engine.pipeline import RankingPipeline  # type: ignore[import]

            pipeline = RankingPipeline(db_session=self._db)
            raw_results: list[dict[str, Any]] = await pipeline.run(
                entity_type=entity_type,
                timeframe=timeframe,
                limit=limit,
            )
        except ImportError:
            self._logger.warning(
                "ranking_engine not available — returning stub results",
                entity_type=entity_type,
            )
            raw_results = self._stub_results(entity_type, limit)

        return [
            RankedItem(
                id=str(r.get("id", "")),
                entity_type=entity_type,
                rank=idx + 1,
                title=r.get("title") or r.get("name") or "Unknown",
                description=r.get("description") or r.get("abstract"),
                trend_score=float(r.get("trend_score", 0.0)),
                momentum_score=float(r.get("momentum_score", 0.0)),
                citation_count=r.get("citation_count"),
                github_stars=r.get("stars"),
                weekly_growth=r.get("weekly_star_growth") or r.get("weekly_growth"),
                tags=r.get("topics") or r.get("categories") or [],
                external_url=r.get("html_url") or r.get("pdf_url"),
                published_at=(
                    datetime.fromisoformat(r["published_at"])
                    if r.get("published_at")
                    else None
                ),
            )
            for idx, r in enumerate(raw_results)
        ]

    async def _fetch_topic_rankings(
        self,
        timeframe: Timeframe,
        limit: int,
    ) -> list[TopicRanking]:
        """Fetch topic rankings from PostgreSQL, ordered by trend_score."""
        try:
            from sqlalchemy import text  # type: ignore[import]

            timeframe_days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}[timeframe]
            query = text("""
                SELECT id, name, slug, description, trend_score,
                       paper_count, repo_count, keywords
                FROM topics
                ORDER BY trend_score DESC
                LIMIT :limit
            """)
            result = await self._db.execute(query, {"limit": limit})
            rows = result.mappings().all()

            return [
                TopicRanking(
                    id=str(row["id"]),
                    name=row["name"],
                    slug=row["slug"],
                    description=row.get("description"),
                    trend_score=float(row.get("trend_score", 0.0)),
                    rank=idx + 1,
                    paper_count=int(row.get("paper_count", 0)),
                    repo_count=int(row.get("repo_count", 0)),
                    top_keywords=(
                        json.loads(row["keywords"])[:5]
                        if row.get("keywords")
                        else []
                    ),
                )
                for idx, row in enumerate(rows)
            ]
        except Exception as exc:
            self._logger.error("Failed to fetch topic rankings from DB", error=str(exc))
            return []

    # -----------------------------------------------------------------------
    # Private: Forecast enrichment
    # -----------------------------------------------------------------------
    async def _enrich_with_forecasts(
        self,
        items: list[RankedItem],
    ) -> list[RankedItem]:
        """Attach ForecastSummary to each RankedItem from the forecast service."""
        if self._forecast_svc is None:
            return items

        enriched: list[RankedItem] = []
        for item in items:
            try:
                metrics = await self._forecast_svc.get_momentum_metrics(item.id)
                forecast_series = await self._forecast_svc.get_forecast(
                    entity_id=item.id,
                    entity_type=item.entity_type,
                    periods=30,
                )
                item.forecast = ForecastSummary(
                    entity_id=item.id,
                    predicted_trend_score=forecast_series.predicted_score,
                    momentum=metrics.momentum,
                    confidence_interval_low=forecast_series.ci_low,
                    confidence_interval_high=forecast_series.ci_high,
                    forecast_horizon_days=30,
                )
            except Exception as exc:
                self._logger.debug(
                    "Failed to enrich item with forecast",
                    entity_id=item.id,
                    error=str(exc),
                )
            enriched.append(item)
        return enriched

    async def _enrich_topics_with_forecasts(
        self,
        topics: list[TopicRanking],
    ) -> list[TopicRanking]:
        """Attach ForecastSummary to each TopicRanking."""
        if self._forecast_svc is None:
            return topics

        enriched: list[TopicRanking] = []
        for topic in topics:
            try:
                forecast_series = await self._forecast_svc.get_forecast(
                    entity_id=topic.id,
                    entity_type="topic",
                    periods=30,
                )
                metrics = await self._forecast_svc.get_momentum_metrics(topic.id)
                topic.forecast = ForecastSummary(
                    entity_id=topic.id,
                    predicted_trend_score=forecast_series.predicted_score,
                    momentum=metrics.momentum,
                    confidence_interval_low=forecast_series.ci_low,
                    confidence_interval_high=forecast_series.ci_high,
                    forecast_horizon_days=30,
                )
            except Exception as exc:
                self._logger.debug(
                    "Failed to enrich topic with forecast",
                    topic_id=topic.id,
                    error=str(exc),
                )
            enriched.append(topic)
        return enriched

    # -----------------------------------------------------------------------
    # Private: Redis cache helpers
    # -----------------------------------------------------------------------
    def _build_cache_key(
        self,
        entity_type: EntityType,
        timeframe: Timeframe,
        limit: int,
        include_forecast: bool,
    ) -> str:
        key = f"{self.CACHE_KEY_PREFIX}:{entity_type}:{timeframe}:{limit}"
        if include_forecast:
            key += ":with_forecast"
        return key

    async def _get_from_cache(self, key: str) -> list[dict[str, Any]] | None:
        """Retrieve and deserialize a cached list from Redis."""
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            self._logger.warning("Cache read error", key=key, error=str(exc))
            return None

    async def _set_cache(
        self,
        key: str,
        data: list[dict[str, Any]],
    ) -> None:
        """Serialize and store a list in Redis with TTL."""
        try:
            serialized = json.dumps(data, default=str)
            await self._redis.setex(key, self.CACHE_TTL_SECONDS, serialized)
            self._logger.debug("Cached result", key=key, ttl=self.CACHE_TTL_SECONDS)
        except Exception as exc:
            self._logger.warning("Cache write error", key=key, error=str(exc))

    # -----------------------------------------------------------------------
    # Private: Stub data (when ranking_engine is not installed)
    # -----------------------------------------------------------------------
    @staticmethod
    def _stub_results(entity_type: EntityType, limit: int) -> list[dict[str, Any]]:
        """Return placeholder ranked items for development/testing."""
        items = []
        for i in range(min(limit, 5)):
            if entity_type == "paper":
                items.append(
                    {
                        "id": f"paper-stub-{i}",
                        "title": f"Sample Research Paper #{i + 1}",
                        "abstract": "Placeholder abstract for development.",
                        "trend_score": round(0.9 - i * 0.05, 2),
                        "momentum_score": round(0.5 - i * 0.05, 2),
                        "citation_count": 100 - i * 10,
                        "categories": ["cs.AI"],
                        "published_at": "2024-01-01T00:00:00+00:00",
                    }
                )
            elif entity_type == "repository":
                items.append(
                    {
                        "id": f"repo-stub-{i}",
                        "name": f"sample-repo-{i + 1}",
                        "description": "Placeholder repository for development.",
                        "trend_score": round(0.88 - i * 0.05, 2),
                        "momentum_score": round(0.45 - i * 0.05, 2),
                        "stars": 5000 - i * 500,
                        "weekly_star_growth": 200 - i * 20,
                        "topics": ["ai", "llm"],
                    }
                )
            else:  # topic
                items.append(
                    {
                        "id": f"topic-stub-{i}",
                        "title": f"Sample Topic #{i + 1}",
                        "trend_score": round(0.85 - i * 0.05, 2),
                        "momentum_score": round(0.4 - i * 0.05, 2),
                        "paper_count": 500 - i * 50,
                    }
                )
        return items
