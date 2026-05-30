"""
Trends API router.

Endpoints:
  GET  /api/v1/trends/               - Ranked trending items (papers + repos + topics)
  GET  /api/v1/trends/topics         - Trending topics with momentum metrics
  GET  /api/v1/trends/{entity_id}/signals  - Time-series signals for an entity
  GET  /api/v1/trends/dashboard      - Dashboard summary statistics
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.models.paper import Paper
from backend.models.repository import Repository
from backend.models.topic import Topic
from backend.models.trend_signal import TrendSignal, TrendScore, EntityType, SignalType
from backend.schemas.trend import (
    DashboardSummarySchema,
    EntityCountSchema,
    ForecastSeriesSchema,
    SignalSeriesSchema,
    TopicListResponse,
    TopicSchema,
    TrendingItemSchema,
    TrendingListResponse,
    TrendScoreSchema,
    TrendSignalSchema,
)
from backend.services.cache_service import CacheService, build_key, cache

router = APIRouter(
    prefix="/trends",
    tags=["Trends"],
    responses={
        404: {"description": "Entity not found"},
        500: {"description": "Internal server error"},
    },
)

# Timeframe → days mapping
TIMEFRAME_DAYS: Dict[str, int] = {
    "24h": 1,
    "7d": 7,
    "30d": 30,
}


# ---------------------------------------------------------------------------
# Helper: build TrendingItemSchema from ORM objects
# ---------------------------------------------------------------------------


def _paper_to_trending(paper: Paper, rank: int) -> TrendingItemSchema:
    return TrendingItemSchema(
        entity_id=paper.id,
        entity_type="paper",
        rank=rank,
        title=paper.title,
        description=(paper.abstract or "")[:300] if paper.abstract else None,
        url=paper.url,
        trend_score=paper.trend_score or 0.0,
        metadata={
            "arxiv_id": paper.arxiv_id,
            "source": paper.source.value if paper.source else None,
            "citation_count": paper.citation_count,
            "has_implementation": paper.has_implementation,
            "categories": paper.categories,
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
        },
        updated_at=paper.updated_at,
    )


def _repo_to_trending(repo: Repository, rank: int) -> TrendingItemSchema:
    return TrendingItemSchema(
        entity_id=repo.id,
        entity_type="repository",
        rank=rank,
        title=repo.full_name,
        description=repo.description,
        url=repo.url,
        trend_score=repo.trend_score or 0.0,
        metadata={
            "language": repo.language,
            "stars": repo.stars,
            "forks": repo.forks,
            "stars_today": repo.stars_today,
            "topics": repo.topics,
        },
        updated_at=repo.updated_at,
    )


def _topic_to_trending(topic: Topic, rank: int) -> TrendingItemSchema:
    return TrendingItemSchema(
        entity_id=topic.id,
        entity_type="topic",
        rank=rank,
        title=topic.name,
        description=topic.description,
        url=None,
        trend_score=topic.trend_score or 0.0,
        metadata={
            "paper_count": topic.paper_count,
            "repo_count": topic.repo_count,
            "momentum": topic.momentum,
            "velocity": topic.velocity,
            "acceleration": topic.acceleration,
        },
        updated_at=topic.updated_at,
    )


# ---------------------------------------------------------------------------
# GET /trends/
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=TrendingListResponse,
    summary="Get ranked trending items",
    description=(
        "Returns a unified, ranked list of trending AI research entities "
        "(papers, repositories, and topics) sorted by computed trend score. "
        "Supports optional filtering by entity type and timeframe."
    ),
)
async def get_trending(
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    entity_type: Optional[str] = Query(
        None,
        description="Filter by entity type: paper | repository | topic",
    ),
    timeframe: str = Query(
        "7d",
        description="Time window for trend computation: 24h | 7d | 30d",
        regex="^(24h|7d|30d)$",
    ),
    db: AsyncSession = Depends(get_db),
) -> TrendingListResponse:
    """Return ranked trending items with optional entity_type and timeframe filters."""
    cache_svc = CacheService()
    cache_key = build_key(
        "trends",
        "list",
        str(entity_type),
        timeframe,
        str(limit),
        str(offset),
    )

    # Cache hit
    cached = await cache_svc.get(cache_key)
    if cached:
        return TrendingListResponse(**cached)

    days = TIMEFRAME_DAYS.get(timeframe, 7)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    items: List[TrendingItemSchema] = []

    # Fetch papers
    if entity_type in (None, "paper"):
        stmt = (
            select(Paper)
            .where(Paper.trend_score.isnot(None))
            .where(Paper.updated_at >= since)
            .order_by(desc(Paper.trend_score))
            .limit(limit if entity_type == "paper" else limit // 2 + 1)
        )
        result = await db.execute(stmt)
        papers = result.scalars().all()
        items.extend(_paper_to_trending(p, 0) for p in papers)

    # Fetch repos
    if entity_type in (None, "repository"):
        stmt = (
            select(Repository)
            .where(Repository.trend_score.isnot(None))
            .where(Repository.updated_at >= since)
            .order_by(desc(Repository.trend_score))
            .limit(limit if entity_type == "repository" else limit // 2 + 1)
        )
        result = await db.execute(stmt)
        repos = result.scalars().all()
        items.extend(_repo_to_trending(r, 0) for r in repos)

    # Fetch topics
    if entity_type in (None, "topic"):
        stmt = (
            select(Topic)
            .where(Topic.trend_score.isnot(None))
            .order_by(desc(Topic.trend_score))
            .limit(limit if entity_type == "topic" else max(5, limit // 5))
        )
        result = await db.execute(stmt)
        topics = result.scalars().all()
        items.extend(_topic_to_trending(t, 0) for t in topics)

    # Sort by trend_score descending and apply pagination
    items.sort(key=lambda x: x.trend_score, reverse=True)
    total = len(items)
    page_items = items[offset : offset + limit]

    # Assign ranks
    for i, item in enumerate(page_items, start=offset + 1):
        item.rank = i

    response = TrendingListResponse(
        items=page_items,
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
        timeframe=timeframe,
        entity_type=entity_type,
        generated_at=datetime.now(timezone.utc),
    )

    await cache_svc.set(cache_key, response.model_dump(mode="json"), ttl=60)
    return response


# ---------------------------------------------------------------------------
# GET /trends/topics
# ---------------------------------------------------------------------------


@router.get(
    "/topics",
    response_model=TopicListResponse,
    summary="Get trending topics with momentum",
)
async def get_trending_topics(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query(
        "trend_score",
        description="Sort field: trend_score | momentum | velocity | paper_count",
        regex="^(trend_score|momentum|velocity|paper_count)$",
    ),
    db: AsyncSession = Depends(get_db),
) -> TopicListResponse:
    """Return topics ranked by trend dynamics (momentum, velocity, trend_score)."""
    cache_svc = CacheService()
    cache_key = build_key("trends", "topics", sort_by, str(limit), str(offset))
    cached = await cache_svc.get(cache_key)
    if cached:
        return TopicListResponse(**cached)

    sort_column_map = {
        "trend_score": Topic.trend_score,
        "momentum": Topic.momentum,
        "velocity": Topic.velocity,
        "paper_count": Topic.paper_count,
    }
    sort_col = sort_column_map.get(sort_by, Topic.trend_score)

    count_stmt = select(func.count()).select_from(Topic)
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = (
        select(Topic)
        .where(sort_col.isnot(None))
        .order_by(desc(sort_col))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    topics = result.scalars().all()

    resp = TopicListResponse(
        items=[TopicSchema.model_validate(t) for t in topics],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=60)
    return resp


# ---------------------------------------------------------------------------
# GET /trends/{entity_id}/signals
# ---------------------------------------------------------------------------


@router.get(
    "/{entity_id}/signals",
    response_model=SignalSeriesSchema,
    summary="Get time-series signals for an entity",
)
async def get_entity_signals(
    entity_id: uuid.UUID,
    entity_type: str = Query(
        ...,
        description="paper | repository | topic",
        regex="^(paper|repository|topic)$",
    ),
    signal_type: Optional[str] = Query(
        None,
        description="Filter by signal type: mentions | stars | citations | engagement",
    ),
    days: int = Query(30, ge=1, le=365, description="Number of days of history to return"),
    db: AsyncSession = Depends(get_db),
) -> SignalSeriesSchema:
    """Return time-series signal data for a given entity."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date()

    stmt = (
        select(TrendSignal)
        .where(TrendSignal.entity_id == entity_id)
        .where(TrendSignal.entity_type == EntityType(entity_type))
        .where(TrendSignal.signal_date >= since)
        .order_by(TrendSignal.signal_date)
    )

    if signal_type:
        try:
            st = SignalType(signal_type)
            stmt = stmt.where(TrendSignal.signal_type == st)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid signal_type '{signal_type}'",
            )

    result = await db.execute(stmt)
    signals = result.scalars().all()

    data_points = [TrendSignalSchema.model_validate(s) for s in signals]

    return SignalSeriesSchema(
        entity_id=entity_id,
        entity_type=entity_type,
        signal_type=signal_type or "all",
        data_points=data_points,
        start_date=since,
        end_date=datetime.now(timezone.utc).date(),
        total_points=len(data_points),
    )


# ---------------------------------------------------------------------------
# GET /trends/dashboard
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard",
    response_model=DashboardSummarySchema,
    summary="Get dashboard summary statistics",
)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
) -> DashboardSummarySchema:
    """Return aggregated statistics for the main dashboard overview card."""
    cache_svc = CacheService()
    cache_key = build_key("dashboard", "summary")
    cached = await cache_svc.get(cache_key)
    if cached:
        return DashboardSummarySchema(**cached)

    # Entity counts
    paper_count = (await db.execute(select(func.count()).select_from(Paper))).scalar_one()
    repo_count = (await db.execute(select(func.count()).select_from(Repository))).scalar_one()
    topic_count = (await db.execute(select(func.count()).select_from(Topic))).scalar_one()

    # Top trending papers (limit 5)
    top_papers_result = await db.execute(
        select(Paper)
        .where(Paper.trend_score.isnot(None))
        .order_by(desc(Paper.trend_score))
        .limit(5)
    )
    top_papers = [
        _paper_to_trending(p, i + 1)
        for i, p in enumerate(top_papers_result.scalars().all())
    ]

    # Top trending repos (limit 5)
    top_repos_result = await db.execute(
        select(Repository)
        .where(Repository.trend_score.isnot(None))
        .order_by(desc(Repository.trend_score))
        .limit(5)
    )
    top_repos = [
        _repo_to_trending(r, i + 1)
        for i, r in enumerate(top_repos_result.scalars().all())
    ]

    # Top trending topics (limit 5)
    top_topics_result = await db.execute(
        select(Topic)
        .where(Topic.trend_score.isnot(None))
        .order_by(desc(Topic.trend_score))
        .limit(5)
    )
    top_topics_orm = top_topics_result.scalars().all()
    top_topics = [TopicSchema.model_validate(t) for t in top_topics_orm]

    # Hottest topic
    hottest_topic = top_topics[0] if top_topics else None

    # Papers ingested today
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    papers_today = (
        await db.execute(
            select(func.count()).select_from(Paper).where(Paper.created_at >= today_start)
        )
    ).scalar_one()
    repos_today = (
        await db.execute(
            select(func.count())
            .select_from(Repository)
            .where(Repository.created_at >= today_start)
        )
    ).scalar_one()

    summary = DashboardSummarySchema(
        total_entities=EntityCountSchema(
            papers=paper_count,
            repositories=repo_count,
            topics=topic_count,
        ),
        top_trending_papers=top_papers,
        top_trending_repos=top_repos,
        top_trending_topics=top_topics,
        hottest_topic=hottest_topic,
        papers_ingested_today=papers_today,
        repos_ingested_today=repos_today,
        generated_at=datetime.now(timezone.utc),
    )

    await cache_svc.set(cache_key, summary.model_dump(mode="json"), ttl=120)
    return summary
