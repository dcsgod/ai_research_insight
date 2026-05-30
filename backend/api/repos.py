"""
GitHub Repositories API router.

Endpoints:
  GET /api/v1/repos/           - List repos with filtering
  GET /api/v1/repos/trending   - Trending repos by language/timeframe
  GET /api/v1/repos/search     - Search repos by name/description
  GET /api/v1/repos/{repo_id}  - Get repository details
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, func, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.models.repository import Repository
from backend.models.trend_signal import TrendScore, EntityType
from backend.services.cache_service import CacheService, build_key
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/repos",
    tags=["Repositories"],
    responses={404: {"description": "Repository not found"}},
)


# ---------------------------------------------------------------------------
# Response schemas (inline for simplicity; move to schemas/ for larger projects)
# ---------------------------------------------------------------------------


class RepositoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    github_id: Optional[int] = None
    name: str
    full_name: str
    owner: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    language: Optional[str] = None
    topics: Optional[List[str]] = None
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    stars_today: Optional[int] = None
    open_issues_count: int = 0
    trend_score: Optional[float] = None
    last_ingested: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class RepositoryWithScore(RepositoryResponse):
    growth_velocity: Optional[float] = None
    github_activity: Optional[float] = None
    community_engagement: Optional[float] = None
    final_score: Optional[float] = None


class RepositoryListResponse(BaseModel):
    items: List[RepositoryResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# ---------------------------------------------------------------------------
# GET /repos/
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=RepositoryListResponse,
    summary="List repositories with filtering",
)
async def list_repos(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    language: Optional[str] = Query(None, description="Filter by primary language"),
    min_stars: Optional[int] = Query(None, ge=0, description="Minimum star count"),
    min_trend_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    sort_by: str = Query(
        "trend_score",
        regex="^(trend_score|stars|stars_today|forks|updated_at|created_at)$",
    ),
    db: AsyncSession = Depends(get_db),
) -> RepositoryListResponse:
    """Return a paginated, filtered list of tracked GitHub repositories."""
    cache_svc = CacheService()
    cache_key = build_key(
        "repos", "list",
        language or "all", str(min_stars), str(min_trend_score),
        sort_by, str(limit), str(offset),
    )
    cached = await cache_svc.get(cache_key)
    if cached:
        return RepositoryListResponse(**cached)

    filters = []
    if language:
        filters.append(func.lower(Repository.language) == language.lower())
    if min_stars is not None:
        filters.append(Repository.stars >= min_stars)
    if min_trend_score is not None:
        filters.append(Repository.trend_score >= min_trend_score)

    sort_map = {
        "trend_score": desc(Repository.trend_score),
        "stars": desc(Repository.stars),
        "stars_today": desc(Repository.stars_today),
        "forks": desc(Repository.forks),
        "updated_at": desc(Repository.updated_at),
        "created_at": desc(Repository.created_at),
    }
    order = sort_map.get(sort_by, desc(Repository.trend_score))

    count_stmt = select(func.count()).select_from(Repository)
    if filters:
        count_stmt = count_stmt.where(and_(*filters))
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = select(Repository).order_by(order).offset(offset).limit(limit)
    if filters:
        stmt = stmt.where(and_(*filters))

    result = await db.execute(stmt)
    repos = result.scalars().all()

    resp = RepositoryListResponse(
        items=[RepositoryResponse.model_validate(r) for r in repos],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=120)
    return resp


# ---------------------------------------------------------------------------
# GET /repos/trending  (must be before /{repo_id})
# ---------------------------------------------------------------------------


@router.get(
    "/trending",
    response_model=RepositoryListResponse,
    summary="Get trending repositories",
)
async def get_trending_repos(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    language: Optional[str] = Query(None),
    timeframe: str = Query("7d", regex="^(24h|7d|30d)$"),
    db: AsyncSession = Depends(get_db),
) -> RepositoryListResponse:
    """
    Return trending repositories, optionally filtered by language and timeframe.

    Timeframe filters on ``updated_at`` to surface recently-active repos.
    """
    timeframe_days = {"24h": 1, "7d": 7, "30d": 30}
    since = datetime.now(timezone.utc) - timedelta(days=timeframe_days[timeframe])

    cache_svc = CacheService()
    cache_key = build_key("repos", "trending", language or "all", timeframe, str(limit), str(offset))
    cached = await cache_svc.get(cache_key)
    if cached:
        return RepositoryListResponse(**cached)

    filters = [
        Repository.trend_score.isnot(None),
        Repository.updated_at >= since,
    ]
    if language:
        filters.append(func.lower(Repository.language) == language.lower())

    count_stmt = select(func.count()).select_from(Repository).where(and_(*filters))
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = (
        select(Repository)
        .where(and_(*filters))
        .order_by(desc(Repository.trend_score), desc(Repository.stars_today))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()

    resp = RepositoryListResponse(
        items=[RepositoryResponse.model_validate(r) for r in repos],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=60)
    return resp


# ---------------------------------------------------------------------------
# GET /repos/search  (must be before /{repo_id})
# ---------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=RepositoryListResponse,
    summary="Search repositories",
)
async def search_repos(
    q: str = Query(..., min_length=1, max_length=512),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    language: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> RepositoryListResponse:
    """Search repositories by name, description, or topic keywords."""
    filters = [
        or_(
            Repository.name.ilike(f"%{q}%"),
            Repository.full_name.ilike(f"%{q}%"),
            Repository.description.ilike(f"%{q}%"),
        )
    ]
    if language:
        filters.append(func.lower(Repository.language) == language.lower())

    count_stmt = select(func.count()).select_from(Repository).where(and_(*filters))
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = (
        select(Repository)
        .where(and_(*filters))
        .order_by(desc(Repository.stars), desc(Repository.trend_score))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()

    return RepositoryListResponse(
        items=[RepositoryResponse.model_validate(r) for r in repos],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
    )


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{repo_id}",
    response_model=RepositoryWithScore,
    summary="Get repository details",
)
async def get_repo(
    repo_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> RepositoryWithScore:
    """Return full details for a single repository including trend score factors."""
    cache_svc = CacheService()
    cache_key = build_key("repos", "detail", str(repo_id))
    cached = await cache_svc.get(cache_key)
    if cached:
        return RepositoryWithScore(**cached)

    result = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = result.scalar_one_or_none()
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repository {repo_id} not found")

    # Fetch trend score breakdown
    ts_result = await db.execute(
        select(TrendScore).where(TrendScore.entity_id == repo_id)
    )
    ts = ts_result.scalar_one_or_none()

    resp = RepositoryWithScore(
        **RepositoryResponse.model_validate(repo).model_dump(),
        growth_velocity=ts.growth_velocity if ts else None,
        github_activity=ts.github_activity if ts else None,
        community_engagement=ts.community_engagement if ts else None,
        final_score=ts.final_score if ts else None,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=300)
    return resp
