"""
LLM Insights API router.

Endpoints:
  GET  /api/v1/insights/{entity_id}  - Get existing insight for an entity
  POST /api/v1/insights/generate     - Generate a new LLM insight
  GET  /api/v1/insights/daily        - Get the daily AI market summary
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.models.insight import Insight
from backend.models.trend_signal import EntityType
from backend.services.cache_service import CacheService, build_key
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(
    prefix="/insights",
    tags=["Insights"],
    responses={404: {"description": "Insight not found"}},
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class InsightResponse(BaseModel):
    """Response schema for a single LLM insight."""

    id: uuid.UUID
    entity_id: uuid.UUID
    entity_type: str
    insight_text: str
    insight_type: str
    llm_model: str
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    created_at: datetime


class InsightGenerateRequest(BaseModel):
    """Request body for generating a new insight."""

    entity_id: uuid.UUID = Field(..., description="UUID of the entity to analyse")
    entity_type: str = Field(..., description="paper | repository | topic")
    insight_type: str = Field(
        "summary",
        description="summary | analysis | forecast_narrative",
    )
    force_regenerate: bool = Field(
        False,
        description="Generate even if a recent insight already exists",
    )


class InsightGenerateResponse(BaseModel):
    """Response when an insight generation job is started."""

    job_id: str
    entity_id: uuid.UUID
    entity_type: str
    status: str
    queued_at: datetime
    message: str = "Insight generation queued. Poll GET /api/v1/insights/{entity_id}"


class DailyInsightResponse(BaseModel):
    """Daily AI market summary."""

    summary_date: date
    insight_text: str
    llm_model: str
    tokens_used: Optional[int] = None
    top_papers: List[Dict[str, Any]] = Field(default_factory=list)
    top_topics: List[str] = Field(default_factory=list)
    created_at: datetime


# ---------------------------------------------------------------------------
# GET /insights/daily  (must come BEFORE /{entity_id})
# ---------------------------------------------------------------------------


@router.get(
    "/daily",
    response_model=DailyInsightResponse,
    summary="Get daily AI market summary",
)
async def get_daily_insight(
    summary_date: Optional[date] = Query(
        None,
        description="Date of the summary (defaults to today UTC)",
    ),
    db: AsyncSession = Depends(get_db),
) -> DailyInsightResponse:
    """
    Return the LLM-generated daily market summary for AI research activity.

    If no summary exists for the requested date, returns a 404 with a suggestion
    to trigger generation.
    """
    target_date = summary_date or datetime.now(timezone.utc).date()

    cache_svc = CacheService()
    cache_key = build_key("insights", "daily", str(target_date))
    cached = await cache_svc.get(cache_key)
    if cached:
        return DailyInsightResponse(**cached)

    # Fetch daily digest insight from DB
    stmt = (
        select(Insight)
        .where(Insight.insight_type == "daily_digest")
        .where(func.date(Insight.created_at) == target_date)
        .order_by(desc(Insight.created_at))
        .limit(1)
    )

    from sqlalchemy import func
    result = await db.execute(stmt)
    insight = result.scalar_one_or_none()

    if insight is None:
        raise HTTPException(
            status_code=404,
            detail=f"No daily insight found for {target_date}. "
            "Generate one via POST /api/v1/insights/generate with insight_type='daily_digest'",
        )

    resp = DailyInsightResponse(
        summary_date=target_date,
        insight_text=insight.insight_text,
        llm_model=insight.llm_model,
        tokens_used=insight.tokens_used,
        top_papers=[],
        top_topics=[],
        created_at=insight.created_at,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=3600)
    return resp


# ---------------------------------------------------------------------------
# GET /insights/{entity_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{entity_id}",
    response_model=InsightResponse,
    summary="Get existing insight for an entity",
)
async def get_insight(
    entity_id: uuid.UUID,
    entity_type: str = Query(
        ...,
        description="paper | repository | topic",
        regex="^(paper|repository|topic)$",
    ),
    insight_type: Optional[str] = Query(
        None,
        description="Filter by insight type (e.g. 'summary', 'analysis')",
    ),
    db: AsyncSession = Depends(get_db),
) -> InsightResponse:
    """Return the most recent insight for the given entity."""
    try:
        et = EntityType(entity_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid entity_type '{entity_type}'")

    stmt = (
        select(Insight)
        .where(Insight.entity_id == entity_id)
        .where(Insight.entity_type == et)
        .order_by(desc(Insight.created_at))
        .limit(1)
    )
    if insight_type:
        stmt = stmt.where(Insight.insight_type == insight_type)

    result = await db.execute(stmt)
    insight = result.scalar_one_or_none()

    if insight is None:
        raise HTTPException(
            status_code=404,
            detail=f"No insight found for {entity_type} {entity_id}. "
            "Generate via POST /api/v1/insights/generate",
        )

    return InsightResponse(
        id=insight.id,
        entity_id=insight.entity_id,
        entity_type=insight.entity_type.value,
        insight_text=insight.insight_text,
        insight_type=insight.insight_type,
        llm_model=insight.llm_model,
        tokens_used=insight.tokens_used,
        prompt_tokens=insight.prompt_tokens,
        completion_tokens=insight.completion_tokens,
        created_at=insight.created_at,
    )


# ---------------------------------------------------------------------------
# POST /insights/generate
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=InsightGenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate a new LLM insight",
)
async def generate_insight(
    request: InsightGenerateRequest,
    db: AsyncSession = Depends(get_db),
) -> InsightGenerateResponse:
    """
    Queue an LLM insight generation job for the specified entity.

    If an insight already exists and ``force_regenerate=False``, returns a
    409 Conflict pointing to the existing insight endpoint.
    """
    try:
        et = EntityType(request.entity_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid entity_type '{request.entity_type}'")

    # Check for existing fresh insight (within last 24h)
    if not request.force_regenerate:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        stmt = (
            select(Insight)
            .where(Insight.entity_id == request.entity_id)
            .where(Insight.entity_type == et)
            .where(Insight.created_at >= cutoff)
            .limit(1)
        )
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A fresh insight already exists for this entity (created {existing.created_at}). "
                    "Set force_regenerate=true to override."
                ),
            )

    job_id = str(uuid.uuid4())

    logger.info(
        "Insight generation queued",
        extra={
            "job_id": job_id,
            "entity_id": str(request.entity_id),
            "entity_type": request.entity_type,
            "insight_type": request.insight_type,
            "model": settings.LLM_MODEL,
        },
    )

    # In production, push to task queue (Celery / ARQ / Redis Queue)
    # Here we acknowledge with a stub job_id
    return InsightGenerateResponse(
        job_id=job_id,
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        status="queued",
        queued_at=datetime.now(timezone.utc),
    )
