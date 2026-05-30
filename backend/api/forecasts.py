"""
Forecasts API router.

Endpoints:
  GET  /api/v1/forecasts/{entity_id}      - Get forecast series for an entity
  GET  /api/v1/forecasts/topics/top       - Top trending topic forecasts
  POST /api/v1/forecasts/compute          - Trigger forecast computation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.models.insight import ForecastResult
from backend.models.topic import Topic
from backend.models.trend_signal import EntityType
from backend.schemas.trend import ForecastPointSchema, ForecastSeriesSchema, TopicSchema
from backend.services.cache_service import CacheService, build_key
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/forecasts",
    tags=["Forecasts"],
    responses={404: {"description": "Entity or forecast not found"}},
)


# ---------------------------------------------------------------------------
# Request / Response helpers
# ---------------------------------------------------------------------------


class ForecastComputeRequest(BaseModel):
    entity_id: uuid.UUID = Field(..., description="Entity UUID to compute forecast for")
    entity_type: str = Field(..., description="paper | repository | topic")
    horizon_days: int = Field(30, ge=1, le=365, description="Forecast horizon in days")
    model: str = Field("prophet", description="Model to use: prophet | arima | lstm")
    force_recompute: bool = Field(False, description="Recompute even if a fresh forecast exists")


class ForecastComputeResponse(BaseModel):
    job_id: str
    entity_id: uuid.UUID
    entity_type: str
    status: str
    queued_at: datetime


# ---------------------------------------------------------------------------
# GET /forecasts/topics/top  (must be BEFORE /{entity_id})
# ---------------------------------------------------------------------------


@router.get(
    "/topics/top",
    response_model=List[Dict[str, Any]],
    summary="Get top trending topic forecasts",
)
async def get_top_topic_forecasts(
    limit: int = Query(10, ge=1, le=50),
    horizon_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Return forecast series for the top ``limit`` trending topics.

    Topics are ranked by their current ``trend_score``, then enriched with
    the most recent forecast series stored in ``forecast_results``.
    """
    cache_svc = CacheService()
    cache_key = build_key("forecasts", "topics_top", str(limit), str(horizon_days))
    cached = await cache_svc.get(cache_key)
    if cached:
        return cached

    # Fetch top topics
    topics_result = await db.execute(
        select(Topic)
        .where(Topic.trend_score.isnot(None))
        .order_by(desc(Topic.trend_score))
        .limit(limit)
    )
    topics = topics_result.scalars().all()

    results = []
    for topic in topics:
        # Fetch stored forecast points
        fr_result = await db.execute(
            select(ForecastResult)
            .where(ForecastResult.entity_id == topic.id)
            .where(ForecastResult.entity_type == EntityType.TOPIC)
            .order_by(ForecastResult.forecast_date)
        )
        forecast_rows = fr_result.scalars().all()

        points = [
            ForecastPointSchema(
                forecast_date=fr.forecast_date,
                predicted_value=fr.predicted_value,
                confidence_lower=fr.confidence_lower,
                confidence_upper=fr.confidence_upper,
            )
            for fr in forecast_rows
        ]

        results.append({
            "topic": TopicSchema.model_validate(topic).model_dump(mode="json"),
            "forecast": {
                "entity_id": str(topic.id),
                "entity_type": "topic",
                "model_used": forecast_rows[0].model_used if forecast_rows else "unknown",
                "horizon_days": horizon_days,
                "points": [p.model_dump(mode="json") for p in points],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        })

    await cache_svc.set(cache_key, results, ttl=300)
    return results


# ---------------------------------------------------------------------------
# GET /forecasts/{entity_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{entity_id}",
    response_model=ForecastSeriesSchema,
    summary="Get forecast series for an entity",
)
async def get_entity_forecast(
    entity_id: uuid.UUID,
    entity_type: str = Query(
        ...,
        description="paper | repository | topic",
        regex="^(paper|repository|topic)$",
    ),
    db: AsyncSession = Depends(get_db),
) -> ForecastSeriesSchema:
    """
    Return the stored forecast series for a given entity.

    If no forecast exists, raises HTTP 404.
    """
    try:
        et = EntityType(entity_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid entity_type '{entity_type}'")

    result = await db.execute(
        select(ForecastResult)
        .where(ForecastResult.entity_id == entity_id)
        .where(ForecastResult.entity_type == et)
        .order_by(ForecastResult.forecast_date)
    )
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for {entity_type} {entity_id}. "
            "Trigger computation via POST /api/v1/forecasts/compute",
        )

    points = [
        ForecastPointSchema(
            forecast_date=r.forecast_date,
            predicted_value=r.predicted_value,
            confidence_lower=r.confidence_lower,
            confidence_upper=r.confidence_upper,
        )
        for r in rows
    ]

    return ForecastSeriesSchema(
        entity_id=entity_id,
        entity_type=entity_type,
        model_used=rows[0].model_used,
        horizon_days=rows[0].horizon_days or len(rows),
        points=points,
        generated_at=rows[-1].created_at,
    )


# ---------------------------------------------------------------------------
# POST /forecasts/compute
# ---------------------------------------------------------------------------


@router.post(
    "/compute",
    response_model=ForecastComputeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger forecast computation",
)
async def compute_forecast(
    request: ForecastComputeRequest,
    db: AsyncSession = Depends(get_db),
) -> ForecastComputeResponse:
    """
    Enqueue a forecast computation job for the specified entity.

    Returns immediately with a ``job_id`` that can be used to track progress.
    The forecast will be stored in ``forecast_results`` upon completion and
    available via ``GET /api/v1/forecasts/{entity_id}``.
    """
    import asyncio

    try:
        et = EntityType(request.entity_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid entity_type '{request.entity_type}'")

    job_id = str(uuid.uuid4())
    logger.info(
        "Forecast computation queued",
        extra={
            "job_id": job_id,
            "entity_id": str(request.entity_id),
            "entity_type": request.entity_type,
            "model": request.model,
            "horizon_days": request.horizon_days,
        },
    )

    # In a real system, push to a Celery / ARQ task queue.
    # Here we register a background task placeholder.
    async def _compute() -> None:
        """Placeholder for real forecast computation."""
        await asyncio.sleep(1)
        logger.info("Forecast computation completed (stub)", extra={"job_id": job_id})

    asyncio.create_task(_compute(), name=f"forecast-{job_id[:8]}")

    return ForecastComputeResponse(
        job_id=job_id,
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        status="queued",
        queued_at=datetime.now(timezone.utc),
    )
