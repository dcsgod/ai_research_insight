"""
Ingestion control API router.

Endpoints:
  POST /api/v1/ingestion/run              - Trigger full ingestion across all sources
  POST /api/v1/ingestion/run/{source}     - Trigger ingestion for a specific source
  GET  /api/v1/ingestion/status           - Get current or latest ingestion status
  GET  /api/v1/ingestion/history          - Get ingestion history (recent jobs)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from backend.services.ingestion_service import IngestionService
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/ingestion",
    tags=["Ingestion"],
    responses={
        400: {"description": "Invalid source or request"},
        500: {"description": "Ingestion service error"},
    },
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

VALID_SOURCES = {"arxiv", "github", "huggingface", "paperswithcode", "reddit"}


class IngestionRunRequest(BaseModel):
    """Request body for triggering an ingestion run."""

    sources: Optional[List[str]] = Field(
        None,
        description="Sources to ingest from. Omit for all sources.",
        example=["arxiv", "github"],
    )
    force_refresh: bool = Field(
        False,
        description="Re-ingest records even if they already exist in the DB",
    )
    max_results_per_source: Optional[int] = Field(
        None,
        ge=10,
        le=2000,
        description="Override the default max results per source",
    )


class IngestionRunResponse(BaseModel):
    """Response returned when an ingestion job is queued."""

    job_id: str
    status: str
    sources: List[str]
    queued_at: datetime
    estimated_duration_seconds: Optional[int] = None
    message: str = "Ingestion queued. Poll /api/v1/ingestion/status for progress."


class IngestionStatusResponse(BaseModel):
    """Current ingestion status."""

    job_id: Optional[str] = None
    status: str
    sources: Optional[List[str]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_upserted: Optional[int] = None
    total_signals_written: Optional[int] = None
    processed: Optional[Dict[str, int]] = None
    errors: Optional[Dict[str, int]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /ingestion/run
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=IngestionRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger full ingestion",
    description="Queue a full ingestion cycle across all (or specified) data sources.",
)
async def run_ingestion(request: IngestionRunRequest) -> IngestionRunResponse:
    """Start an ingestion job and return immediately with a job_id."""
    # Validate requested sources
    sources = request.sources or list(VALID_SOURCES)
    invalid = [s for s in sources if s not in VALID_SOURCES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sources: {invalid}. Must be from {VALID_SOURCES}",
        )

    svc = IngestionService()
    try:
        job_id = await svc.run_ingestion(
            sources=sources,
            force_refresh=request.force_refresh,
        )
    except Exception as exc:
        logger.error("Failed to queue ingestion job", exc_info=exc)
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {exc}")

    return IngestionRunResponse(
        job_id=job_id,
        status="queued",
        sources=sources,
        queued_at=datetime.now(timezone.utc),
        estimated_duration_seconds=len(sources) * 45,
    )


# ---------------------------------------------------------------------------
# POST /ingestion/run/{source}
# ---------------------------------------------------------------------------


@router.post(
    "/run/{source}",
    response_model=IngestionRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger ingestion for a specific source",
)
async def run_source_ingestion(
    source: str = Path(
        ...,
        description="Source to ingest: arxiv | github | huggingface | paperswithcode | reddit",
    ),
    force_refresh: bool = Query(False),
) -> IngestionRunResponse:
    """Queue an ingestion job for a single data source."""
    if source not in VALID_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source '{source}'. Valid sources: {VALID_SOURCES}",
        )

    svc = IngestionService()
    try:
        job_id = await svc.run_ingestion(sources=[source], force_refresh=force_refresh)
    except Exception as exc:
        logger.error("Failed to queue source ingestion", exc_info=exc, extra={"source": source})
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestionRunResponse(
        job_id=job_id,
        status="queued",
        sources=[source],
        queued_at=datetime.now(timezone.utc),
        estimated_duration_seconds=60,
    )


# ---------------------------------------------------------------------------
# GET /ingestion/status
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    response_model=IngestionStatusResponse,
    summary="Get ingestion status",
    description="Return the status of a specific job, or the most recent run if no job_id provided.",
)
async def get_ingestion_status(
    job_id: Optional[str] = Query(None, description="Specific job UUID to query"),
) -> IngestionStatusResponse:
    """Return ingestion status."""
    svc = IngestionService()

    try:
        state = await svc.get_status(job_id=job_id)
    except Exception as exc:
        logger.error("Failed to fetch ingestion status", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not state:
        return IngestionStatusResponse(status="idle")

    def _parse_dt(v: Any) -> Optional[datetime]:
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                return None
        return v

    return IngestionStatusResponse(
        job_id=state.get("job_id"),
        status=state.get("status", "unknown"),
        sources=state.get("sources"),
        started_at=_parse_dt(state.get("started_at")),
        completed_at=_parse_dt(state.get("completed_at")),
        total_upserted=state.get("total_upserted"),
        total_signals_written=state.get("total_signals_written"),
        processed=state.get("processed"),
        errors=state.get("errors"),
        error=state.get("error"),
    )


# ---------------------------------------------------------------------------
# GET /ingestion/history
# ---------------------------------------------------------------------------


@router.get(
    "/history",
    response_model=List[Dict[str, Any]],
    summary="Get ingestion history",
    description="Return the N most recent ingestion job summaries, newest first.",
)
async def get_ingestion_history(
    limit: int = Query(10, ge=1, le=50, description="Number of history entries to return"),
) -> List[Dict[str, Any]]:
    """Return a list of recent ingestion job summaries."""
    svc = IngestionService()
    try:
        history = await svc.get_history(limit=limit)
    except Exception as exc:
        logger.error("Failed to fetch ingestion history", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return history
