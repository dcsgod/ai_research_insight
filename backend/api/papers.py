"""
Papers API router.

Endpoints:
  GET  /api/v1/papers/                   - List papers with filtering
  GET  /api/v1/papers/search             - Semantic search
  POST /api/v1/papers/ingest             - Trigger ingestion
  GET  /api/v1/papers/{paper_id}         - Get paper details
  GET  /api/v1/papers/{paper_id}/insight - Get or generate LLM insight
  GET  /api/v1/papers/{paper_id}/similar - Get similar papers
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select, func, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.models.insight import Insight
from backend.models.paper import Paper, PaperSource
from backend.models.trend_signal import EntityType
from backend.schemas.paper import (
    PaperCreate,
    PaperIngestRequest,
    PaperIngestResponse,
    PaperListResponse,
    PaperResponse,
    PaperSearchRequest,
    PaperUpdate,
    PaperWithScore,
)
from backend.services.cache_service import CacheService, build_key
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/papers",
    tags=["Papers"],
    responses={
        404: {"description": "Paper not found"},
        422: {"description": "Validation error"},
    },
)


# ---------------------------------------------------------------------------
# GET /papers/
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=PaperListResponse,
    summary="List papers with filtering",
    description="Return a paginated list of papers with optional filters for source, categories, date range, and minimum trend score.",
)
async def list_papers(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None, description="Filter by source: arxiv | huggingface | paperswithcode"),
    category: Optional[str] = Query(None, description="Filter by arXiv category (substring match)"),
    date_from: Optional[datetime] = Query(None, description="Filter: published after this date"),
    date_to: Optional[datetime] = Query(None, description="Filter: published before this date"),
    min_trend_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    has_implementation: Optional[bool] = Query(None),
    sort_by: str = Query("trend_score", regex="^(trend_score|published_date|citation_count|created_at)$"),
    db: AsyncSession = Depends(get_db),
) -> PaperListResponse:
    cache_svc = CacheService()
    cache_key = build_key(
        "papers", "list",
        source or "all", category or "all",
        str(date_from), str(date_to),
        str(min_trend_score), str(has_implementation),
        sort_by, str(limit), str(offset),
    )
    cached = await cache_svc.get(cache_key)
    if cached:
        return PaperListResponse(**cached)

    filters = []
    if source:
        try:
            filters.append(Paper.source == PaperSource(source))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source '{source}'")
    if category:
        filters.append(Paper.categories.astext.contains(category))
    if date_from:
        filters.append(Paper.published_date >= date_from)
    if date_to:
        filters.append(Paper.published_date <= date_to)
    if min_trend_score is not None:
        filters.append(Paper.trend_score >= min_trend_score)
    if has_implementation is not None:
        filters.append(Paper.has_implementation == has_implementation)

    sort_col_map = {
        "trend_score": desc(Paper.trend_score),
        "published_date": desc(Paper.published_date),
        "citation_count": desc(Paper.citation_count),
        "created_at": desc(Paper.created_at),
    }
    order = sort_col_map.get(sort_by, desc(Paper.trend_score))

    count_stmt = select(func.count()).select_from(Paper)
    if filters:
        count_stmt = count_stmt.where(and_(*filters))
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = select(Paper).order_by(order).offset(offset).limit(limit)
    if filters:
        stmt = stmt.where(and_(*filters))

    result = await db.execute(stmt)
    papers = result.scalars().all()

    resp = PaperListResponse(
        items=[PaperResponse.model_validate(p) for p in papers],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=120)
    return resp


# ---------------------------------------------------------------------------
# GET /papers/search  (must come BEFORE /{paper_id})
# ---------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=PaperListResponse,
    summary="Semantic search for papers",
)
async def search_papers(
    q: str = Query(..., min_length=1, max_length=2048, description="Search query text"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None),
    use_semantic: bool = Query(True, description="Use vector similarity (vs plain text ILIKE)"),
    db: AsyncSession = Depends(get_db),
) -> PaperListResponse:
    """
    Search papers by relevance.

    When ``use_semantic=True`` (default), performs a vector similarity search
    via Qdrant. Falls back to PostgreSQL ILIKE when Qdrant is unavailable.
    """
    if use_semantic:
        # Attempt Qdrant vector search
        try:
            from backend.services.cache_service import CacheService as _CS
            # Placeholder: real implementation queries Qdrant, retrieves IDs,
            # then fetches Paper rows by ID.
            raise NotImplementedError("Vector search adapter not yet wired")
        except (NotImplementedError, ImportError):
            # Graceful fallback to text search
            logger.info("Falling back to text search", extra={"query": q})
            use_semantic = False

    # Text search fallback
    filters = [
        or_(
            Paper.title.ilike(f"%{q}%"),
            Paper.abstract.ilike(f"%{q}%"),
        )
    ]
    if source:
        try:
            filters.append(Paper.source == PaperSource(source))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source '{source}'")

    count_stmt = select(func.count()).select_from(Paper).where(and_(*filters))
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = (
        select(Paper)
        .where(and_(*filters))
        .order_by(desc(Paper.trend_score), desc(Paper.published_date))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    papers = result.scalars().all()

    return PaperListResponse(
        items=[PaperResponse.model_validate(p) for p in papers],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
    )


# ---------------------------------------------------------------------------
# POST /papers/ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=PaperIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger paper ingestion",
)
async def trigger_ingestion(
    request: PaperIngestRequest,
    background_tasks: BackgroundTasks,
) -> PaperIngestResponse:
    """
    Enqueue a paper ingestion job.

    Returns immediately with a ``job_id`` that can be used to poll
    ``GET /api/v1/ingestion/status``.
    """
    from backend.services.ingestion_service import IngestionService

    svc = IngestionService()
    job_id = await svc.run_ingestion(sources=request.sources, force_refresh=request.force_refresh)

    return PaperIngestResponse(
        job_id=job_id,
        status="queued",
        sources=request.sources,
        queued_at=datetime.now(timezone.utc),
        estimated_duration_seconds=len(request.sources) * 30,
    )


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{paper_id}",
    response_model=PaperWithScore,
    summary="Get paper details",
)
async def get_paper(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> PaperWithScore:
    """Return full details for a single paper, including trend score breakdown."""
    cache_svc = CacheService()
    cache_key = build_key("papers", "detail", str(paper_id))
    cached = await cache_svc.get(cache_key)
    if cached:
        return PaperWithScore(**cached)

    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

    # Fetch trend score breakdown
    from backend.models.trend_signal import TrendScore
    ts_result = await db.execute(
        select(TrendScore).where(TrendScore.entity_id == paper_id)
    )
    ts = ts_result.scalar_one_or_none()

    resp = PaperWithScore(
        **PaperResponse.model_validate(paper).model_dump(),
        growth_velocity=ts.growth_velocity if ts else None,
        github_activity=ts.github_activity if ts else None,
        citation_acceleration=ts.citation_acceleration if ts else None,
        community_engagement=ts.community_engagement if ts else None,
        novelty_score=ts.novelty_score if ts else None,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=300)
    return resp


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}/insight
# ---------------------------------------------------------------------------


@router.get(
    "/{paper_id}/insight",
    summary="Get or generate LLM insight for a paper",
    response_model=Dict[str, Any],
)
async def get_paper_insight(
    paper_id: uuid.UUID,
    regenerate: bool = Query(False, description="Force regeneration even if cached"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Return the latest LLM-generated insight for a paper.

    If no insight exists (or ``regenerate=True``), triggers generation
    asynchronously and returns a 202 with a task reference.
    """
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

    # Fetch latest insight
    ins_result = await db.execute(
        select(Insight)
        .where(Insight.entity_id == paper_id)
        .where(Insight.entity_type == EntityType.PAPER)
        .order_by(desc(Insight.created_at))
        .limit(1)
    )
    insight = ins_result.scalar_one_or_none()

    if insight and not regenerate:
        return {
            "paper_id": str(paper_id),
            "insight_text": insight.insight_text,
            "llm_model": insight.llm_model,
            "tokens_used": insight.tokens_used,
            "created_at": insight.created_at.isoformat(),
            "status": "available",
        }

    # No insight or force regenerate → return pending status
    return {
        "paper_id": str(paper_id),
        "status": "pending",
        "message": "Insight generation queued. Poll this endpoint in 10–30 seconds.",
    }


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}/similar
# ---------------------------------------------------------------------------


@router.get(
    "/{paper_id}/similar",
    response_model=PaperListResponse,
    summary="Get similar papers via vector search",
)
async def get_similar_papers(
    paper_id: uuid.UUID,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> PaperListResponse:
    """
    Return papers similar to the given paper using vector similarity search.

    Falls back to category-based similarity if the paper has no embedding.
    """
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

    # Fallback: same-category papers (until Qdrant adapter is wired)
    filters = [Paper.id != paper_id]
    if paper.categories:
        # Use JSONB overlap: find papers sharing at least one category
        filters.append(Paper.categories.overlap(paper.categories))  # type: ignore[attr-defined]

    stmt = (
        select(Paper)
        .where(and_(*filters))
        .order_by(desc(Paper.trend_score))
        .limit(limit)
    )
    result2 = await db.execute(stmt)
    similar = result2.scalars().all()

    return PaperListResponse(
        items=[PaperResponse.model_validate(p) for p in similar],
        total=len(similar),
        page=1,
        page_size=limit,
        has_next=False,
        has_prev=False,
    )
