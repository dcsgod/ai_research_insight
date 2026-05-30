"""
Topics API router.

Endpoints:
  GET /api/v1/topics/                     - List all topics sorted by trend_score
  GET /api/v1/topics/graph                - Topic relationship graph for D3
  GET /api/v1/topics/{topic_id}           - Get topic with papers/repos
  GET /api/v1/topics/{topic_id}/forecast  - Get topic forecast
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.models.insight import ForecastResult
from backend.models.paper import Paper
from backend.models.repository import Repository
from backend.models.topic import Topic
from backend.models.trend_signal import EntityType
from backend.schemas.trend import (
    ForecastPointSchema,
    ForecastSeriesSchema,
    TopicGraphEdge,
    TopicGraphNode,
    TopicGraphSchema,
    TopicListResponse,
    TopicSchema,
    TopicWithRelations,
)
from backend.services.cache_service import CacheService, build_key
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/topics",
    tags=["Topics"],
    responses={404: {"description": "Topic not found"}},
)


# ---------------------------------------------------------------------------
# GET /topics/
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=TopicListResponse,
    summary="List all topics sorted by trend_score",
)
async def list_topics(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query(
        "trend_score",
        regex="^(trend_score|momentum|velocity|paper_count|name)$",
    ),
    min_paper_count: Optional[int] = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
) -> TopicListResponse:
    """Return a paginated list of topics ordered by the selected sort field."""
    cache_svc = CacheService()
    cache_key = build_key("topics", "list", sort_by, str(min_paper_count), str(limit), str(offset))
    cached = await cache_svc.get(cache_key)
    if cached:
        return TopicListResponse(**cached)

    sort_map = {
        "trend_score": desc(Topic.trend_score),
        "momentum": desc(Topic.momentum),
        "velocity": desc(Topic.velocity),
        "paper_count": desc(Topic.paper_count),
        "name": Topic.name,
    }
    order = sort_map.get(sort_by, desc(Topic.trend_score))

    stmt = select(Topic).order_by(order)
    count_stmt = select(func.count()).select_from(Topic)
    if min_paper_count is not None:
        stmt = stmt.where(Topic.paper_count >= min_paper_count)
        count_stmt = count_stmt.where(Topic.paper_count >= min_paper_count)

    total = (await db.execute(count_stmt)).scalar_one()
    result = await db.execute(stmt.offset(offset).limit(limit))
    topics = result.scalars().all()

    resp = TopicListResponse(
        items=[TopicSchema.model_validate(t) for t in topics],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
        has_prev=offset > 0,
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=120)
    return resp


# ---------------------------------------------------------------------------
# GET /topics/graph  (must be before /{topic_id})
# ---------------------------------------------------------------------------


@router.get(
    "/graph",
    response_model=TopicGraphSchema,
    summary="Get topic relationship graph data",
    description=(
        "Returns nodes and edges suitable for a D3.js force-directed layout. "
        "Edges represent co-occurrence of topics across papers/repos. "
        "Node size encodes trend_score; edge weight encodes co-occurrence strength."
    ),
)
async def get_topic_graph(
    limit: int = Query(50, ge=5, le=200, description="Maximum number of topic nodes"),
    min_edge_weight: float = Query(0.1, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
) -> TopicGraphSchema:
    """Return topic graph data (nodes + edges) for D3 visualisation."""
    cache_svc = CacheService()
    cache_key = build_key("topics", "graph", str(limit), str(min_edge_weight))
    cached = await cache_svc.get(cache_key)
    if cached:
        return TopicGraphSchema(**cached)

    # Fetch top topics as nodes
    result = await db.execute(
        select(Topic)
        .where(Topic.trend_score.isnot(None))
        .order_by(desc(Topic.trend_score))
        .limit(limit)
    )
    topics = result.scalars().all()

    nodes = [
        TopicGraphNode(
            id=str(t.id),
            label=t.name,
            trend_score=t.trend_score,
            paper_count=t.paper_count,
            repo_count=t.repo_count,
        )
        for t in topics
    ]

    # Build synthetic edges from keyword overlap (real implementation uses
    # co-occurrence counts from a separate join table)
    edges: List[TopicGraphEdge] = []
    topic_keywords: Dict[str, set] = {}
    for t in topics:
        kws = set(t.keywords or [])
        topic_keywords[str(t.id)] = kws

    topic_ids = list(topic_keywords.keys())
    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            id_a, id_b = topic_ids[i], topic_ids[j]
            kw_a, kw_b = topic_keywords[id_a], topic_keywords[id_b]
            if not kw_a or not kw_b:
                continue
            overlap = len(kw_a & kw_b)
            union = len(kw_a | kw_b)
            weight = overlap / union if union > 0 else 0.0
            if weight >= min_edge_weight:
                edges.append(
                    TopicGraphEdge(
                        source=id_a,
                        target=id_b,
                        weight=round(weight, 4),
                    )
                )

    graph = TopicGraphSchema(
        nodes=nodes,
        edges=edges,
        generated_at=datetime.now(timezone.utc),
    )
    await cache_svc.set(cache_key, graph.model_dump(mode="json"), ttl=300)
    return graph


# ---------------------------------------------------------------------------
# GET /topics/{topic_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{topic_id}",
    response_model=TopicWithRelations,
    summary="Get topic with related papers and repos",
)
async def get_topic(
    topic_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> TopicWithRelations:
    """Return full topic details including top related papers and repositories."""
    cache_svc = CacheService()
    cache_key = build_key("topics", "detail", str(topic_id))
    cached = await cache_svc.get(cache_key)
    if cached:
        return TopicWithRelations(**cached)

    result = await db.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalar_one_or_none()
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")

    # Fetch top related papers (category keyword matching as a proxy)
    top_papers: List[Dict[str, Any]] = []
    if topic.keywords:
        # Simple: find papers whose categories overlap with topic keywords
        for keyword in (topic.keywords or [])[:3]:
            p_result = await db.execute(
                select(Paper)
                .where(Paper.categories.astext.contains(keyword))
                .order_by(desc(Paper.trend_score))
                .limit(5)
            )
            for p in p_result.scalars().all():
                entry = {
                    "id": str(p.id),
                    "title": p.title,
                    "trend_score": p.trend_score,
                    "arxiv_id": p.arxiv_id,
                    "url": p.url,
                }
                if entry not in top_papers:
                    top_papers.append(entry)
                if len(top_papers) >= 5:
                    break
            if len(top_papers) >= 5:
                break

    # Fetch top related repos (topic keyword matching)
    top_repos: List[Dict[str, Any]] = []
    if topic.keywords:
        for keyword in (topic.keywords or [])[:3]:
            r_result = await db.execute(
                select(Repository)
                .where(Repository.topics.astext.contains(keyword))
                .order_by(desc(Repository.stars))
                .limit(5)
            )
            for r in r_result.scalars().all():
                entry = {
                    "id": str(r.id),
                    "full_name": r.full_name,
                    "stars": r.stars,
                    "trend_score": r.trend_score,
                    "url": r.url,
                }
                if entry not in top_repos:
                    top_repos.append(entry)
                if len(top_repos) >= 5:
                    break
            if len(top_repos) >= 5:
                break

    resp = TopicWithRelations(
        **TopicSchema.model_validate(topic).model_dump(),
        top_papers=top_papers[:5],
        top_repos=top_repos[:5],
    )
    await cache_svc.set(cache_key, resp.model_dump(mode="json"), ttl=300)
    return resp


# ---------------------------------------------------------------------------
# GET /topics/{topic_id}/forecast
# ---------------------------------------------------------------------------


@router.get(
    "/{topic_id}/forecast",
    response_model=ForecastSeriesSchema,
    summary="Get topic forecast series",
)
async def get_topic_forecast(
    topic_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ForecastSeriesSchema:
    """Return stored forecast series for a topic."""
    result = await db.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalar_one_or_none()
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")

    fr_result = await db.execute(
        select(ForecastResult)
        .where(ForecastResult.entity_id == topic_id)
        .where(ForecastResult.entity_type == EntityType.TOPIC)
        .order_by(ForecastResult.forecast_date)
    )
    rows = fr_result.scalars().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast available for topic {topic_id}. "
            "Trigger via POST /api/v1/forecasts/compute",
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
        entity_id=topic_id,
        entity_type="topic",
        model_used=rows[0].model_used,
        horizon_days=rows[0].horizon_days or len(rows),
        points=points,
        generated_at=rows[-1].created_at,
    )
