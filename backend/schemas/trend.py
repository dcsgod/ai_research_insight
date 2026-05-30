"""
Pydantic schemas for trend signals, scores, topics, forecasts, and dashboard.

Covers:
- TrendSignalSchema, TrendScoreSchema
- TrendingItemSchema (unified ranked result)
- ForecastPointSchema, ForecastSeriesSchema
- TopicSchema, TopicWithRelations
- DashboardSummarySchema
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Trend Signals
# ---------------------------------------------------------------------------


class TrendSignalSchema(BaseModel):
    """A single time-series data point for one entity."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    entity_id: uuid.UUID
    entity_type: str = Field(..., description="paper | repository | topic")
    signal_date: date
    value: float
    signal_type: str = Field(..., description="mentions | stars | citations | engagement")
    source: Optional[str] = None
    created_at: datetime


class SignalSeriesSchema(BaseModel):
    """A time-ordered series of signals for one entity and signal type."""

    entity_id: uuid.UUID
    entity_type: str
    signal_type: str
    data_points: List[TrendSignalSchema]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_points: int = 0


# ---------------------------------------------------------------------------
# Trend Scores
# ---------------------------------------------------------------------------


class TrendScoreSchema(BaseModel):
    """Multi-factor trend score for a single entity."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    entity_id: uuid.UUID
    entity_type: str
    growth_velocity: Optional[float] = Field(None, ge=0.0, le=1.0)
    github_activity: Optional[float] = Field(None, ge=0.0, le=1.0)
    citation_acceleration: Optional[float] = Field(None, ge=0.0, le=1.0)
    community_engagement: Optional[float] = Field(None, ge=0.0, le=1.0)
    novelty_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    final_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    computed_at: datetime


# ---------------------------------------------------------------------------
# Trending Items (unified ranked results)
# ---------------------------------------------------------------------------


class TrendingItemSchema(BaseModel):
    """
    A unified trending item representing a paper, repo, or topic.

    Used for the main trending feed where different entity types are
    mixed and ranked by their final_score.
    """

    entity_id: uuid.UUID
    entity_type: str = Field(..., description="paper | repository | topic")
    rank: int = Field(..., ge=1, description="Position in the current trending list")
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    trend_score: float = Field(..., ge=0.0, le=1.0)
    score_breakdown: Optional[TrendScoreSchema] = None
    # Entity-specific supplementary data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Sparkline: last N days of signal values
    sparkline: Optional[List[float]] = Field(
        None, description="Last 7-day signal values for sparkline chart"
    )
    updated_at: Optional[datetime] = None


class TrendingListResponse(BaseModel):
    """Paginated list of trending items."""

    items: List[TrendingItemSchema]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool
    timeframe: str = Field(..., description="24h | 7d | 30d")
    entity_type: Optional[str] = Field(None, description="Filter applied (if any)")
    generated_at: datetime


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------


class ForecastPointSchema(BaseModel):
    """A single point in a forecast series."""

    forecast_date: date
    predicted_value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class ForecastSeriesSchema(BaseModel):
    """Complete forecast series for one entity."""

    model_config = ConfigDict(from_attributes=True)

    entity_id: uuid.UUID
    entity_type: str
    model_used: str
    horizon_days: int
    points: List[ForecastPointSchema]
    # Historical actuals for context
    historical_points: Optional[List[TrendSignalSchema]] = None
    generated_at: datetime


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------


class TopicSchema(BaseModel):
    """Core topic representation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    paper_count: int = 0
    repo_count: int = 0
    trend_score: Optional[float] = None
    momentum: Optional[float] = None
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class TopicWithRelations(TopicSchema):
    """Topic schema enriched with related papers and repositories."""

    top_papers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top 5 related papers by trend score",
    )
    top_repos: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top 5 related repositories by stars",
    )
    recent_signals: Optional[SignalSeriesSchema] = None
    forecast: Optional[ForecastSeriesSchema] = None
    insight: Optional[str] = None


class TopicListResponse(BaseModel):
    """Paginated list of topics."""

    items: List[TopicSchema]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# ---------------------------------------------------------------------------
# Topic Graph (for D3 visualisation)
# ---------------------------------------------------------------------------


class TopicGraphNode(BaseModel):
    """A node in the topic relationship graph."""

    id: str = Field(..., description="Node ID (topic UUID as string)")
    label: str
    trend_score: Optional[float] = None
    paper_count: int = 0
    repo_count: int = 0
    group: Optional[str] = None  # Cluster group for colouring


class TopicGraphEdge(BaseModel):
    """An edge in the topic relationship graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    weight: float = Field(1.0, ge=0.0, description="Co-occurrence weight")
    relation_type: str = Field("co_occurrence", description="Edge type label")


class TopicGraphSchema(BaseModel):
    """Graph data structure for D3 force-directed layout."""

    nodes: List[TopicGraphNode]
    edges: List[TopicGraphEdge]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Dashboard Summary
# ---------------------------------------------------------------------------


class EntityCountSchema(BaseModel):
    """Row counts per entity type."""

    papers: int = 0
    repositories: int = 0
    topics: int = 0


class DashboardSummarySchema(BaseModel):
    """
    High-level statistics for the main dashboard overview card.
    """

    total_entities: EntityCountSchema
    top_trending_papers: List[TrendingItemSchema]
    top_trending_repos: List[TrendingItemSchema]
    top_trending_topics: List[TopicSchema]
    # System health
    last_ingestion_at: Optional[datetime] = None
    next_ingestion_at: Optional[datetime] = None
    papers_ingested_today: int = 0
    repos_ingested_today: int = 0
    # Rate-of-change stats
    average_trend_velocity: Optional[float] = None
    hottest_topic: Optional[TopicSchema] = None
    generated_at: datetime
