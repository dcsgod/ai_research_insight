"""
Pydantic schemas for research paper request/response validation.

Schemas follow the pattern:
  Base → Create / Update → Response → specialized variants

All response schemas use ``model_config = ConfigDict(from_attributes=True)``
so they can be constructed directly from SQLAlchemy ORM instances.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class PaperBase(BaseModel):
    """Shared fields used by create and response schemas."""

    title: str = Field(..., min_length=1, max_length=1024, description="Full paper title")
    abstract: Optional[str] = Field(None, description="Full abstract text")
    authors: Optional[List[str]] = Field(None, description="List of author names")
    url: Optional[str] = Field(None, max_length=2048, description="Canonical paper URL")
    pdf_url: Optional[str] = Field(None, max_length=2048, description="Direct PDF URL")
    published_date: Optional[datetime] = Field(None, description="UTC publication datetime")
    categories: Optional[List[str]] = Field(
        None, description="Subject categories e.g. ['cs.LG', 'cs.AI']"
    )
    source: str = Field("arxiv", description="arxiv | huggingface | paperswithcode")
    citation_count: int = Field(0, ge=0, description="Number of citations")
    github_url: Optional[str] = Field(None, max_length=2048, description="Implementation repo URL")
    has_implementation: bool = Field(False, description="True if a code implementation exists")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        allowed = {"arxiv", "huggingface", "paperswithcode"}
        if v.lower() not in allowed:
            raise ValueError(f"source must be one of {allowed}")
        return v.lower()


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class PaperCreate(PaperBase):
    """Schema for ingesting a new paper."""

    arxiv_id: Optional[str] = Field(
        None,
        max_length=32,
        description="arXiv identifier (e.g. '2301.00001v2')",
    )
    embedding_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Qdrant point ID (set after embedding is stored)",
    )


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class PaperUpdate(BaseModel):
    """Schema for partial updates to an existing paper (PATCH semantics)."""

    title: Optional[str] = Field(None, min_length=1, max_length=1024)
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    citation_count: Optional[int] = Field(None, ge=0)
    github_url: Optional[str] = Field(None, max_length=2048)
    has_implementation: Optional[bool] = None
    trend_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    embedding_id: Optional[str] = Field(None, max_length=128)
    categories: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class PaperResponse(PaperBase):
    """Full paper representation returned from the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    arxiv_id: Optional[str] = None
    trend_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    embedding_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Paper with pre-computed trend score details
# ---------------------------------------------------------------------------


class PaperWithScore(PaperResponse):
    """Paper response enriched with multi-factor trend score breakdown."""

    growth_velocity: Optional[float] = None
    github_activity: Optional[float] = None
    citation_acceleration: Optional[float] = None
    community_engagement: Optional[float] = None
    novelty_score: Optional[float] = None
    # Cached LLM insight snippet
    insight_snippet: Optional[str] = Field(
        None, max_length=512, description="First 512 chars of the latest LLM insight"
    )


# ---------------------------------------------------------------------------
# List / Pagination
# ---------------------------------------------------------------------------


class PaperListResponse(BaseModel):
    """Paginated list of papers."""

    items: List[PaperResponse]
    total: int = Field(..., ge=0, description="Total matching rows before pagination")
    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(..., ge=1, le=200)
    has_next: bool
    has_prev: bool


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class PaperSearchRequest(BaseModel):
    """Request body for semantic paper search."""

    query: str = Field(..., min_length=1, max_length=2048, description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Pagination offset")
    source_filter: Optional[List[str]] = Field(
        None, description="Filter by source (arxiv, huggingface, paperswithcode)"
    )
    category_filter: Optional[List[str]] = Field(
        None, description="Filter by arxiv categories"
    )
    min_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity / trend score"
    )
    date_from: Optional[datetime] = Field(None, description="Filter papers published after this date")
    date_to: Optional[datetime] = Field(None, description="Filter papers published before this date")
    use_semantic: bool = Field(True, description="Use vector similarity search (vs keyword)")

    @field_validator("source_filter", mode="before")
    @classmethod
    def validate_sources(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        allowed = {"arxiv", "huggingface", "paperswithcode"}
        invalid = [s for s in v if s.lower() not in allowed]
        if invalid:
            raise ValueError(f"Invalid sources: {invalid}. Must be from {allowed}")
        return [s.lower() for s in v]


# ---------------------------------------------------------------------------
# Ingestion trigger
# ---------------------------------------------------------------------------


class PaperIngestRequest(BaseModel):
    """Request body to trigger paper ingestion."""

    sources: List[str] = Field(
        default=["arxiv", "huggingface", "paperswithcode"],
        description="Which sources to ingest from",
    )
    max_results: int = Field(100, ge=1, le=2000)
    categories: Optional[List[str]] = Field(
        None, description="arXiv categories to fetch (overrides settings default)"
    )
    force_refresh: bool = Field(
        False, description="Re-ingest even if paper already exists"
    )


class PaperIngestResponse(BaseModel):
    """Response for a triggered ingestion job."""

    job_id: str
    status: str
    sources: List[str]
    queued_at: datetime
    estimated_duration_seconds: Optional[int] = None
