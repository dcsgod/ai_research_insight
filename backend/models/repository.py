"""
AI Research Intelligence Platform — Repository ORM Model
Represents GitHub repositories and HuggingFace model repos.
"""
import uuid
from sqlalchemy import (
    Column, String, Text, Float, Integer,
    DateTime, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from backend.db.database import Base


class Repository(Base):
    """GitHub/HuggingFace repository model."""
    __tablename__ = "repositories"

    # ─── Primary Key ─────────────────────────────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ─── Identity ────────────────────────────────────────────────────────
    github_id = Column(String(100), nullable=True, unique=False)
    external_id = Column(String(255), nullable=True, index=True)
    source = Column(String(50), nullable=False, default="github", index=True)

    # ─── Content ─────────────────────────────────────────────────────────
    name = Column(String(255), nullable=False)
    full_name = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    homepage = Column(Text, nullable=True)
    readme_summary = Column(Text, nullable=True)

    # ─── Classification ──────────────────────────────────────────────────
    language = Column(String(100), nullable=True, index=True)
    topics = Column(JSONB, nullable=True, default=list)   # ["llm", "rag", ...]
    owner = Column(String(255), nullable=True, index=True)

    # ─── Metrics ─────────────────────────────────────────────────────────
    stars = Column(Integer, nullable=False, default=0, index=True)
    forks = Column(Integer, nullable=False, default=0)
    watchers = Column(Integer, nullable=False, default=0)
    open_issues = Column(Integer, nullable=False, default=0)
    stars_today = Column(Integer, nullable=True, default=0)   # Trending stars/day
    stars_this_week = Column(Integer, nullable=True, default=0)
    contributors_count = Column(Integer, nullable=True, default=0)
    commit_count_30d = Column(Integer, nullable=True, default=0)

    # ─── Ranking ─────────────────────────────────────────────────────────
    trend_score = Column(Float, nullable=True, default=0.0, index=True)
    growth_velocity = Column(Float, nullable=True, default=0.0)
    novelty_score = Column(Float, nullable=True, default=0.0)
    engagement_score = Column(Float, nullable=True, default=0.0)
    github_activity_score = Column(Float, nullable=True, default=0.0)

    # ─── Vector DB ───────────────────────────────────────────────────────
    embedding_id = Column(String(255), nullable=True)
    topic_id = Column(Integer, nullable=True, index=True)

    # ─── Deduplication ───────────────────────────────────────────────────
    content_hash = Column(String(64), nullable=True, index=True)

    # ─── Dates ───────────────────────────────────────────────────────────
    repo_created_at = Column(DateTime(timezone=True), nullable=True)
    repo_updated_at = Column(DateTime(timezone=True), nullable=True)
    last_pushed_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # ─── Timestamps ──────────────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    last_ingested = Column(DateTime(timezone=True), nullable=True)

    # ─── Raw Data ────────────────────────────────────────────────────────
    raw_data = Column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint("external_id", "source", name="uq_repo_external_source"),
        Index("ix_repos_trend_score_desc", trend_score.desc()),
        Index("ix_repos_stars_desc", stars.desc()),
        Index("ix_repos_language_stars", "language", stars.desc()),
        Index("ix_repos_topics_gin", topics, postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Repository id={self.id} full_name='{self.full_name}' stars={self.stars}>"
