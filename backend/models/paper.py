"""
AI Research Intelligence Platform — SQLAlchemy Paper Model
Represents research papers from arXiv, HuggingFace, PapersWithCode.
"""
import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean,
    DateTime, Enum, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from backend.db.database import Base


class PaperSource(str, PyEnum):
    """Source of the paper."""
    ARXIV = "arxiv"
    HUGGINGFACE = "huggingface"
    PAPERSWITHCODE = "paperswithcode"


class Paper(Base):
    """
    Research paper model.
    Aggregates papers from multiple AI research sources.
    """
    __tablename__ = "papers"

    # ─── Primary Key ─────────────────────────────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ─── Source Identity ─────────────────────────────────────────────────
    arxiv_id = Column(String(50), nullable=True, unique=False)
    external_id = Column(String(255), nullable=True, index=True)  # Source-specific ID
    source = Column(
        Enum(PaperSource, name="paper_source_enum"),
        nullable=False,
        default=PaperSource.ARXIV,
        index=True,
    )

    # ─── Content ─────────────────────────────────────────────────────────
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    authors = Column(JSONB, nullable=True, default=list)    # ["Author Name", ...]
    url = Column(Text, nullable=True)
    pdf_url = Column(Text, nullable=True)
    github_url = Column(Text, nullable=True)

    # ─── Classification ──────────────────────────────────────────────────
    categories = Column(JSONB, nullable=True, default=list)   # ["cs.AI", "cs.LG", ...]
    primary_category = Column(String(50), nullable=True, index=True)
    keywords = Column(JSONB, nullable=True, default=list)

    # ─── Dates ───────────────────────────────────────────────────────────
    published_date = Column(DateTime(timezone=True), nullable=True, index=True)
    updated_date = Column(DateTime(timezone=True), nullable=True)

    # ─── Metrics ─────────────────────────────────────────────────────────
    citation_count = Column(Integer, nullable=False, default=0)
    has_implementation = Column(Boolean, nullable=False, default=False)
    implementation_count = Column(Integer, nullable=False, default=0)

    # ─── Ranking ─────────────────────────────────────────────────────────
    trend_score = Column(Float, nullable=True, default=0.0, index=True)
    growth_velocity = Column(Float, nullable=True, default=0.0)
    novelty_score = Column(Float, nullable=True, default=0.0)
    engagement_score = Column(Float, nullable=True, default=0.0)

    # ─── Vector DB ───────────────────────────────────────────────────────
    embedding_id = Column(String(255), nullable=True)   # Qdrant point ID

    # ─── Topic ───────────────────────────────────────────────────────────
    topic_id = Column(Integer, nullable=True, index=True)    # BERTopic topic ID
    topic_probability = Column(Float, nullable=True)

    # ─── AI Metadata ─────────────────────────────────────────────────────
    content_hash = Column(String(64), nullable=True, index=True)  # For deduplication
    raw_data = Column(JSONB, nullable=True)                        # Original API response

    # ─── Timestamps ──────────────────────────────────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    last_ingested = Column(
        DateTime(timezone=True),
        nullable=True,
        default=lambda: datetime.now(timezone.utc),
    )

    # ─── Constraints & Indexes ───────────────────────────────────────────
    __table_args__ = (
        UniqueConstraint("external_id", "source", name="uq_paper_external_source"),
        Index("ix_papers_trend_score_desc", trend_score.desc()),
        Index("ix_papers_published_date_desc", published_date.desc()),
        Index("ix_papers_source_category", "source", "primary_category"),
        Index("ix_papers_categories_gin", categories, postgresql_using="gin"),
        Index("ix_papers_keywords_gin", keywords, postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Paper id={self.id} title='{self.title[:50]}...' source={self.source}>"
