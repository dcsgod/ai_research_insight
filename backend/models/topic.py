"""
SQLAlchemy ORM model for research topics.

Table: topics
A topic clusters papers and repositories around a shared research theme
(e.g. "Diffusion Models", "LLM Reasoning", "RLHF").
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


class Topic(Base):
    """
    A research topic derived from clustering or manual curation.

    Momentum, velocity, and acceleration capture the rate-of-change of
    interest over time:
    - velocity:     First derivative of trend_score (change per day).
    - acceleration: Second derivative (how fast velocity is changing).
    - momentum:     Exponential moving average of velocity.

    Attributes:
        id:           Internal UUID primary key.
        name:         Human-readable topic name (e.g. "Mixture of Experts").
        description:  Short description of the topic.
        keywords:     JSON array of keyword/keyphrase strings used for matching.
        paper_count:  Number of papers currently associated with this topic.
        repo_count:   Number of repositories currently associated with this topic.
        trend_score:  Computed trending score in [0, 1].
        momentum:     EMA of velocity; high positive = sustained growth.
        velocity:     Rate of trend_score change (score/day).
        acceleration: Rate of velocity change (score/day²).
        embedding_id: ID of the topic's centroid vector in Qdrant.
        created_at:   Row creation timestamp (UTC).
        updated_at:   Last row update timestamp (UTC), auto-maintained.
    """

    __tablename__ = "topics"

    # -------------------------------------------------------------------------
    # Primary key
    # -------------------------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------
    name: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        unique=True,
        index=True,
    )
    slug: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        unique=True,
        index=True,
        comment="URL-safe slug derived from name",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    keywords: Mapped[Optional[List[Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="JSON array of keyword strings for topic matching",
    )

    # -------------------------------------------------------------------------
    # Aggregate counts
    # -------------------------------------------------------------------------
    paper_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    repo_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # -------------------------------------------------------------------------
    # Trend dynamics
    # -------------------------------------------------------------------------
    trend_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Normalised trend score in [0, 1]",
    )
    momentum: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Exponential moving average of velocity",
    )
    velocity: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Rate of trend_score change per day",
    )
    acceleration: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Rate of velocity change per day",
    )

    # -------------------------------------------------------------------------
    # Vector DB reference
    # -------------------------------------------------------------------------
    embedding_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Qdrant point ID for the topic centroid embedding",
    )

    # -------------------------------------------------------------------------
    # Audit timestamps
    # -------------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # -------------------------------------------------------------------------
    # Composite indexes
    # -------------------------------------------------------------------------
    __table_args__ = (
        Index("ix_topics_trend_score_desc", trend_score.desc()),
        Index("ix_topics_momentum_desc", momentum.desc()),
        Index(
            "ix_topics_keywords_gin",
            "keywords",
            postgresql_using="gin",
        ),
        {"comment": "Research topic clusters aggregating papers and repositories"},
    )

    # -------------------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<Topic id={self.id!s:.8s} name={self.name!r} "
            f"trend_score={self.trend_score}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation (JSON-serialisable)."""
        return {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "keywords": self.keywords,
            "paper_count": self.paper_count,
            "repo_count": self.repo_count,
            "trend_score": self.trend_score,
            "momentum": self.momentum,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
