"""
SQLAlchemy ORM models for trend signals and computed trend scores.

Tables:
  - trend_signals  : Time-series data points per entity
  - trend_scores   : Computed multi-factor scores per entity
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Float,
    Index,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.database import Base


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EntityType(str, enum.Enum):
    """The kind of entity a signal or score belongs to."""

    PAPER = "paper"
    REPOSITORY = "repository"
    TOPIC = "topic"


class SignalType(str, enum.Enum):
    """
    The dimension of activity captured by a TrendSignal row.

    - MENTIONS:    Social/forum mention count (Reddit, HN, Twitter)
    - STARS:       GitHub stars gained in the period
    - CITATIONS:   New citation count in the period
    - ENGAGEMENT:  Composite engagement metric (downloads, clones, etc.)
    """

    MENTIONS = "mentions"
    STARS = "stars"
    CITATIONS = "citations"
    ENGAGEMENT = "engagement"


# ---------------------------------------------------------------------------
# TrendSignal
# ---------------------------------------------------------------------------


class TrendSignal(Base):
    """
    A single time-series data point for one entity on one signal dimension.

    One row per (entity_id, entity_type, signal_type, signal_date).
    Queries aggregate over signal_date to build sparklines and compute
    growth rates.

    Attributes:
        id:           UUID primary key.
        entity_id:    UUID of the related Paper, Repository, or Topic row.
        entity_type:  Discriminator (paper | repository | topic).
        signal_date:  Calendar date the signal was recorded.
        value:        Measured value for that day (e.g. star count, mention count).
        signal_type:  Dimension being measured (mentions | stars | citations | engagement).
        source:       Optional free-text label for the data source.
        created_at:   Row creation timestamp.
    """

    __tablename__ = "trend_signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="UUID of the related entity (Paper / Repository / Topic)",
    )
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType, name="entity_type_enum", create_constraint=True),
        nullable=False,
        index=True,
    )
    signal_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )
    value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    signal_type: Mapped[SignalType] = mapped_column(
        Enum(SignalType, name="signal_type_enum", create_constraint=True),
        nullable=False,
        index=True,
    )
    source: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Data source label (e.g. 'github_trending', 'reddit')",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_signals_entity_type_date", "entity_id", "entity_type", "signal_date"),
        Index("ix_signals_date_type_value", "signal_date", "signal_type", "value"),
        Index("ix_signals_entity_signal_date", "entity_id", "signal_type", "signal_date"),
        {
            "comment": "Time-series signal values per entity for trend computation",
            "postgresql_partition_by": "RANGE (signal_date)",
        },
    )

    def __repr__(self) -> str:
        return (
            f"<TrendSignal entity_id={self.entity_id!s:.8s} "
            f"type={self.signal_type.value} date={self.signal_date} value={self.value}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "entity_id": str(self.entity_id),
            "entity_type": self.entity_type.value,
            "signal_date": self.signal_date.isoformat(),
            "value": self.value,
            "signal_type": self.signal_type.value,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# TrendScore
# ---------------------------------------------------------------------------


class TrendScore(Base):
    """
    Computed multi-factor trend score for a single entity.

    One row per entity; updated each time the scoring pipeline runs.
    The final_score is a weighted combination of the factor scores.

    Attributes:
        id:                   UUID primary key.
        entity_id:            UUID of the scored entity.
        entity_type:          Discriminator (paper | repository | topic).
        growth_velocity:      Normalised rate of recent signal growth.
        github_activity:      Normalised GitHub engagement sub-score.
        citation_acceleration: Normalised citation acceleration sub-score.
        community_engagement: Normalised social mention sub-score.
        novelty_score:        Recency-adjusted originality sub-score.
        final_score:          Weighted aggregate of all sub-scores in [0, 1].
        computed_at:          When this score was last computed.
    """

    __tablename__ = "trend_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True,
        comment="UUID of the scored entity",
    )
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType, name="entity_type_enum", create_constraint=False),
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Factor sub-scores (all normalised to [0, 1])
    # -------------------------------------------------------------------------
    growth_velocity: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Normalised first derivative of primary signal",
    )
    github_activity: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Normalised GitHub stars/forks/commits activity",
    )
    citation_acceleration: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Normalised acceleration in citation count",
    )
    community_engagement: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Normalised social/forum engagement score",
    )
    novelty_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Recency-adjusted originality (decays for older entities)",
    )
    final_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Weighted final trend score in [0, 1]",
    )

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_trend_scores_final_desc", final_score.desc()),
        Index("ix_trend_scores_entity_type", "entity_type", final_score.desc()),
        {"comment": "Computed multi-factor trend scores per entity"},
    )

    def __repr__(self) -> str:
        return (
            f"<TrendScore entity_id={self.entity_id!s:.8s} "
            f"final={self.final_score} computed={self.computed_at}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "entity_id": str(self.entity_id),
            "entity_type": self.entity_type.value,
            "growth_velocity": self.growth_velocity,
            "github_activity": self.github_activity,
            "citation_acceleration": self.citation_acceleration,
            "community_engagement": self.community_engagement,
            "novelty_score": self.novelty_score,
            "final_score": self.final_score,
            "computed_at": self.computed_at.isoformat(),
        }
