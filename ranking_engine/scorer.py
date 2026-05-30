"""
Modular scoring system for the AI Research Intelligence Platform.

Implements the trend_score formula:
    trend_score = 0.35 * growth_velocity
                + 0.25 * github_activity
                + 0.20 * citation_acceleration
                + 0.10 * community_engagement
                + 0.10 * novelty_score

Each component is normalized to [0, 1] before weighting.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScoreWeights:
    """Configurable weights for the trend-score formula.

    All weights must be non-negative and sum to 1.0 (within floating-point
    tolerance).  Validates on post-init to catch misconfiguration early.
    """

    growth_velocity: float = 0.35
    github_activity: float = 0.25
    citation_acceleration: float = 0.20
    community_engagement: float = 0.10
    novelty_score: float = 0.10

    def __post_init__(self) -> None:
        weights = [
            self.growth_velocity,
            self.github_activity,
            self.citation_acceleration,
            self.community_engagement,
            self.novelty_score,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative.")
        total = sum(weights)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                "Adjust the individual weight values."
            )

    def as_dict(self) -> Dict[str, float]:
        return {
            "growth_velocity": self.growth_velocity,
            "github_activity": self.github_activity,
            "citation_acceleration": self.citation_acceleration,
            "community_engagement": self.community_engagement,
            "novelty_score": self.novelty_score,
        }


@dataclass
class ScoringContext:
    """All input signals required to compute a trend score for one entity."""

    entity_id: str
    entity_type: str  # "paper" | "repo" | "topic"

    # Raw signals
    growth_velocity: Optional[float] = None       # pre-computed if available
    github_activity: Optional[float] = None       # pre-computed if available
    citation_acceleration: Optional[float] = None # pre-computed if available
    community_engagement: Optional[float] = None  # pre-computed if available
    novelty_score: Optional[float] = None         # pre-computed if available

    # GitHub raw metrics (used when github_activity is None)
    github_stars: int = 0
    github_forks: int = 0
    github_watchers: int = 0
    github_stars_today: int = 0

    # Community raw metrics (used when community_engagement is None)
    upvotes: int = 0
    comments: int = 0
    shares: int = 0

    # Citation history as a list of cumulative citation counts (one per period)
    citation_history: List[int] = field(default_factory=list)

    # Timeseries of general signal values + corresponding timestamps (unix)
    signals_history: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Sentence-transformer embedding of the entity's text
    embedding: Optional[NDArray[np.float32]] = None

    # Arbitrary extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendScore:
    """Output of the scoring pipeline for a single entity."""

    entity_id: str
    entity_type: str

    # Component scores (all in [0, 1])
    growth_velocity_score: float = 0.0
    github_activity_score: float = 0.0
    citation_acceleration_score: float = 0.0
    community_engagement_score: float = 0.0
    novelty_score: float = 0.0

    # Weighted aggregate
    final_score: float = 0.0

    # Diagnostics
    weights_used: Dict[str, float] = field(default_factory=dict)
    computed_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "growth_velocity_score": round(self.growth_velocity_score, 6),
            "github_activity_score": round(self.github_activity_score, 6),
            "citation_acceleration_score": round(self.citation_acceleration_score, 6),
            "community_engagement_score": round(self.community_engagement_score, 6),
            "novelty_score": round(self.novelty_score, 6),
            "final_score": round(self.final_score, 6),
            "weights_used": self.weights_used,
            "computed_at": self.computed_at,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------


def min_max_normalize(values: List[float], eps: float = 1e-9) -> List[float]:
    """Scale a list of values to [0, 1] using min-max normalization.

    Args:
        values: Input values.
        eps: Small constant to avoid division by zero when all values equal.

    Returns:
        Normalized list in [0, 1].
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    span = hi - lo
    if span < eps:
        return [0.5] * len(values)
    return ((arr - lo) / span).tolist()


def z_score_normalize(values: List[float], eps: float = 1e-9) -> List[float]:
    """Z-score normalize a list of values.

    Returns the standardized values (mean=0, std=1).  Useful for detecting
    outliers before feeding into downstream models.

    Args:
        values: Input values.
        eps: Small constant to avoid division by zero when std is 0.

    Returns:
        Z-score normalized list.
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    mean, std = arr.mean(), arr.std()
    if std < eps:
        return [0.0] * len(values)
    return ((arr - mean) / std).tolist()


def time_decay_factor(
    age_seconds: float,
    half_life_seconds: float = 86_400.0,  # 1 day default
) -> float:
    """Compute an exponential time-decay weight.

    Older signals receive less weight:  w = exp(-λ * age)
    where λ = ln(2) / half_life.

    Args:
        age_seconds: How old the signal is in seconds.
        half_life_seconds: Time in seconds at which weight halves (default 1 day).

    Returns:
        Decay factor in (0, 1].
    """
    if age_seconds < 0:
        age_seconds = 0.0
    lam = math.log(2) / max(half_life_seconds, 1e-9)
    return math.exp(-lam * age_seconds)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class TrendScorer:
    """Computes weighted trend scores for research entities.

    Example::

        scorer = TrendScorer()
        ctx = ScoringContext(
            entity_id="arxiv:2501.00001",
            entity_type="paper",
            signals_history=[10, 15, 22, 30, 45],
            timestamps=[...],
            github_stars=800,
            github_forks=120,
            citation_history=[0, 2, 5, 9, 16],
            upvotes=320,
            comments=48,
        )
        score = scorer.score(ctx)
        print(score.final_score)
    """

    # Maximum plausible raw values used for clamped normalization
    _GITHUB_STARS_MAX = 100_000
    _GITHUB_FORKS_MAX = 20_000
    _GITHUB_WATCHERS_MAX = 5_000
    _GITHUB_STARS_TODAY_MAX = 500

    def __init__(self, weights: Optional[ScoreWeights] = None) -> None:
        """Initialize the scorer with optional custom weights.

        Args:
            weights: ScoreWeights instance.  Defaults to the standard formula
                     (0.35, 0.25, 0.20, 0.10, 0.10).
        """
        self.weights = weights or ScoreWeights()
        logger.info("TrendScorer initialized with weights: %s", self.weights.as_dict())

    # ------------------------------------------------------------------
    # Component computations
    # ------------------------------------------------------------------

    def compute_growth_velocity(
        self,
        signals: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> float:
        """Estimate growth velocity via linear-regression slope, normalized to [0, 1].

        A positive slope indicates an increasing trend.  The result is clipped
        to [0, 1] so it can be used directly as a component score.

        Args:
            signals: Ordered sequence of signal values (e.g., daily views).
            timestamps: Unix timestamps corresponding to each signal.  If None,
                        evenly-spaced integer indices are assumed.

        Returns:
            Normalized growth velocity in [0, 1].
        """
        if len(signals) < 2:
            logger.debug("Not enough signals for growth_velocity; returning 0.")
            return 0.0

        n = len(signals)
        if timestamps is None or len(timestamps) != n:
            x = np.arange(n, dtype=float)
        else:
            x = np.array(timestamps, dtype=float)
            # Normalize time axis to avoid numerical issues
            x = (x - x.min()) / max(x.max() - x.min(), 1e-9)

        y = np.array(signals, dtype=float)

        # Weighted least-squares: more recent points weighted higher
        decay_weights = np.array(
            [time_decay_factor(float(n - 1 - i) * 3600, half_life_seconds=72 * 3600)
             for i in range(n)],
            dtype=float,
        )

        # Weighted mean
        w_sum = decay_weights.sum()
        x_mean = (decay_weights * x).sum() / w_sum
        y_mean = (decay_weights * y).sum() / w_sum

        numerator = (decay_weights * (x - x_mean) * (y - y_mean)).sum()
        denominator = (decay_weights * (x - x_mean) ** 2).sum()

        if abs(denominator) < 1e-12:
            return 0.0

        slope = numerator / denominator

        # Normalize: map slope through sigmoid to get a [0, 1] score
        # A slope of 0 → 0.5; strongly positive → 1; strongly negative → 0
        # Scale factor controls sensitivity
        scale = max(abs(y_mean), 1.0)
        normalized_slope = slope / scale
        velocity_score = 1.0 / (1.0 + math.exp(-5.0 * normalized_slope))

        return float(np.clip(velocity_score, 0.0, 1.0))

    def compute_github_activity(
        self,
        stars: int = 0,
        forks: int = 0,
        watchers: int = 0,
        stars_today: int = 0,
    ) -> float:
        """Compute a composite GitHub activity score normalized to [0, 1].

        Uses a weighted combination of stars, forks, watchers, and daily stars,
        each individually clamped against known maximums to avoid outlier
        domination.

        Args:
            stars: Total repository stars.
            forks: Total repository forks.
            watchers: Total repository watchers.
            stars_today: Stars gained today (recency signal).

        Returns:
            Composite GitHub activity score in [0, 1].
        """
        s = min(stars, self._GITHUB_STARS_MAX) / self._GITHUB_STARS_MAX
        f = min(forks, self._GITHUB_FORKS_MAX) / self._GITHUB_FORKS_MAX
        w = min(watchers, self._GITHUB_WATCHERS_MAX) / self._GITHUB_WATCHERS_MAX
        st = min(stars_today, self._GITHUB_STARS_TODAY_MAX) / self._GITHUB_STARS_TODAY_MAX

        # Weighted composite: stars_today weighted highest (recency)
        score = 0.35 * s + 0.25 * f + 0.15 * w + 0.25 * st
        return float(np.clip(score, 0.0, 1.0))

    def compute_citation_acceleration(
        self, citation_history: List[int]
    ) -> float:
        """Compute citation acceleration using the second derivative, normalized to [0, 1].

        The second derivative of cumulative citations captures whether citation
        growth is speeding up or slowing down.

        Args:
            citation_history: Cumulative citation counts ordered by time
                              (e.g., [0, 3, 7, 12, 20, 35]).

        Returns:
            Acceleration score in [0, 1].
        """
        if len(citation_history) < 3:
            # Not enough data; fall back to simple growth fraction
            if len(citation_history) == 2:
                delta = citation_history[-1] - citation_history[0]
                base = max(citation_history[0], 1)
                return float(np.clip(delta / base, 0.0, 1.0))
            return 0.0

        arr = np.array(citation_history, dtype=float)
        first_diff = np.diff(arr)          # velocity
        second_diff = np.diff(first_diff)  # acceleration

        if len(second_diff) == 0:
            return 0.0

        mean_accel = second_diff.mean()

        # Normalize via sigmoid (acceleration relative to average velocity)
        avg_velocity = max(first_diff.mean(), 1e-9)
        normalized = mean_accel / avg_velocity
        score = 1.0 / (1.0 + math.exp(-3.0 * normalized))
        return float(np.clip(score, 0.0, 1.0))

    def compute_community_engagement(
        self,
        upvotes: int = 0,
        comments: int = 0,
        shares: int = 0,
    ) -> float:
        """Compute a log-normalized community engagement score in [0, 1].

        Using log normalization mitigates the outsized influence of viral
        outliers while still rewarding highly engaged content.

        Args:
            upvotes: Total upvotes / reactions.
            comments: Total comments.
            shares: Total shares / reposts.

        Returns:
            Engagement score in [0, 1].
        """
        # Log1p to handle zeros
        log_upvotes = math.log1p(max(upvotes, 0))
        log_comments = math.log1p(max(comments, 0))
        log_shares = math.log1p(max(shares, 0))

        # Weighted sum (shares carry the highest social signal)
        raw = 0.40 * log_upvotes + 0.30 * log_comments + 0.30 * log_shares

        # Map into [0, 1] assuming a reasonable upper bound of log1p(10_000) ≈ 9.2
        upper_bound = math.log1p(10_000)
        score = raw / upper_bound
        return float(np.clip(score, 0.0, 1.0))

    def compute_novelty_score(
        self,
        embedding: Optional[NDArray[np.float32]],
        known_embeddings: Optional[List[NDArray[np.float32]]] = None,
    ) -> float:
        """Measure novelty as the maximum cosine distance to known embeddings.

        A high novelty score means the entity is dissimilar to everything
        currently known — useful for surfacing emerging research directions.

        Args:
            embedding: Normalized embedding of the new entity.
            known_embeddings: Corpus of reference embeddings.

        Returns:
            Novelty score in [0, 1]. Returns 1.0 if no known_embeddings exist
            (truly novel) or if embedding is None.
        """
        if embedding is None:
            return 0.5  # Unknown novelty → neutral

        if not known_embeddings:
            return 1.0  # Nothing to compare against → maximally novel

        emb = np.array(embedding, dtype=float)
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-10:
            return 0.0

        emb_unit = emb / emb_norm
        max_distance = 0.0

        for known in known_embeddings:
            k = np.array(known, dtype=float)
            k_norm = np.linalg.norm(k)
            if k_norm < 1e-10:
                continue
            k_unit = k / k_norm
            cosine_sim = float(np.clip(np.dot(emb_unit, k_unit), -1.0, 1.0))
            cosine_dist = 1.0 - cosine_sim  # in [0, 2]
            # Normalize distance to [0, 1]
            max_distance = max(max_distance, cosine_dist / 2.0)

        return float(np.clip(max_distance, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Scoring entry points
    # ------------------------------------------------------------------

    def score(
        self,
        context: ScoringContext,
        known_embeddings: Optional[List[NDArray[np.float32]]] = None,
    ) -> TrendScore:
        """Compute the full trend score for a single entity.

        If a component value is already present in the context it is used
        directly; otherwise it is computed from the raw signals.

        Args:
            context: ScoringContext with entity signals.
            known_embeddings: Optional corpus embeddings for novelty computation.

        Returns:
            TrendScore with all component scores and the weighted final score.
        """
        logger.debug("Scoring entity %s (%s)", context.entity_id, context.entity_type)

        # --- Growth velocity ---
        gv = (
            context.growth_velocity
            if context.growth_velocity is not None
            else self.compute_growth_velocity(
                context.signals_history, context.timestamps
            )
        )

        # --- GitHub activity ---
        gh = (
            context.github_activity
            if context.github_activity is not None
            else self.compute_github_activity(
                stars=context.github_stars,
                forks=context.github_forks,
                watchers=context.github_watchers,
                stars_today=context.github_stars_today,
            )
        )

        # --- Citation acceleration ---
        ca = (
            context.citation_acceleration
            if context.citation_acceleration is not None
            else self.compute_citation_acceleration(context.citation_history)
        )

        # --- Community engagement ---
        ce = (
            context.community_engagement
            if context.community_engagement is not None
            else self.compute_community_engagement(
                upvotes=context.upvotes,
                comments=context.comments,
                shares=context.shares,
            )
        )

        # --- Novelty ---
        ns = (
            context.novelty_score
            if context.novelty_score is not None
            else self.compute_novelty_score(context.embedding, known_embeddings)
        )

        # --- Weighted aggregate ---
        w = self.weights
        final = (
            w.growth_velocity * gv
            + w.github_activity * gh
            + w.citation_acceleration * ca
            + w.community_engagement * ce
            + w.novelty_score * ns
        )

        trend_score = TrendScore(
            entity_id=context.entity_id,
            entity_type=context.entity_type,
            growth_velocity_score=gv,
            github_activity_score=gh,
            citation_acceleration_score=ca,
            community_engagement_score=ce,
            novelty_score=ns,
            final_score=float(np.clip(final, 0.0, 1.0)),
            weights_used=w.as_dict(),
            metadata=context.metadata,
        )

        logger.debug(
            "Entity %s scored: %.4f (gv=%.3f gh=%.3f ca=%.3f ce=%.3f ns=%.3f)",
            context.entity_id,
            trend_score.final_score,
            gv, gh, ca, ce, ns,
        )
        return trend_score

    def batch_score(
        self,
        contexts: List[ScoringContext],
        known_embeddings: Optional[List[NDArray[np.float32]]] = None,
    ) -> List[TrendScore]:
        """Score a list of entities in batch.

        Args:
            contexts: List of ScoringContext objects.
            known_embeddings: Shared reference embeddings for novelty.

        Returns:
            List of TrendScore objects in the same order as ``contexts``.
        """
        logger.info("Batch scoring %d entities.", len(contexts))
        results: List[TrendScore] = []
        for ctx in contexts:
            try:
                results.append(self.score(ctx, known_embeddings=known_embeddings))
            except Exception as exc:
                logger.error(
                    "Failed to score entity %s: %s", ctx.entity_id, exc, exc_info=True
                )
                # Append a zero score so the list length matches input
                results.append(
                    TrendScore(
                        entity_id=ctx.entity_id,
                        entity_type=ctx.entity_type,
                        metadata={"error": str(exc)},
                    )
                )
        return results
