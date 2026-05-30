"""
Daily signal aggregation and momentum analysis for the AI Research Intelligence Platform.

Provides:
  - SignalAggregator: Aggregates raw trend signals into a daily DataFrame.
  - Rolling statistics (mean, std) over configurable windows.
  - Z-score based anomaly detection for trend breakouts.
  - MomentumMetrics: velocity, acceleration, momentum score.
  - Signal normalization to [0, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class TrendSignal:
    """A single raw trend signal observation.

    Attributes:
        entity_id: The entity this signal belongs to.
        signal_type: Category of the signal (e.g. "stars", "citations", "views").
        value: Numeric value of the signal.
        timestamp: When the signal was observed.
        source: Data source identifier (e.g. "github", "arxiv", "reddit").
        metadata: Arbitrary extra fields.
    """

    entity_id: str
    signal_type: str
    value: float
    timestamp: datetime
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MomentumMetrics:
    """Computed momentum summary for a single entity/signal pair.

    Attributes:
        entity_id: Entity identifier.
        signal_type: Signal being measured.
        velocity: Rate of change per day (1st derivative, rolling 7-day).
        acceleration: Change in velocity (2nd derivative, rolling 14-day).
        momentum_score: Composite score in [0, 1] (combines velocity + accel).
        is_breakout: Whether a trend breakout was detected.
        window_used_days: The rolling window used for momentum computation.
    """

    entity_id: str
    signal_type: str
    velocity: float = 0.0
    acceleration: float = 0.0
    momentum_score: float = 0.0
    is_breakout: bool = False
    window_used_days: int = 7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "signal_type": self.signal_type,
            "velocity": round(self.velocity, 6),
            "acceleration": round(self.acceleration, 6),
            "momentum_score": round(self.momentum_score, 6),
            "is_breakout": self.is_breakout,
            "window_used_days": self.window_used_days,
        }


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------


class SignalAggregator:
    """Aggregates, enriches, and analyzes raw trend signals.

    All public methods operate on either a list of TrendSignal objects or a
    pre-built Pandas DataFrame (with columns: date, entity_id, signal_type,
    value).  The design allows the aggregator to be used independently of the
    rest of the pipeline for ad-hoc analysis.

    Args:
        breakout_z_threshold: Z-score above which a point is flagged as a
            trend breakout (default 2.5).
        default_rolling_windows: Rolling windows (in days) to compute
            statistics over.
    """

    def __init__(
        self,
        breakout_z_threshold: float = 2.5,
        default_rolling_windows: Optional[List[int]] = None,
    ) -> None:
        self._z_threshold = breakout_z_threshold
        self._windows: List[int] = default_rolling_windows or [7, 14, 30]
        logger.info(
            "SignalAggregator initialized (z_threshold=%.2f, windows=%s)",
            breakout_z_threshold,
            self._windows,
        )

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def aggregate_daily(
        self,
        signals: List[TrendSignal],
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """Aggregate a list of TrendSignal objects into a daily time series.

        Each row in the output DataFrame represents one (entity_id,
        signal_type, date) triple with the aggregated value.

        Args:
            signals: List of TrendSignal observations (may span multiple days).
            agg_func: Aggregation function to apply within each day ("sum",
                      "mean", "max", or "last").

        Returns:
            DataFrame with columns: date, entity_id, signal_type, value.
            Rows are sorted by entity_id, signal_type, date.

        Raises:
            ValueError: If ``agg_func`` is not supported.
        """
        if not signals:
            logger.warning("aggregate_daily received empty signals list.")
            return pd.DataFrame(columns=["date", "entity_id", "signal_type", "value"])

        _supported = {"sum", "mean", "max", "last"}
        if agg_func not in _supported:
            raise ValueError(
                f"agg_func must be one of {_supported}, got '{agg_func}'."
            )

        records = [
            {
                "entity_id": s.entity_id,
                "signal_type": s.signal_type,
                "value": s.value,
                "date": pd.Timestamp(s.timestamp).normalize(),  # day-level
            }
            for s in signals
        ]

        raw_df = pd.DataFrame(records)
        group_keys = ["entity_id", "signal_type", "date"]

        if agg_func == "last":
            # Keep the last value per group (by timestamp order)
            raw_df_sorted = raw_df.sort_values("date")
            daily = (
                raw_df_sorted.groupby(group_keys, sort=False)["value"]
                .last()
                .reset_index()
            )
        else:
            agg_map = {"sum": "sum", "mean": "mean", "max": "max"}
            daily = (
                raw_df.groupby(group_keys)["value"]
                .agg(agg_map[agg_func])
                .reset_index()
            )

        daily = daily.sort_values(["entity_id", "signal_type", "date"]).reset_index(drop=True)
        logger.debug(
            "aggregate_daily: %d signals → %d rows (agg=%s).",
            len(signals), len(daily), agg_func,
        )
        return daily

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def compute_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        group_keys: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Add rolling mean and std columns for each window.

        Rolling statistics are computed within each (entity_id, signal_type)
        group so that different entities don't bleed into each other.

        Args:
            df: Daily aggregated DataFrame with at least ["date", "value"]
                and optionally ["entity_id", "signal_type"].
            windows: Rolling window sizes in days.  Defaults to
                     ``self._windows``.
            group_keys: Columns to group by before rolling.  Defaults to
                        ["entity_id", "signal_type"] if present.

        Returns:
            DataFrame with additional columns:
              - ``roll_mean_{w}d`` and ``roll_std_{w}d`` for each window w.
        """
        windows = windows or self._windows
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date")

        # Determine grouping
        auto_keys = [k for k in ["entity_id", "signal_type"] if k in result.columns]
        gkeys = group_keys or auto_keys

        if gkeys:
            grouped = result.groupby(gkeys)["value"]
            for win in windows:
                roll = grouped.transform(
                    lambda s, w=win: s.rolling(w, min_periods=1).mean()
                )
                roll_std = grouped.transform(
                    lambda s, w=win: s.rolling(w, min_periods=1).std().fillna(0)
                )
                result[f"roll_mean_{win}d"] = roll
                result[f"roll_std_{win}d"] = roll_std
        else:
            for win in windows:
                result[f"roll_mean_{win}d"] = (
                    result["value"].rolling(win, min_periods=1).mean()
                )
                result[f"roll_std_{win}d"] = (
                    result["value"].rolling(win, min_periods=1).std().fillna(0)
                )

        logger.debug("compute_rolling_stats: added stats for windows %s.", windows)
        return result

    # ------------------------------------------------------------------
    # Breakout detection
    # ------------------------------------------------------------------

    def detect_trend_breakout(
        self,
        df: pd.DataFrame,
        window: int = 30,
        column: str = "value",
    ) -> bool:
        """Detect whether the most recent value is a statistically significant breakout.

        Uses a Z-score computed over a rolling window of ``window`` days.
        A breakout is detected when the latest Z-score exceeds
        ``self._z_threshold``.

        Args:
            df: DataFrame with a ``column`` to analyse, sorted by date.
            window: Look-back window in rows.
            column: Column to run the Z-score test on.

        Returns:
            True if the latest point is a breakout, False otherwise.
        """
        if column not in df.columns or len(df) < 3:
            return False

        values = df[column].values.astype(float)
        lookback = values[-window:] if len(values) >= window else values
        mean = lookback.mean()
        std = lookback.std()

        if std < 1e-9:
            return False

        z_score = (values[-1] - mean) / std
        is_breakout = abs(z_score) >= self._z_threshold
        logger.debug(
            "detect_trend_breakout: latest=%f mean=%f std=%f z=%.3f breakout=%s",
            values[-1], mean, std, z_score, is_breakout,
        )
        return bool(is_breakout)

    # ------------------------------------------------------------------
    # Momentum metrics
    # ------------------------------------------------------------------

    def compute_momentum_metrics(
        self,
        df: pd.DataFrame,
        entity_id: str = "unknown",
        signal_type: str = "general",
        velocity_window: int = 7,
    ) -> MomentumMetrics:
        """Compute velocity, acceleration, and a composite momentum score.

        Velocity = slope of a linear regression fit to the last
        ``velocity_window`` values.

        Acceleration = difference in velocity between the last and previous
        ``velocity_window`` windows.

        Momentum score = sigmoid(0.5 * velocity + 0.5 * acceleration),
        mapping the composite to [0, 1].

        Args:
            df: Daily DataFrame with a "value" column, sorted chronologically.
            entity_id: Entity label for the output.
            signal_type: Signal label for the output.
            velocity_window: Rolling window for velocity computation.

        Returns:
            MomentumMetrics dataclass.
        """
        if "value" not in df.columns or len(df) < 2:
            return MomentumMetrics(entity_id=entity_id, signal_type=signal_type)

        values = df["value"].values.astype(float)

        def _slope(arr: np.ndarray) -> float:
            """Fit a line to arr and return the slope."""
            if len(arr) < 2:
                return 0.0
            x = np.arange(len(arr), dtype=float)
            x_mean, y_mean = x.mean(), arr.mean()
            denom = ((x - x_mean) ** 2).sum()
            if abs(denom) < 1e-12:
                return 0.0
            return float(((x - x_mean) * (arr - y_mean)).sum() / denom)

        # Velocity over last velocity_window points
        recent = values[-velocity_window:] if len(values) >= velocity_window else values
        velocity = _slope(recent)

        # Acceleration = change in velocity (use two non-overlapping windows)
        if len(values) >= 2 * velocity_window:
            prev_window = values[-(2 * velocity_window):-velocity_window]
            prev_velocity = _slope(prev_window)
        else:
            prev_velocity = 0.0
        acceleration = velocity - prev_velocity

        # Composite momentum score via sigmoid
        combined = 0.5 * velocity + 0.5 * acceleration
        # Scale relative to average signal magnitude
        avg_magnitude = max(abs(values).mean(), 1e-9)
        normalized = combined / avg_magnitude
        momentum_score = float(1.0 / (1.0 + np.exp(-5.0 * normalized)))

        is_breakout = self.detect_trend_breakout(df, window=30)

        metrics = MomentumMetrics(
            entity_id=entity_id,
            signal_type=signal_type,
            velocity=velocity,
            acceleration=acceleration,
            momentum_score=momentum_score,
            is_breakout=is_breakout,
            window_used_days=velocity_window,
        )
        logger.debug(
            "Momentum for %s/%s: vel=%.4f accel=%.4f score=%.4f breakout=%s",
            entity_id, signal_type, velocity, acceleration, momentum_score, is_breakout,
        )
        return metrics

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_signals(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "min-max",
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        """Normalize specified columns to [0, 1].

        Args:
            df: Input DataFrame.
            columns: Columns to normalize.  Defaults to ["value"] if present.
            method: "min-max" or "z-score".  Z-score is clipped to [0, 1]
                    after standardizing.
            eps: Small constant to prevent division by zero.

        Returns:
            Copy of ``df`` with the specified columns normalized in-place.

        Raises:
            ValueError: If ``method`` is not "min-max" or "z-score".
        """
        if method not in {"min-max", "z-score"}:
            raise ValueError(f"method must be 'min-max' or 'z-score', got '{method}'.")

        columns = columns or (["value"] if "value" in df.columns else [])
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                logger.warning("normalize_signals: column '%s' not found.", col)
                continue
            values = result[col].values.astype(float)
            if method == "min-max":
                lo, hi = values.min(), values.max()
                span = hi - lo
                if span < eps:
                    result[col] = 0.5
                else:
                    result[col] = (values - lo) / span
            else:  # z-score
                mean, std = values.mean(), values.std()
                if std < eps:
                    result[col] = 0.0
                else:
                    z = (values - mean) / std
                    # Clip to [-3, 3] then rescale to [0, 1]
                    z_clipped = np.clip(z, -3, 3)
                    result[col] = (z_clipped + 3) / 6.0

        logger.debug("normalize_signals: normalized %d columns using %s.", len(columns), method)
        return result

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_all_momentum(
        self,
        df: pd.DataFrame,
        velocity_window: int = 7,
    ) -> List[MomentumMetrics]:
        """Compute momentum metrics for every (entity_id, signal_type) group.

        Args:
            df: Aggregated daily DataFrame with entity_id, signal_type, date,
                and value columns.
            velocity_window: Window size in days for velocity computation.

        Returns:
            List of MomentumMetrics, one per group.
        """
        if not {"entity_id", "signal_type", "value"}.issubset(df.columns):
            logger.error(
                "compute_all_momentum requires entity_id, signal_type, and value columns."
            )
            return []

        results: List[MomentumMetrics] = []
        groups = df.groupby(["entity_id", "signal_type"])
        for (eid, stype), group in groups:
            group_sorted = group.sort_values("date").reset_index(drop=True)
            metrics = self.compute_momentum_metrics(
                group_sorted,
                entity_id=str(eid),
                signal_type=str(stype),
                velocity_window=velocity_window,
            )
            results.append(metrics)

        logger.info(
            "compute_all_momentum: computed metrics for %d entity/signal pairs.", len(results)
        )
        return results
