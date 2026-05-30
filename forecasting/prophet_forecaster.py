"""
Prophet-based forecasting for trend momentum prediction.

Provides:
  - ProphetForecaster: Train and forecast time series using Facebook Prophet.
  - ForecastResult: Structured output with dates, yhat, bounds, trend, momentum.
  - Cold-start fallback to linear regression when < 14 data points are available.
  - Model persistence to disk via pickle.
  - Changepoint detection enabled; holiday effects disabled for AI trends.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Prophet is an optional heavy dependency; import lazily so the rest of the
# platform can function even if Prophet isn't installed.
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning(
        "Prophet is not installed. ProphetForecaster will fall back to linear "
        "regression for all forecasts. Install with: pip install prophet"
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Complete forecast output for a single entity and signal type.

    Attributes:
        entity_id: Identifier of the entity being forecast.
        signal_type: Name of the signal (e.g. "stars", "citations", "views").
        dates: Forecast dates as a list of datetime objects.
        yhat: Point forecast values.
        yhat_lower: Lower bound of the 80 % prediction interval.
        yhat_upper: Upper bound of the 80 % prediction interval.
        trend: Underlying trend component extracted by Prophet.
        momentum: Slope of yhat over the first 7 forecast days (scalar).
        velocity: Rate of change of the signal at the end of the training data.
        acceleration: Change in velocity (second derivative at training end).
        model_type: "prophet" or "linear" (cold-start fallback).
        trained_on_points: Number of historical data points used for training.
        metadata: Arbitrary extra info.
    """

    entity_id: str
    signal_type: str
    dates: List[datetime]
    yhat: List[float]
    yhat_lower: List[float]
    yhat_upper: List[float]
    trend: List[float]
    momentum: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    model_type: str = "prophet"
    trained_on_points: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "signal_type": self.signal_type,
            "dates": [d.isoformat() for d in self.dates],
            "yhat": [round(v, 4) for v in self.yhat],
            "yhat_lower": [round(v, 4) for v in self.yhat_lower],
            "yhat_upper": [round(v, 4) for v in self.yhat_upper],
            "trend": [round(v, 4) for v in self.trend],
            "momentum": round(self.momentum, 6),
            "velocity": round(self.velocity, 6),
            "acceleration": round(self.acceleration, 6),
            "model_type": self.model_type,
            "trained_on_points": self.trained_on_points,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helper math utilities
# ---------------------------------------------------------------------------


def _linear_regression(x: NDArray, y: NDArray) -> Tuple[float, float]:
    """Fit a simple OLS line and return (slope, intercept)."""
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n == 1 else 0.0
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if abs(denom) < 1e-12:
        return 0.0, y_mean
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def compute_velocity(signals: List[float]) -> float:
    """Compute rate of change (first derivative at the tail of the series).

    Uses the average of the last 3 first-differences to reduce noise.

    Args:
        signals: Ordered list of signal values.

    Returns:
        Velocity scalar.
    """
    if len(signals) < 2:
        return 0.0
    arr = np.array(signals, dtype=float)
    diffs = np.diff(arr)
    return float(diffs[-min(3, len(diffs)):].mean())


def compute_acceleration(signals: List[float]) -> float:
    """Compute rate of change of velocity (second derivative at tail).

    Args:
        signals: Ordered list of signal values.

    Returns:
        Acceleration scalar.
    """
    if len(signals) < 3:
        return 0.0
    arr = np.array(signals, dtype=float)
    first_diff = np.diff(arr)
    second_diff = np.diff(first_diff)
    if len(second_diff) == 0:
        return 0.0
    return float(second_diff[-min(3, len(second_diff)):].mean())


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------


class ProphetForecaster:
    """Trains and runs Prophet forecasts for individual entities.

    Handles the cold-start case (< 14 data points) by falling back to a
    simple linear regression projection.  All trained models are cached on
    disk so they can be loaded without re-training.

    Args:
        model_cache_dir: Directory to store serialized models.
        changepoint_prior_scale: Controls changepoint flexibility (higher =
            more sensitive to trend changes).
        interval_width: Prediction interval width (default 0.80).
        min_points_for_prophet: Minimum historical data points required to
            use Prophet (below this, linear regression is used).
    """

    COLD_START_THRESHOLD: int = 14

    def __init__(
        self,
        model_cache_dir: str = "/tmp/prophet_cache",
        changepoint_prior_scale: float = 0.05,
        interval_width: float = 0.80,
        min_points_for_prophet: int = 14,
    ) -> None:
        self._cache_dir = Path(model_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._changepoint_prior_scale = changepoint_prior_scale
        self._interval_width = interval_width
        self._min_points = min_points_for_prophet

        # In-memory model registry: (entity_id, signal_type) → model
        self._models: Dict[Tuple[str, str], Any] = {}
        # Training data registry for incremental updates
        self._training_data: Dict[Tuple[str, str], pd.DataFrame] = {}

        logger.info(
            "ProphetForecaster initialized (cache=%s, cps=%.3f)",
            self._cache_dir,
            changepoint_prior_scale,
        )

    # ------------------------------------------------------------------
    # Model cache
    # ------------------------------------------------------------------

    def _cache_path(self, entity_id: str, signal_type: str) -> Path:
        safe_id = entity_id.replace("/", "_").replace(":", "_")
        return self._cache_dir / f"{safe_id}__{signal_type}.pkl"

    def _save_model(self, entity_id: str, signal_type: str, model: Any) -> None:
        path = self._cache_path(entity_id, signal_type)
        try:
            with open(path, "wb") as f:
                pickle.dump(model, f)
            logger.debug("Model saved to %s", path)
        except Exception as exc:
            logger.warning("Failed to save model %s: %s", path, exc)

    def _load_model(self, entity_id: str, signal_type: str) -> Optional[Any]:
        key = (entity_id, signal_type)
        if key in self._models:
            return self._models[key]
        path = self._cache_path(entity_id, signal_type)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                self._models[key] = model
                logger.debug("Model loaded from %s", path)
                return model
            except Exception as exc:
                logger.warning("Failed to load model %s: %s", path, exc)
        return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_prophet_model(self) -> "Prophet":
        """Construct a configured Prophet model."""
        model = Prophet(
            changepoint_prior_scale=self._changepoint_prior_scale,
            interval_width=self._interval_width,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            # Holiday effects disabled: AI research trends don't follow
            # conventional holiday patterns
        )
        return model

    def train(
        self,
        entity_id: str,
        signal_type: str,
        historical_data: pd.DataFrame,
    ) -> None:
        """Train (or re-train) the forecasting model for an entity/signal pair.

        The ``historical_data`` DataFrame must have at least two columns:
          - ``ds``: datetime-like column (dates)
          - ``y``: numeric values to forecast

        If fewer than ``min_points_for_prophet`` rows are provided, the model
        is stored as a dict with regression coefficients (cold-start mode).

        Args:
            entity_id: Unique entity identifier.
            signal_type: Name of the signal (e.g. "stars", "citations").
            historical_data: DataFrame with 'ds' and 'y' columns.
        """
        if not {"ds", "y"}.issubset(historical_data.columns):
            raise ValueError("historical_data must have 'ds' and 'y' columns.")

        df = historical_data[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna().sort_values("ds").reset_index(drop=True)

        key = (entity_id, signal_type)
        self._training_data[key] = df

        n = len(df)
        logger.info(
            "Training %s/%s with %d data points.", entity_id, signal_type, n
        )

        if n < self.COLD_START_THRESHOLD or not PROPHET_AVAILABLE:
            # Cold-start: fit a simple linear regression
            x = np.arange(n, dtype=float)
            y = df["y"].values.astype(float)
            slope, intercept = _linear_regression(x, y)
            model = {
                "type": "linear",
                "slope": slope,
                "intercept": intercept,
                "last_x": float(n - 1),
                "last_ds": df["ds"].iloc[-1],
                "n_points": n,
            }
            logger.info(
                "Cold-start linear model for %s/%s: slope=%.4f", entity_id, signal_type, slope
            )
        else:
            model = self._build_prophet_model()
            model.fit(df)

        self._models[key] = model
        self._save_model(entity_id, signal_type, model)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        entity_id: str,
        signal_type: str = "general",
        periods: int = 30,
        freq: str = "D",
    ) -> ForecastResult:
        """Generate a forecast for a trained entity/signal pair.

        Args:
            entity_id: Entity to forecast.
            signal_type: Signal name (must match a previously trained key).
            periods: Number of future periods to forecast.
            freq: Frequency string ('D' = daily, 'W' = weekly, etc.).

        Returns:
            ForecastResult with dates, predictions, and momentum metrics.

        Raises:
            RuntimeError: If no model has been trained for this entity/signal.
        """
        key = (entity_id, signal_type)
        model = self._models.get(key) or self._load_model(entity_id, signal_type)
        if model is None:
            raise RuntimeError(
                f"No model found for {entity_id}/{signal_type}. Call train() first."
            )

        if isinstance(model, dict) and model.get("type") == "linear":
            return self._linear_forecast(entity_id, signal_type, model, periods, freq)
        else:
            return self._prophet_forecast(entity_id, signal_type, model, periods, freq)

    def _linear_forecast(
        self,
        entity_id: str,
        signal_type: str,
        model: Dict[str, Any],
        periods: int,
        freq: str,
    ) -> ForecastResult:
        """Produce a linear-regression forecast for cold-start entities."""
        last_x = model["last_x"]
        slope = model["slope"]
        intercept = model["intercept"]
        last_ds: datetime = pd.to_datetime(model["last_ds"]).to_pydatetime()

        if freq == "D":
            delta = timedelta(days=1)
        elif freq == "W":
            delta = timedelta(weeks=1)
        else:
            delta = timedelta(days=1)

        dates: List[datetime] = []
        yhat: List[float] = []
        # Simple uncertainty: ± 20 % of the predicted value
        for i in range(1, periods + 1):
            x = last_x + i
            y = max(slope * x + intercept, 0.0)
            dates.append(last_ds + delta * i)
            yhat.append(y)

        uncertainty = [0.20 * abs(v) for v in yhat]
        yhat_lower = [max(y - u, 0.0) for y, u in zip(yhat, uncertainty)]
        yhat_upper = [y + u for y, u in zip(yhat, uncertainty)]

        # Training signal for momentum computation
        training_df = self._training_data.get((entity_id, signal_type))
        signals = training_df["y"].tolist() if training_df is not None else []

        momentum = self.compute_momentum(yhat)
        vel = compute_velocity(signals)
        accel = compute_acceleration(signals)

        return ForecastResult(
            entity_id=entity_id,
            signal_type=signal_type,
            dates=dates,
            yhat=yhat,
            yhat_lower=yhat_lower,
            yhat_upper=yhat_upper,
            trend=yhat,  # trend == yhat for linear model
            momentum=momentum,
            velocity=vel,
            acceleration=accel,
            model_type="linear",
            trained_on_points=model.get("n_points", 0),
        )

    def _prophet_forecast(
        self,
        entity_id: str,
        signal_type: str,
        model: "Prophet",
        periods: int,
        freq: str,
    ) -> ForecastResult:
        """Produce a Prophet forecast."""
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast_df = model.predict(future)

        # Extract only the future rows
        fut = forecast_df.tail(periods).reset_index(drop=True)

        dates = pd.to_datetime(fut["ds"]).dt.to_pydatetime().tolist()
        yhat = fut["yhat"].clip(lower=0).tolist()
        yhat_lower = fut["yhat_lower"].clip(lower=0).tolist()
        yhat_upper = fut["yhat_upper"].tolist()
        trend_col = fut["trend"].tolist() if "trend" in fut.columns else yhat

        training_df = self._training_data.get((entity_id, signal_type))
        signals = training_df["y"].tolist() if training_df is not None else []
        n_points = len(signals)

        momentum = self.compute_momentum(yhat)
        vel = compute_velocity(signals)
        accel = compute_acceleration(signals)

        return ForecastResult(
            entity_id=entity_id,
            signal_type=signal_type,
            dates=dates,
            yhat=yhat,
            yhat_lower=yhat_lower,
            yhat_upper=yhat_upper,
            trend=trend_col,
            momentum=momentum,
            velocity=vel,
            acceleration=accel,
            model_type="prophet",
            trained_on_points=n_points,
        )

    # ------------------------------------------------------------------
    # Momentum / velocity / acceleration
    # ------------------------------------------------------------------

    @staticmethod
    def compute_momentum(forecast_yhat: List[float], window: int = 7) -> float:
        """Compute the average slope of yhat over the first ``window`` forecast days.

        A positive momentum means the signal is expected to grow; negative
        means decline.

        Args:
            forecast_yhat: Point forecast values (chronological order).
            window: Number of days to consider for momentum (default 7).

        Returns:
            Momentum scalar (slope units / day).
        """
        yhat = forecast_yhat[:window]
        if len(yhat) < 2:
            return 0.0
        arr = np.array(yhat, dtype=float)
        x = np.arange(len(arr), dtype=float)
        slope, _ = _linear_regression(x, arr)
        return float(slope)

    @staticmethod
    def compute_velocity(signals: List[float]) -> float:
        """Rate of change at the tail of the historical signal.

        Args:
            signals: Historical signal values (chronological).

        Returns:
            Velocity scalar.
        """
        return compute_velocity(signals)

    @staticmethod
    def compute_acceleration(signals: List[float]) -> float:
        """Second derivative of the historical signal.

        Args:
            signals: Historical signal values (chronological).

        Returns:
            Acceleration scalar.
        """
        return compute_acceleration(signals)

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def batch_forecast(
        self,
        entities: List[Dict[str, Any]],
        periods: int = 30,
        freq: str = "D",
    ) -> List[ForecastResult]:
        """Forecast multiple entities in a single call.

        Each element of ``entities`` should be a dict with at minimum:
          - ``entity_id`` (str)
          - ``signal_type`` (str)
          - ``historical_data`` (pd.DataFrame with ds, y columns)

        Models are trained if not already cached.

        Args:
            entities: List of entity specification dicts.
            periods: Forecast horizon in periods.
            freq: Frequency string.

        Returns:
            List of ForecastResult objects (same order as input).
        """
        results: List[ForecastResult] = []
        for spec in entities:
            eid = spec["entity_id"]
            stype = spec.get("signal_type", "general")
            hist = spec.get("historical_data")
            try:
                if hist is not None and not hist.empty:
                    self.train(eid, stype, hist)
                result = self.forecast(eid, stype, periods=periods, freq=freq)
                results.append(result)
                logger.info(
                    "Batch forecast %s/%s: momentum=%.4f", eid, stype, result.momentum
                )
            except Exception as exc:
                logger.error("Batch forecast failed for %s/%s: %s", eid, stype, exc)
        return results
