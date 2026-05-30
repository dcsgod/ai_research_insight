"""
forecast_service.py — Forecast Service
=========================================
FastAPI service layer for time-series trend forecasting.

Provides:
- Per-entity forecasts using Prophet (default) or XGBoost (fallback)
- Bulk forecast computation for all tracked entities
- Momentum metrics derived from forecast slopes
- Redis caching (6-hour TTL)
- PostgreSQL persistence of computed series

Dependencies:
- ProphetForecaster  → forecasting.prophet_forecaster.ProphetForecaster
- XGBoostForecaster  → forecasting.xgboost_forecaster.XGBoostForecaster
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger("forecast_service")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EntityType = Literal["paper", "repository", "topic", "model"]
ForecasterType = Literal["prophet", "xgboost"]


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class ForecastPoint(BaseModel):
    """A single data point in a forecast series."""

    date: datetime
    predicted_value: float
    ci_low: float   # Lower bound of confidence interval
    ci_high: float  # Upper bound of confidence interval


class ForecastSeries(BaseModel):
    """Complete forecast series for a single entity."""

    entity_id: str
    entity_type: EntityType
    forecaster_used: ForecasterType = "prophet"
    periods: int = 30
    forecast_points: list[ForecastPoint] = Field(default_factory=list)
    # Summary statistics derived from forecast
    predicted_score: float = Field(
        default=0.0,
        description="Predicted trend score at end of forecast horizon",
    )
    ci_low: float = 0.0
    ci_high: float = 0.0
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    model_accuracy: float | None = Field(
        default=None,
        description="Backtesting MAPE score (lower is better)",
    )


class MomentumMetrics(BaseModel):
    """Momentum and velocity metrics for a single entity."""

    entity_id: str
    entity_type: EntityType
    momentum: float = Field(
        description=(
            "Rate of change in trend score. "
            "Positive = accelerating, negative = decelerating."
        )
    )
    velocity_7d: float = Field(description="7-day rate of change")
    velocity_30d: float = Field(description="30-day rate of change")
    acceleration: float = Field(
        description="Second derivative — how quickly momentum is changing"
    )
    is_breakout: bool = Field(
        default=False,
        description="True if momentum exceeds 2 standard deviations above historical mean",
    )
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# ForecastService
# ---------------------------------------------------------------------------
class ForecastService:
    """
    Service class for entity trend forecasting.

    Responsibilities:
    - Select appropriate forecasting model per entity
    - Store computed forecasts in PostgreSQL
    - Cache forecast series in Redis (6-hour TTL)
    - Expose momentum/velocity metrics derived from forecast slopes

    Dependencies (injected via constructor):
    - redis_client: async Redis client
    - db_session:   SQLAlchemy AsyncSession
    """

    CACHE_TTL_SECONDS: int = 6 * 60 * 60  # 6 hours
    CACHE_KEY_PREFIX: str = "forecasts"
    MIN_DATA_POINTS: int = 14  # Minimum history needed for Prophet
    DEFAULT_PERIODS: int = 30  # Default forecast horizon (days)

    def __init__(
        self,
        redis_client: Any | None = None,
        db_session: Any | None = None,
    ) -> None:
        self._redis = redis_client
        self._db = db_session
        self._log = structlog.get_logger("forecast_service")

        # Lazy-loaded forecasters
        self._prophet_forecaster: Any | None = None
        self._xgboost_forecaster: Any | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    async def get_forecast(
        self,
        entity_id: str,
        entity_type: EntityType,
        periods: int = 30,
        force_refresh: bool = False,
        preferred_forecaster: ForecasterType = "prophet",
    ) -> ForecastSeries:
        """
        Retrieve or compute a forecast series for a given entity.

        Checks Redis cache first. On miss, fetches historical time-series
        from PostgreSQL, runs the forecaster, caches, and stores the result.

        Args:
            entity_id:           Unique entity identifier.
            entity_type:         Type of entity to forecast.
            periods:             Forecast horizon in days.
            force_refresh:       Bypass cache and recompute.
            preferred_forecaster: 'prophet' or 'xgboost'.

        Returns:
            ForecastSeries with forecast_points and summary statistics.
        """
        periods = max(1, min(periods, 365))
        cache_key = self._build_cache_key(entity_id, entity_type, periods)

        # ── Try cache ─────────────────────────────────────────────────────
        if not force_refresh:
            cached = await self._get_from_cache(cache_key)
            if cached is not None:
                self._log.debug("Forecast cache hit", entity_id=entity_id)
                return ForecastSeries(**cached)

        # ── Fetch historical data ─────────────────────────────────────────
        history = await self._fetch_history(entity_id, entity_type)

        if len(history) < self.MIN_DATA_POINTS:
            self._log.warning(
                "Insufficient history for forecasting",
                entity_id=entity_id,
                data_points=len(history),
                min_required=self.MIN_DATA_POINTS,
            )
            return self._empty_forecast(entity_id, entity_type, periods)

        # ── Run forecaster ─────────────────────────────────────────────────
        series = await self._run_forecast(
            entity_id=entity_id,
            entity_type=entity_type,
            history=history,
            periods=periods,
            preferred_forecaster=preferred_forecaster,
        )

        # ── Cache result ───────────────────────────────────────────────────
        await self._set_cache(cache_key, series.model_dump(mode="json"))

        # ── Persist to PostgreSQL ──────────────────────────────────────────
        await self._persist_forecast(series)

        return series

    async def compute_all_forecasts(self) -> dict[str, int]:
        """
        Compute forecasts for all tracked entities.

        Fetches all entity IDs from PostgreSQL for each entity type,
        runs forecasts in batches, and stores results.

        Returns:
            Dict with counts of forecasts computed per entity type.
        """
        self._log.info("Starting bulk forecast computation")
        counts: dict[str, int] = {}

        for entity_type in ("paper", "repository", "topic"):
            entity_ids = await self._fetch_all_entity_ids(entity_type)  # type: ignore[arg-type]
            count = 0
            errors = 0

            self._log.info(
                "Computing forecasts for entity type",
                entity_type=entity_type,
                total=len(entity_ids),
            )

            for entity_id in entity_ids:
                try:
                    await self.get_forecast(
                        entity_id=entity_id,
                        entity_type=entity_type,  # type: ignore[arg-type]
                        periods=self.DEFAULT_PERIODS,
                        force_refresh=True,
                    )
                    count += 1
                except Exception as exc:
                    errors += 1
                    self._log.warning(
                        "Forecast failed for entity",
                        entity_id=entity_id,
                        entity_type=entity_type,
                        error=str(exc),
                    )

            counts[entity_type] = count
            self._log.info(
                "Finished forecasts for entity type",
                entity_type=entity_type,
                computed=count,
                errors=errors,
            )

        self._log.info("Bulk forecast computation complete", counts=counts)
        return counts

    async def get_momentum_metrics(
        self,
        entity_id: str,
        entity_type: EntityType = "paper",
    ) -> MomentumMetrics:
        """
        Compute momentum and velocity metrics for a given entity.

        Derives metrics from the forecast slope and historical trend data.

        Args:
            entity_id:   Unique entity identifier.
            entity_type: Entity type for history lookup.

        Returns:
            MomentumMetrics with velocity and acceleration values.
        """
        cache_key = f"momentum:{entity_id}"
        cached = await self._get_from_cache(cache_key)
        if cached is not None:
            return MomentumMetrics(**cached)

        history = await self._fetch_history(entity_id, entity_type)

        if len(history) < 7:
            return MomentumMetrics(
                entity_id=entity_id,
                entity_type=entity_type,
                momentum=0.0,
                velocity_7d=0.0,
                velocity_30d=0.0,
                acceleration=0.0,
                is_breakout=False,
            )

        # Compute momentum from historical trend scores
        scores = [h["value"] for h in history]
        scores_7d = scores[-7:]
        scores_30d = scores[-30:] if len(scores) >= 30 else scores

        velocity_7d = (scores_7d[-1] - scores_7d[0]) / max(len(scores_7d) - 1, 1)
        velocity_30d = (scores_30d[-1] - scores_30d[0]) / max(len(scores_30d) - 1, 1)

        # Acceleration: change in velocity (compare last 7d vs prior 7d)
        if len(scores) >= 14:
            prior_7d = scores[-14:-7]
            prior_velocity = (prior_7d[-1] - prior_7d[0]) / max(len(prior_7d) - 1, 1)
            acceleration = velocity_7d - prior_velocity
        else:
            acceleration = 0.0

        # Breakout detection: momentum > mean + 2*std of all weekly velocities
        weekly_velocities = []
        for i in range(7, len(scores)):
            weekly_velocities.append(scores[i] - scores[i - 7])

        is_breakout = False
        if len(weekly_velocities) >= 4:
            import statistics
            mean_vel = statistics.mean(weekly_velocities)
            std_vel = statistics.stdev(weekly_velocities) if len(weekly_velocities) > 1 else 0.0
            threshold = mean_vel + 2 * std_vel
            is_breakout = velocity_7d > threshold

        metrics = MomentumMetrics(
            entity_id=entity_id,
            entity_type=entity_type,
            momentum=velocity_7d,
            velocity_7d=round(velocity_7d, 6),
            velocity_30d=round(velocity_30d, 6),
            acceleration=round(acceleration, 6),
            is_breakout=is_breakout,
        )

        # Cache momentum metrics for 1 hour
        await self._set_cache(
            cache_key,
            metrics.model_dump(mode="json"),
            ttl=3600,
        )
        return metrics

    # -----------------------------------------------------------------------
    # Private: Forecaster invocation
    # -----------------------------------------------------------------------
    async def _run_forecast(
        self,
        entity_id: str,
        entity_type: EntityType,
        history: list[dict[str, Any]],
        periods: int,
        preferred_forecaster: ForecasterType,
    ) -> ForecastSeries:
        """Select and run the appropriate forecasting model."""

        # Try Prophet first
        if preferred_forecaster == "prophet":
            try:
                return await self._run_prophet(entity_id, entity_type, history, periods)
            except Exception as exc:
                self._log.warning(
                    "Prophet forecast failed, falling back to XGBoost",
                    entity_id=entity_id,
                    error=str(exc),
                )

        # Fallback: XGBoost
        try:
            return await self._run_xgboost(entity_id, entity_type, history, periods)
        except Exception as exc:
            self._log.error(
                "XGBoost forecast also failed",
                entity_id=entity_id,
                error=str(exc),
            )
            return self._empty_forecast(entity_id, entity_type, periods)

    async def _run_prophet(
        self,
        entity_id: str,
        entity_type: EntityType,
        history: list[dict[str, Any]],
        periods: int,
    ) -> ForecastSeries:
        """Run Facebook Prophet time-series forecast."""
        import asyncio

        def _sync_prophet() -> ForecastSeries:
            try:
                from forecasting.prophet_forecaster import ProphetForecaster  # type: ignore[import]
                forecaster = ProphetForecaster()
            except ImportError:
                # Inline stub implementation
                return self._stub_forecast(entity_id, entity_type, periods, "prophet")

            forecast_df = forecaster.forecast(history=history, periods=periods)
            return self._df_to_series(
                entity_id=entity_id,
                entity_type=entity_type,
                df=forecast_df,
                periods=periods,
                forecaster="prophet",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_prophet)

    async def _run_xgboost(
        self,
        entity_id: str,
        entity_type: EntityType,
        history: list[dict[str, Any]],
        periods: int,
    ) -> ForecastSeries:
        """Run XGBoost-based trend forecast."""
        import asyncio

        def _sync_xgboost() -> ForecastSeries:
            try:
                from forecasting.xgboost_forecaster import XGBoostForecaster  # type: ignore[import]
                forecaster = XGBoostForecaster()
            except ImportError:
                return self._stub_forecast(entity_id, entity_type, periods, "xgboost")

            forecast_df = forecaster.forecast(history=history, periods=periods)
            return self._df_to_series(
                entity_id=entity_id,
                entity_type=entity_type,
                df=forecast_df,
                periods=periods,
                forecaster="xgboost",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_xgboost)

    # -----------------------------------------------------------------------
    # Private: Database helpers
    # -----------------------------------------------------------------------
    async def _fetch_history(
        self,
        entity_id: str,
        entity_type: EntityType,
        days: int = 90,
    ) -> list[dict[str, Any]]:
        """Fetch historical trend score time-series from PostgreSQL."""
        if self._db is None:
            return self._synthetic_history(days)

        try:
            from sqlalchemy import text as sa_text

            query = sa_text("""
                SELECT recorded_at AS date, trend_score AS value
                FROM entity_trend_history
                WHERE entity_id = :entity_id
                  AND entity_type = :entity_type
                  AND recorded_at >= NOW() - INTERVAL ':days days'
                ORDER BY recorded_at ASC
            """)
            result = await self._db.execute(
                query,
                {"entity_id": entity_id, "entity_type": entity_type, "days": days},
            )
            rows = result.mappings().all()
            return [{"date": str(row["date"]), "value": float(row["value"])} for row in rows]
        except Exception as exc:
            self._log.warning(
                "DB history fetch failed — using synthetic data",
                entity_id=entity_id,
                error=str(exc),
            )
            return self._synthetic_history(days)

    async def _fetch_all_entity_ids(self, entity_type: EntityType) -> list[str]:
        """Retrieve all entity IDs of a given type from PostgreSQL."""
        table_map: dict[str, str] = {
            "paper": "papers",
            "repository": "repositories",
            "topic": "topics",
            "model": "hf_models",
        }
        table = table_map.get(entity_type, "papers")

        if self._db is None:
            return []

        try:
            from sqlalchemy import text as sa_text

            result = await self._db.execute(sa_text(f"SELECT id FROM {table}"))
            return [str(row[0]) for row in result.fetchall()]
        except Exception as exc:
            self._log.error(
                "Failed to fetch entity IDs",
                entity_type=entity_type,
                error=str(exc),
            )
            return []

    async def _persist_forecast(self, series: ForecastSeries) -> None:
        """Upsert a ForecastSeries into PostgreSQL."""
        if self._db is None:
            return

        try:
            from sqlalchemy import text as sa_text

            await self._db.execute(
                sa_text("""
                    INSERT INTO entity_forecasts
                        (entity_id, entity_type, forecaster, periods,
                         predicted_score, ci_low, ci_high,
                         model_accuracy, generated_at, payload)
                    VALUES
                        (:entity_id, :entity_type, :forecaster, :periods,
                         :predicted_score, :ci_low, :ci_high,
                         :model_accuracy, :generated_at, :payload)
                    ON CONFLICT (entity_id, entity_type)
                    DO UPDATE SET
                        forecaster      = EXCLUDED.forecaster,
                        periods         = EXCLUDED.periods,
                        predicted_score = EXCLUDED.predicted_score,
                        ci_low          = EXCLUDED.ci_low,
                        ci_high         = EXCLUDED.ci_high,
                        model_accuracy  = EXCLUDED.model_accuracy,
                        generated_at    = EXCLUDED.generated_at,
                        payload         = EXCLUDED.payload
                """),
                {
                    "entity_id": series.entity_id,
                    "entity_type": series.entity_type,
                    "forecaster": series.forecaster_used,
                    "periods": series.periods,
                    "predicted_score": series.predicted_score,
                    "ci_low": series.ci_low,
                    "ci_high": series.ci_high,
                    "model_accuracy": series.model_accuracy,
                    "generated_at": series.generated_at,
                    "payload": json.dumps(series.model_dump(mode="json"), default=str),
                },
            )
            await self._db.commit()
        except Exception as exc:
            self._log.warning(
                "Failed to persist forecast to DB",
                entity_id=series.entity_id,
                error=str(exc),
            )

    # -----------------------------------------------------------------------
    # Private: Cache helpers
    # -----------------------------------------------------------------------
    def _build_cache_key(
        self,
        entity_id: str,
        entity_type: EntityType,
        periods: int,
    ) -> str:
        return f"{self.CACHE_KEY_PREFIX}:{entity_type}:{entity_id}:{periods}d"

    async def _get_from_cache(self, key: str) -> dict[str, Any] | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            self._log.warning("Cache read error", key=key, error=str(exc))
            return None

    async def _set_cache(
        self,
        key: str,
        data: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        if self._redis is None:
            return
        ttl = ttl or self.CACHE_TTL_SECONDS
        try:
            await self._redis.setex(key, ttl, json.dumps(data, default=str))
        except Exception as exc:
            self._log.warning("Cache write error", key=key, error=str(exc))

    # -----------------------------------------------------------------------
    # Private: Utilities
    # -----------------------------------------------------------------------
    def _df_to_series(
        self,
        entity_id: str,
        entity_type: EntityType,
        df: Any,
        periods: int,
        forecaster: ForecasterType,
    ) -> ForecastSeries:
        """Convert a pandas DataFrame (from forecasters) to ForecastSeries."""
        points: list[ForecastPoint] = []

        # Expected columns: ds/date, yhat/predicted, yhat_lower/ci_low, yhat_upper/ci_high
        for _, row in df.iterrows():
            points.append(
                ForecastPoint(
                    date=row.get("ds") or row.get("date"),
                    predicted_value=float(row.get("yhat") or row.get("predicted", 0)),
                    ci_low=float(row.get("yhat_lower") or row.get("ci_low", 0)),
                    ci_high=float(row.get("yhat_upper") or row.get("ci_high", 1)),
                )
            )

        predicted_score = points[-1].predicted_value if points else 0.0
        ci_low = points[-1].ci_low if points else 0.0
        ci_high = points[-1].ci_high if points else 1.0

        return ForecastSeries(
            entity_id=entity_id,
            entity_type=entity_type,
            forecaster_used=forecaster,
            periods=periods,
            forecast_points=points,
            predicted_score=round(max(0.0, min(1.0, predicted_score)), 4),
            ci_low=round(max(0.0, ci_low), 4),
            ci_high=round(min(1.0, ci_high), 4),
        )

    def _empty_forecast(
        self,
        entity_id: str,
        entity_type: EntityType,
        periods: int,
    ) -> ForecastSeries:
        """Return an empty forecast when computation is not possible."""
        return ForecastSeries(
            entity_id=entity_id,
            entity_type=entity_type,
            forecaster_used="prophet",
            periods=periods,
            forecast_points=[],
            predicted_score=0.0,
            ci_low=0.0,
            ci_high=0.0,
        )

    def _stub_forecast(
        self,
        entity_id: str,
        entity_type: EntityType,
        periods: int,
        forecaster: ForecasterType,
    ) -> ForecastSeries:
        """Generate a synthetic forecast series for development/testing."""
        import math

        base = 0.5
        points: list[ForecastPoint] = []
        now = datetime.now(timezone.utc)

        for i in range(periods):
            noise = math.sin(i * 0.3) * 0.05
            value = min(1.0, max(0.0, base + (i / periods) * 0.2 + noise))
            points.append(
                ForecastPoint(
                    date=now + timedelta(days=i),
                    predicted_value=round(value, 4),
                    ci_low=round(max(0.0, value - 0.1), 4),
                    ci_high=round(min(1.0, value + 0.1), 4),
                )
            )

        return ForecastSeries(
            entity_id=entity_id,
            entity_type=entity_type,
            forecaster_used=forecaster,
            periods=periods,
            forecast_points=points,
            predicted_score=round(points[-1].predicted_value, 4) if points else 0.5,
            ci_low=round(points[-1].ci_low, 4) if points else 0.4,
            ci_high=round(points[-1].ci_high, 4) if points else 0.6,
        )

    @staticmethod
    def _synthetic_history(days: int = 90) -> list[dict[str, Any]]:
        """Generate synthetic trend history for testing."""
        import math
        import random

        random.seed(42)
        base = 0.4
        history = []
        now = datetime.now(timezone.utc)

        for i in range(days):
            date = now - timedelta(days=days - i)
            noise = random.gauss(0, 0.02)
            trend = math.sin(i * 0.1) * 0.1
            value = max(0.0, min(1.0, base + trend + noise + i / (days * 5)))
            history.append({"date": date.isoformat(), "value": round(value, 4)})

        return history
