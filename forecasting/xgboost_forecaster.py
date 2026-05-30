"""
XGBoost-based forecasting as an alternative to Prophet.

Features:
  - Lag features (1, 3, 7, 14, 30 days)
  - Rolling statistics (mean, std) over multiple windows
  - Calendar features (day of week, month, quarter)
  - Trend components (linear trend index)
  - Quantile regression for uncertainty estimation
  - TimeSeriesSplit cross-validation
  - Feature importance reporting
  - Model persistence with pickle
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# XGBoost is an optional heavy dependency; import lazily.
try:
    from xgboost import XGBRegressor  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None  # type: ignore
    TimeSeriesSplit = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    mean_squared_error = None  # type: ignore
    XGBOOST_AVAILABLE = False
    logger.warning(
        "XGBoost or scikit-learn not installed. XGBoostForecaster unavailable. "
        "Install with: pip install xgboost scikit-learn"
    )


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


@dataclass
class XGBForecastResult:
    """Output of the XGBoost forecaster.

    Attributes:
        entity_id: Entity identifier.
        signal_type: Signal being forecast.
        predictions: Point-forecast values.
        lower_bound: Lower quantile predictions (Q10).
        upper_bound: Upper quantile predictions (Q90).
        feature_importance: Dict mapping feature name → importance score.
        cv_metrics: Cross-validation metrics (mae, rmse) from training.
    """

    entity_id: str
    signal_type: str
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    feature_importance: Dict[str, float] = field(default_factory=dict)
    cv_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "signal_type": self.signal_type,
            "predictions": [round(v, 4) for v in self.predictions],
            "lower_bound": [round(v, 4) for v in self.lower_bound],
            "upper_bound": [round(v, 4) for v in self.upper_bound],
            "feature_importance": {k: round(v, 6) for k, v in self.feature_importance.items()},
            "cv_metrics": self.cv_metrics,
        }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


LAG_DAYS = [1, 3, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]


def _engineer_features(series: pd.Series) -> pd.DataFrame:
    """Build a feature matrix from a univariate time series.

    Creates:
      - Lag features for each day in LAG_DAYS
      - Rolling mean and std for each window in ROLLING_WINDOWS
      - Calendar features: day_of_week, month, quarter, is_weekend
      - Linear trend index (position in series)

    Args:
        series: Pandas Series with a DatetimeIndex and float values.

    Returns:
        Feature DataFrame aligned to the input series index.
    """
    df = pd.DataFrame({"y": series})

    # Lag features
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling statistics
    for win in ROLLING_WINDOWS:
        df[f"roll_mean_{win}"] = df["y"].shift(1).rolling(win).mean()
        df[f"roll_std_{win}"] = df["y"].shift(1).rolling(win).std()

    # Calendar features
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    else:
        df["day_of_week"] = 0
        df["month"] = 1
        df["quarter"] = 1
        df["is_weekend"] = 0

    # Linear trend index
    df["trend_index"] = np.arange(len(df), dtype=float)

    return df


def _prepare_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop NaN rows (introduced by lags) and split into X, y.

    Args:
        df: Feature DataFrame with a 'y' target column.

    Returns:
        Tuple (X, y) ready for XGBRegressor.
    """
    df_clean = df.dropna()
    y = df_clean["y"]
    X = df_clean.drop(columns=["y"])
    return X, y


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------


class XGBoostForecaster:
    """XGBoost-based univariate time-series forecaster.

    Trains three XGBRegressor models per entity/signal pair:
      - Median (Q50) for point forecasts
      - Lower (Q10) for the lower prediction bound
      - Upper (Q90) for the upper prediction bound

    All models are persisted to disk via pickle for fast re-loading.

    Args:
        model_cache_dir: Directory to store serialized models.
        n_estimators: Number of XGBoost trees.
        learning_rate: Learning rate for gradient boosting.
        max_depth: Maximum tree depth.
        cv_folds: Number of time-series cross-validation folds.
    """

    def __init__(
        self,
        model_cache_dir: str = "/tmp/xgb_cache",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        cv_folds: int = 3,
    ) -> None:
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "XGBoost and scikit-learn must be installed to use XGBoostForecaster."
            )
        self._cache_dir = Path(model_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._cv_folds = cv_folds

        # Registry: (entity_id, signal_type) → {"q50": model, "q10": model, "q90": model}
        self._models: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Feature names per key
        self._feature_names: Dict[Tuple[str, str], List[str]] = {}
        # Stored series for multi-step prediction
        self._series_store: Dict[Tuple[str, str], pd.Series] = {}

        logger.info(
            "XGBoostForecaster initialized (n_estimators=%d, lr=%.3f, depth=%d)",
            n_estimators, learning_rate, max_depth,
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, entity_id: str, signal_type: str) -> Path:
        safe_id = entity_id.replace("/", "_").replace(":", "_")
        return self._cache_dir / f"xgb__{safe_id}__{signal_type}.pkl"

    def _save_models(self, entity_id: str, signal_type: str) -> None:
        key = (entity_id, signal_type)
        path = self._cache_path(entity_id, signal_type)
        try:
            payload = {
                "models": self._models.get(key),
                "feature_names": self._feature_names.get(key),
                "series": self._series_store.get(key),
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            logger.debug("XGB models saved to %s", path)
        except Exception as exc:
            logger.warning("Failed to save XGB models to %s: %s", path, exc)

    def _load_models(self, entity_id: str, signal_type: str) -> bool:
        path = self._cache_path(entity_id, signal_type)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            key = (entity_id, signal_type)
            self._models[key] = payload["models"]
            self._feature_names[key] = payload["feature_names"]
            self._series_store[key] = payload["series"]
            logger.debug("XGB models loaded from %s", path)
            return True
        except Exception as exc:
            logger.warning("Failed to load XGB models from %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _build_model(self, quantile: Optional[float] = None) -> "XGBRegressor":
        """Build an XGBRegressor configured for point or quantile regression.

        Args:
            quantile: If provided, uses quantile loss (alpha=quantile).
                      If None, uses squared-error loss.

        Returns:
            Configured XGBRegressor instance.
        """
        if quantile is not None:
            return XGBRegressor(
                n_estimators=self._n_estimators,
                learning_rate=self._learning_rate,
                max_depth=self._max_depth,
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
        return XGBRegressor(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        entity_id: str = "default",
        signal_type: str = "general",
        series: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Train three XGBoost models (Q10, Q50, Q90) on the feature matrix.

        Performs TimeSeriesSplit cross-validation on the Q50 model and logs
        MAE/RMSE metrics.

        Args:
            X: Feature matrix (output of _prepare_Xy or custom features).
            y: Target series.
            entity_id: Entity identifier (for caching).
            signal_type: Signal name (for caching).
            series: Original raw series (stored for multi-step prediction).

        Returns:
            Dict with cross-validation metrics {"mae": ..., "rmse": ...}.
        """
        logger.info(
            "Training XGB models for %s/%s (%d samples).", entity_id, signal_type, len(X)
        )

        # Cross-validate Q50 model
        cv_metrics = self._cross_validate(X, y)

        # Fit all three quantile models on full data
        q10_model = self._build_model(quantile=0.10)
        q50_model = self._build_model(quantile=0.50)
        q90_model = self._build_model(quantile=0.90)

        q10_model.fit(X, y)
        q50_model.fit(X, y)
        q90_model.fit(X, y)

        key = (entity_id, signal_type)
        self._models[key] = {"q10": q10_model, "q50": q50_model, "q90": q90_model}
        self._feature_names[key] = list(X.columns)
        if series is not None:
            self._series_store[key] = series

        self._save_models(entity_id, signal_type)
        logger.info(
            "XGB training complete for %s/%s. CV MAE=%.4f RMSE=%.4f",
            entity_id, signal_type, cv_metrics.get("mae", 0), cv_metrics.get("rmse", 0),
        )
        return cv_metrics

    def train_from_series(
        self,
        series: pd.Series,
        entity_id: str = "default",
        signal_type: str = "general",
    ) -> Dict[str, float]:
        """End-to-end training from a raw time series.

        Runs feature engineering, drops NaNs, and calls ``train()``.

        Args:
            series: Pandas Series with DatetimeIndex and float values.
            entity_id: Entity identifier.
            signal_type: Signal name.

        Returns:
            Cross-validation metrics.
        """
        feature_df = _engineer_features(series)
        X, y = _prepare_Xy(feature_df)
        return self.train(X, y, entity_id=entity_id, signal_type=signal_type, series=series)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """TimeSeriesSplit cross-validation using the Q50 model.

        Args:
            X: Feature matrix.
            y: Target series.

        Returns:
            Dict with "mae" and "rmse" averaged across folds.
        """
        n_folds = min(self._cv_folds, len(X) // 10)
        if n_folds < 2:
            logger.debug("Not enough samples for CV; skipping.")
            return {"mae": 0.0, "rmse": 0.0}

        tscv = TimeSeriesSplit(n_splits=n_folds)
        maes, rmses = [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = self._build_model(quantile=0.50)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))

        return {
            "mae": float(np.mean(maes)),
            "rmse": float(np.mean(rmses)),
            "n_folds": n_folds,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        entity_id: str = "default",
        signal_type: str = "general",
    ) -> XGBForecastResult:
        """Predict on a prepared feature matrix using cached models.

        Args:
            X: Feature matrix matching the training feature names.
            entity_id: Entity to load models for.
            signal_type: Signal to load models for.

        Returns:
            XGBForecastResult with point and quantile predictions.

        Raises:
            RuntimeError: If no model is found.
        """
        key = (entity_id, signal_type)
        if key not in self._models and not self._load_models(entity_id, signal_type):
            raise RuntimeError(
                f"No XGB model found for {entity_id}/{signal_type}. Call train() first."
            )

        models = self._models[key]
        # Align feature columns
        feat_names = self._feature_names.get(key)
        if feat_names:
            X_aligned = X.reindex(columns=feat_names, fill_value=0.0)
        else:
            X_aligned = X

        preds_q50 = models["q50"].predict(X_aligned)
        preds_q10 = models["q10"].predict(X_aligned)
        preds_q90 = models["q90"].predict(X_aligned)

        # Feature importance from Q50 model
        importance_raw = models["q50"].feature_importances_
        feature_names = feat_names or list(X.columns)
        importance = dict(zip(feature_names, importance_raw.tolist()))

        return XGBForecastResult(
            entity_id=entity_id,
            signal_type=signal_type,
            predictions=np.clip(preds_q50, 0, None).tolist(),
            lower_bound=np.clip(preds_q10, 0, None).tolist(),
            upper_bound=preds_q90.tolist(),
            feature_importance=importance,
        )

    def predict_future(
        self,
        periods: int,
        entity_id: str = "default",
        signal_type: str = "general",
        last_date: Optional[pd.Timestamp] = None,
        freq: str = "D",
    ) -> XGBForecastResult:
        """Iterative multi-step-ahead forecast using stored series.

        Uses a recursive strategy: each predicted value is fed back as a lag
        feature for the next step.  This introduces error accumulation but
        is appropriate for short horizons (≤ 30 days).

        Args:
            periods: Number of steps to forecast ahead.
            entity_id: Entity identifier.
            signal_type: Signal name.
            last_date: Last date in the training series.  Auto-detected if None.
            freq: Frequency string for future date generation.

        Returns:
            XGBForecastResult with ``periods`` predictions.
        """
        key = (entity_id, signal_type)
        if key not in self._models and not self._load_models(entity_id, signal_type):
            raise RuntimeError(
                f"No XGB model found for {entity_id}/{signal_type}. Call train() first."
            )

        stored_series = self._series_store.get(key)
        if stored_series is None:
            raise RuntimeError(
                f"No stored series for {entity_id}/{signal_type}. "
                "Use train_from_series() to enable multi-step prediction."
            )

        # Extend the series with NaN placeholders for future steps
        if last_date is None:
            last_date = stored_series.index[-1]

        future_index = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=periods,
            freq=freq,
        )

        extended = stored_series.copy()
        all_predictions_q50: List[float] = []
        all_predictions_q10: List[float] = []
        all_predictions_q90: List[float] = []

        models = self._models[key]
        feat_names = self._feature_names.get(key, [])

        for fut_date in future_index:
            # Rebuild features on the extended series
            feat_df = _engineer_features(extended)
            X_last = feat_df.tail(1).drop(columns=["y"], errors="ignore")
            if feat_names:
                X_last = X_last.reindex(columns=feat_names, fill_value=0.0)

            p_q50 = float(max(models["q50"].predict(X_last)[0], 0))
            p_q10 = float(max(models["q10"].predict(X_last)[0], 0))
            p_q90 = float(models["q90"].predict(X_last)[0])

            all_predictions_q50.append(p_q50)
            all_predictions_q10.append(p_q10)
            all_predictions_q90.append(p_q90)

            # Feed Q50 prediction back into the series
            new_row = pd.Series([p_q50], index=[fut_date])
            extended = pd.concat([extended, new_row])

        importance_raw = models["q50"].feature_importances_
        importance = dict(zip(feat_names or [], importance_raw.tolist()))

        return XGBForecastResult(
            entity_id=entity_id,
            signal_type=signal_type,
            predictions=all_predictions_q50,
            lower_bound=all_predictions_q10,
            upper_bound=all_predictions_q90,
            feature_importance=importance,
        )

    # ------------------------------------------------------------------
    # Feature importance reporting
    # ------------------------------------------------------------------

    def get_feature_importance(
        self,
        entity_id: str,
        signal_type: str,
        top_n: int = 15,
    ) -> List[Tuple[str, float]]:
        """Return top-N most important features for a trained model.

        Args:
            entity_id: Entity identifier.
            signal_type: Signal name.
            top_n: Number of features to return.

        Returns:
            List of (feature_name, importance) tuples sorted descending.
        """
        key = (entity_id, signal_type)
        if key not in self._models:
            self._load_models(entity_id, signal_type)
        models = self._models.get(key)
        if not models:
            return []

        feat_names = self._feature_names.get(key, [])
        raw = models["q50"].feature_importances_
        paired = list(zip(feat_names, raw.tolist()))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_n]
