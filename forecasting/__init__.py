"""Forecasting engine for trend momentum prediction."""

from forecasting.prophet_forecaster import ProphetForecaster, ForecastResult
from forecasting.xgboost_forecaster import XGBoostForecaster
from forecasting.aggregator import SignalAggregator, MomentumMetrics

__all__ = [
    "ProphetForecaster",
    "ForecastResult",
    "XGBoostForecaster",
    "SignalAggregator",
    "MomentumMetrics",
]
