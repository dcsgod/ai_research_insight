"""Backend models package — imports all ORM models for SQLAlchemy metadata registration."""
from backend.models.paper import Paper, PaperSource
from backend.models.repository import Repository
from backend.models.topic import Topic
from backend.models.trend_signal import TrendSignal, TrendScore as TrendScoreModel
from backend.models.insight import Insight, ForecastResult

__all__ = [
    "Paper", "PaperSource",
    "Repository",
    "Topic",
    "TrendSignal", "TrendScoreModel",
    "Insight", "ForecastResult",
]
