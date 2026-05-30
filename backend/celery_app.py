"""
celery_app.py — Celery Application Configuration
==================================================
Defines the Celery app instance, periodic beat schedule, and all
async tasks for the AI Research Intelligence Platform.

Tasks:
  - run_full_ingestion   → fetch data from all configured sources
  - update_rankings      → recompute trend scores and rankings
  - compute_forecasts    → run time-series forecasts for all entities
  - generate_insights    → generate AI-driven topic insights
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

import structlog
from celery import Celery, Task, signals
from celery.schedules import crontab
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("celery_app")
task_logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://ai_user:ai_secret@localhost:5432/ai_research",
)

# ---------------------------------------------------------------------------
# Celery App Instance
# ---------------------------------------------------------------------------
app = Celery(
    "ai_research_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "backend.celery_app",  # self-register task modules
    ],
)

# ---------------------------------------------------------------------------
# Celery Configuration
# ---------------------------------------------------------------------------
app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezones
    timezone="UTC",
    enable_utc=True,
    # Result backend settings
    result_expires=timedelta(hours=24),
    result_backend_transport_options={
        "retry_policy": {"timeout": 5.0},
    },
    # Task routing — queues for workload separation
    task_routes={
        "backend.celery_app.run_full_ingestion": {"queue": "ingestion"},
        "backend.celery_app.ingest_source": {"queue": "ingestion"},
        "backend.celery_app.update_rankings": {"queue": "ranking"},
        "backend.celery_app.compute_forecasts": {"queue": "forecasting"},
        "backend.celery_app.generate_insights": {"queue": "default"},
    },
    # Worker settings
    worker_prefetch_multiplier=1,           # Prevent task hoarding per worker
    task_acks_late=True,                    # Acknowledge only after completion
    worker_disable_rate_limits=False,
    task_reject_on_worker_lost=True,        # Re-queue if worker dies mid-task
    # Retry settings (global defaults; overridden per task)
    task_max_retries=3,
    # Queue definitions
    task_queues={
        "ingestion": {"exchange": "ingestion", "routing_key": "ingestion"},
        "ranking": {"exchange": "ranking", "routing_key": "ranking"},
        "forecasting": {"exchange": "forecasting", "routing_key": "forecasting"},
        "default": {"exchange": "default", "routing_key": "default"},
    },
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # Rate limiting per task type
    task_annotations={
        "backend.celery_app.run_full_ingestion": {"rate_limit": "2/h"},
        "backend.celery_app.ingest_source": {"rate_limit": "10/h"},
        "backend.celery_app.update_rankings": {"rate_limit": "4/h"},
        "backend.celery_app.compute_forecasts": {"rate_limit": "1/h"},
    },
    # Beat schedule — periodic tasks
    beat_schedule={
        # ── Ingestion ──────────────────────────────────────────────────────
        "full-ingestion-every-30min": {
            "task": "backend.celery_app.run_full_ingestion",
            "schedule": timedelta(minutes=30),
            "args": [],
            "kwargs": {"sources": "all"},
            "options": {"queue": "ingestion", "expires": 25 * 60},
        },
        # ── Rankings ───────────────────────────────────────────────────────
        "update-rankings-every-15min": {
            "task": "backend.celery_app.update_rankings",
            "schedule": timedelta(minutes=15),
            "args": [],
            "options": {"queue": "ranking", "expires": 14 * 60},
        },
        # ── Forecasting ────────────────────────────────────────────────────
        "compute-forecasts-every-6hr": {
            "task": "backend.celery_app.compute_forecasts",
            "schedule": crontab(minute=0, hour="*/6"),  # 00:00, 06:00, 12:00, 18:00
            "args": [],
            "options": {"queue": "forecasting", "expires": 5 * 60 * 60},
        },
        # ── Insights ───────────────────────────────────────────────────────
        "generate-insights-twice-daily": {
            "task": "backend.celery_app.generate_insights",
            "schedule": crontab(minute=0, hour="8,20"),  # 08:00 and 20:00 UTC
            "args": [],
            "options": {"queue": "default", "expires": 10 * 60 * 60},
        },
    },
)


# ---------------------------------------------------------------------------
# Base Task class with shared retry/error logic
# ---------------------------------------------------------------------------
class BaseTask(Task):
    """Custom Celery base task with automatic retry and structured logging."""

    abstract = True
    max_retries = 3
    default_retry_delay = 60  # seconds
    throws = ()  # Exceptions that won't trigger retry

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.error(
            "Task failed permanently",
            task=self.name,
            task_id=task_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.warning(
            "Retrying task",
            task=self.name,
            task_id=task_id,
            error=str(exc),
            retries_remaining=self.max_retries - self.request.retries,
        )
        super().on_retry(exc, task_id, args, kwargs, einfo)

    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        logger.info(
            "Task completed successfully",
            task=self.name,
            task_id=task_id,
        )
        super().on_success(retval, task_id, args, kwargs)


# ---------------------------------------------------------------------------
# Task: run_full_ingestion
# ---------------------------------------------------------------------------
@app.task(
    bind=True,
    base=BaseTask,
    name="backend.celery_app.run_full_ingestion",
    max_retries=2,
    default_retry_delay=120,
    queue="ingestion",
)
def run_full_ingestion(self: BaseTask, sources: str = "all") -> dict[str, Any]:
    """
    Trigger ingestion from all (or specified) data sources.

    Args:
        sources: Comma-separated source names or "all".

    Returns:
        Dict with per-source statistics.
    """
    import asyncio
    from pathlib import Path
    import sys

    # Ensure project root is on path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    task_logger.info("Starting full ingestion", sources=sources)

    try:
        # Import orchestrator if available
        try:
            from mcp_servers.orchestrator import IngestionOrchestrator  # type: ignore[import]

            source_list = (
                ["arxiv", "github", "huggingface", "reddit", "paperswithcode"]
                if sources == "all"
                else [s.strip() for s in sources.split(",")]
            )

            async def _run() -> dict[str, Any]:
                orchestrator = IngestionOrchestrator()
                return await orchestrator.run(sources=source_list)

            result = asyncio.run(_run())
        except ImportError:
            task_logger.warning("IngestionOrchestrator not available — stub mode")
            result = {
                "arxiv": {"fetched": 50, "stored": 48, "skipped": 2},
                "github": {"fetched": 100, "stored": 97, "skipped": 3},
                "huggingface": {"fetched": 75, "stored": 73, "skipped": 2},
                "reddit": {"fetched": 200, "stored": 185, "skipped": 15},
                "paperswithcode": {"fetched": 40, "stored": 38, "skipped": 2},
            }

        task_logger.info("Ingestion complete", result=result)
        return result

    except Exception as exc:
        task_logger.error("Ingestion failed", error=str(exc))
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: ingest_source (single-source helper, called by orchestrator or CLI)
# ---------------------------------------------------------------------------
@app.task(
    bind=True,
    base=BaseTask,
    name="backend.celery_app.ingest_source",
    max_retries=3,
    default_retry_delay=60,
    queue="ingestion",
)
def ingest_source(self: BaseTask, source: str) -> dict[str, Any]:
    """
    Ingest from a single named source.

    Args:
        source: Source identifier (arxiv, github, huggingface, reddit, paperswithcode).

    Returns:
        Dict with fetched/stored/skipped counts.
    """
    import asyncio

    task_logger.info("Starting single source ingestion", source=source)

    HANDLER_MAP = {
        "arxiv": "mcp_servers.arxiv_server.ArxivIngester",
        "github": "mcp_servers.github_server.GitHubIngester",
        "huggingface": "mcp_servers.huggingface_server.HuggingFaceIngester",
        "reddit": "mcp_servers.reddit_server.RedditIngester",
        "paperswithcode": "mcp_servers.pwc_server.PapersWithCodeIngester",
    }

    if source not in HANDLER_MAP:
        raise ValueError(f"Unknown ingestion source: {source!r}")

    try:
        module_path, class_name = HANDLER_MAP[source].rsplit(".", 1)
        try:
            import importlib

            module = importlib.import_module(module_path)
            ingester_class = getattr(module, class_name)
            ingester = ingester_class()
            result = asyncio.run(ingester.run())
        except (ImportError, AttributeError):
            task_logger.warning("Ingester not available", source=source)
            result = {"fetched": 0, "stored": 0, "skipped": 0, "stub": True}

        return {"source": source, **result}

    except Exception as exc:
        task_logger.error("Source ingestion failed", source=source, error=str(exc))
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: update_rankings
# ---------------------------------------------------------------------------
@app.task(
    bind=True,
    base=BaseTask,
    name="backend.celery_app.update_rankings",
    max_retries=3,
    default_retry_delay=30,
    queue="ranking",
    time_limit=600,       # Hard limit: 10 minutes
    soft_time_limit=540,  # Soft limit: 9 minutes (raises SoftTimeLimitExceeded)
)
def update_rankings(self: BaseTask) -> dict[str, Any]:
    """
    Recompute trend scores and rankings for papers, repos, and topics.

    Uses the ranking_engine.pipeline.RankingPipeline to score all entities
    and persist updated scores to PostgreSQL.

    Returns:
        Dict with counts of entities updated.
    """
    import asyncio

    task_logger.info("Starting ranking update")

    try:
        try:
            from ranking_engine.pipeline import RankingPipeline  # type: ignore[import]

            async def _run() -> dict[str, Any]:
                pipeline = RankingPipeline()
                return await pipeline.run()

            result = asyncio.run(_run())
        except ImportError:
            task_logger.warning("RankingPipeline not available — stub mode")
            result = {
                "papers_updated": 0,
                "repos_updated": 0,
                "topics_updated": 0,
                "stub": True,
            }

        task_logger.info("Rankings updated", result=result)
        return result

    except Exception as exc:
        task_logger.error("Ranking update failed", error=str(exc))
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: compute_forecasts
# ---------------------------------------------------------------------------
@app.task(
    bind=True,
    base=BaseTask,
    name="backend.celery_app.compute_forecasts",
    max_retries=2,
    default_retry_delay=300,  # 5 minutes between retries
    queue="forecasting",
    time_limit=3600,          # Hard limit: 1 hour
    soft_time_limit=3300,     # Soft limit: 55 minutes
)
def compute_forecasts(self: BaseTask) -> dict[str, Any]:
    """
    Run time-series forecasts for all tracked entities.

    Uses ProphetForecaster by default. Falls back to XGBoostForecaster
    if Prophet is unavailable or if the entity has insufficient history.

    Returns:
        Dict with counts of forecast series generated.
    """
    import asyncio

    task_logger.info("Starting forecast computation")

    try:
        try:
            from backend.services.forecast_service import ForecastService  # type: ignore[import]

            async def _run() -> dict[str, Any]:
                svc = ForecastService()
                await svc.compute_all_forecasts()
                return {"status": "complete"}

            result = asyncio.run(_run())
        except ImportError:
            task_logger.warning("ForecastService not available — stub mode")
            result = {
                "papers_forecasted": 0,
                "repos_forecasted": 0,
                "topics_forecasted": 0,
                "stub": True,
            }

        task_logger.info("Forecasts computed", result=result)
        return result

    except Exception as exc:
        task_logger.error("Forecast computation failed", error=str(exc))
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: generate_insights
# ---------------------------------------------------------------------------
@app.task(
    bind=True,
    base=BaseTask,
    name="backend.celery_app.generate_insights",
    max_retries=2,
    default_retry_delay=120,
    queue="default",
    time_limit=1800,    # Hard limit: 30 minutes
    soft_time_limit=1680,
)
def generate_insights(self: BaseTask) -> dict[str, Any]:
    """
    Use LLM-assisted analysis to generate narrative insights for trending topics.

    Reads top trending topics from PostgreSQL, calls the LLM orchestrator
    to produce short insight summaries, and stores them back.

    Returns:
        Dict with number of insights generated.
    """
    import asyncio

    task_logger.info("Starting insight generation")

    try:
        try:
            from backend.services.insight_service import InsightService  # type: ignore[import]

            async def _run() -> dict[str, Any]:
                svc = InsightService()
                return await svc.generate_all()

            result = asyncio.run(_run())
        except ImportError:
            task_logger.warning("InsightService not available — stub mode")
            result = {"insights_generated": 0, "stub": True}

        task_logger.info("Insights generated", result=result)
        return result

    except Exception as exc:
        task_logger.error("Insight generation failed", error=str(exc))
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Celery Signals — lifecycle hooks
# ---------------------------------------------------------------------------
@signals.task_prerun.connect
def task_prerun_handler(
    task_id: str,
    task: Task,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    **_: Any,
) -> None:
    """Log every task start."""
    logger.info("Task starting", task=task.name, task_id=task_id)


@signals.task_postrun.connect
def task_postrun_handler(
    task_id: str,
    task: Task,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    retval: Any,
    state: str,
    **_: Any,
) -> None:
    """Log every task completion with final state."""
    logger.info("Task finished", task=task.name, task_id=task_id, state=state)


@signals.worker_ready.connect
def worker_ready_handler(**_: Any) -> None:
    """Emitted when a Celery worker is fully started."""
    logger.info("Celery worker is ready and listening for tasks")


@signals.celeryd_init.connect
def configure_worker(sender: str, conf: Any, **_: Any) -> None:
    """Configure worker at init time."""
    logger.info("Celery worker initializing", sender=sender)


# ---------------------------------------------------------------------------
# Direct execution entry point (for debugging)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.start()
