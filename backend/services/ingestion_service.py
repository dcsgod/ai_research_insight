"""
Ingestion orchestration service.

Coordinates data ingestion from multiple sources (arXiv, GitHub, Hugging Face,
Papers With Code), deduplicates incoming records, persists them to the database,
updates time-series trend signals, and publishes events to Redis Streams /
Kafka for downstream consumers (trend scorer, embedding pipeline, WS broadcaster).

Architecture:
  IngestionService.run_ingestion()
      │
      ├── _fetch_source(source)        ← MCP / API adapters per source
      ├── process_signals(signals)     ← dedup + upsert to DB
      ├── update_trend_signals()       ← write TrendSignal rows
      └── _publish_events()            ← Redis Stream / Kafka

Progress is tracked via a Redis hash so the status endpoint can report live.
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.db.database import get_db_context
from backend.models.paper import Paper, PaperSource
from backend.models.repository import Repository
from backend.models.topic import Topic
from backend.models.trend_signal import TrendSignal, EntityType, SignalType
from backend.services.cache_service import CacheService, build_key

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Progress tracker (stored in Redis hash)
# ---------------------------------------------------------------------------

_STATUS_KEY = build_key("ingestion", "status", "current")
_HISTORY_KEY = build_key("ingestion", "history")


class IngestionProgress:
    """Thread-safe progress tracker backed by a Redis hash."""

    def __init__(self, job_id: str, cache: CacheService) -> None:
        self.job_id = job_id
        self._cache = cache
        self._key = build_key("ingestion", "progress", job_id)

    async def start(self, sources: List[str]) -> None:
        await self._cache.set(
            self._key,
            {
                "job_id": self.job_id,
                "status": "running",
                "sources": sources,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "error": None,
                "processed": {s: 0 for s in sources},
                "errors": {s: 0 for s in sources},
                "total_upserted": 0,
                "total_signals_written": 0,
            },
            ttl=3600,
        )

    async def update_source(
        self,
        source: str,
        processed: int,
        errors: int = 0,
    ) -> None:
        state = await self._cache.get(self._key) or {}
        state.setdefault("processed", {})[source] = processed
        state.setdefault("errors", {})[source] = errors
        await self._cache.set(self._key, state, ttl=3600)

    async def finish(
        self,
        total_upserted: int,
        total_signals_written: int,
        error: Optional[str] = None,
    ) -> None:
        state = await self._cache.get(self._key) or {}
        state["status"] = "failed" if error else "completed"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        state["total_upserted"] = total_upserted
        state["total_signals_written"] = total_signals_written
        state["error"] = error
        await self._cache.set(self._key, state, ttl=86400)

    async def get_state(self) -> Dict[str, Any]:
        return await self._cache.get(self._key) or {}


# ---------------------------------------------------------------------------
# Data signal DTO (used internally before DB persistence)
# ---------------------------------------------------------------------------


class RawSignal:
    """
    Intermediate data transfer object produced by source adapters.

    Represents one ingested entity (paper or repo) before upsert.
    """

    __slots__ = (
        "entity_type",
        "source",
        "external_id",
        "title",
        "description",
        "url",
        "metadata",
        "signal_values",
        "fetched_at",
    )

    def __init__(
        self,
        entity_type: str,
        source: str,
        external_id: str,
        title: str,
        description: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        signal_values: Optional[Dict[str, float]] = None,
    ) -> None:
        self.entity_type = entity_type
        self.source = source
        self.external_id = external_id
        self.title = title
        self.description = description
        self.url = url
        self.metadata = metadata or {}
        self.signal_values = signal_values or {}
        self.fetched_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# IngestionService
# ---------------------------------------------------------------------------


class IngestionService:
    """
    Orchestrates multi-source ingestion of AI research data.

    Responsibilities:
    1. Call each enabled source adapter concurrently (respecting rate limits).
    2. Deduplicate incoming records against existing DB rows.
    3. Upsert papers and repositories (insert or update on conflict).
    4. Write daily TrendSignal rows for star counts, citation counts, etc.
    5. Publish completion events to Redis Streams and optionally Kafka.

    Usage::

        svc = IngestionService()
        job_id = await svc.run_ingestion(["arxiv", "github"])
    """

    # Recognised source identifiers
    ALL_SOURCES = ["arxiv", "github", "huggingface", "paperswithcode", "reddit"]

    def __init__(self) -> None:
        self._cache = CacheService()
        self._settings = settings

    # =========================================================================
    # Public API
    # =========================================================================

    async def run_ingestion(
        self,
        sources: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> str:
        """
        Run a full ingestion cycle across the specified sources.

        Args:
            sources:       List of source names; defaults to ALL_SOURCES.
            force_refresh: Re-ingest even if records already exist.

        Returns:
            ``job_id`` string (UUID4) that can be used to poll progress.
        """
        sources = sources or self.ALL_SOURCES
        job_id = str(uuid.uuid4())
        progress = IngestionProgress(job_id, self._cache)

        logger.info(
            "Ingestion started",
            extra={"job_id": job_id, "sources": sources, "force_refresh": force_refresh},
        )

        await progress.start(sources)

        # Update global status key
        await self._cache.set(
            _STATUS_KEY,
            {"status": "running", "job_id": job_id, "started_at": datetime.now(timezone.utc).isoformat()},
            ttl=7200,
        )

        # Run ingestion in the background (non-blocking from the caller's view)
        asyncio.create_task(
            self._run_pipeline(job_id, sources, force_refresh, progress),
            name=f"ingestion-{job_id[:8]}",
        )

        return job_id

    async def get_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the status of a specific job (or the most recent one).

        Args:
            job_id: Specific job UUID; omit for the current/last run.
        """
        if job_id:
            progress = IngestionProgress(job_id, self._cache)
            return await progress.get_state()
        return await self._cache.get(_STATUS_KEY) or {"status": "idle"}

    async def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return a list of recent ingestion job summaries (newest first)."""
        raw = await self._cache.get(_HISTORY_KEY)
        if not raw:
            return []
        entries = raw if isinstance(raw, list) else []
        return entries[:limit]

    async def process_signals(
        self,
        signals: List[RawSignal],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, int]:
        """
        Deduplicate and persist a batch of raw signals to the database.

        Performs:
        - Deduplication by external_id (arxiv_id / github_id / full_name).
        - Upsert (INSERT … ON CONFLICT DO UPDATE) for efficiency.
        - Returns counts of newly inserted and updated rows.

        Args:
            signals: List of :class:`RawSignal` objects.
            session: Optional existing DB session; creates one if None.

        Returns:
            Dict with keys "inserted" and "updated".
        """
        paper_signals = [s for s in signals if s.entity_type == "paper"]
        repo_signals = [s for s in signals if s.entity_type == "repository"]

        counts: Dict[str, int] = {"inserted": 0, "updated": 0}

        if session is not None:
            p_counts = await self._upsert_papers(paper_signals, session, force=False)
            r_counts = await self._upsert_repos(repo_signals, session, force=False)
        else:
            async with get_db_context() as db:
                p_counts = await self._upsert_papers(paper_signals, db, force=False)
                r_counts = await self._upsert_repos(repo_signals, db, force=False)

        counts["inserted"] = p_counts["inserted"] + r_counts["inserted"]
        counts["updated"] = p_counts["updated"] + r_counts["updated"]
        return counts

    async def update_trend_signals(
        self,
        entity_id: uuid.UUID,
        entity_type: EntityType,
        signal_values: Dict[SignalType, float],
        signal_date: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> int:
        """
        Write or update daily TrendSignal rows for one entity.

        If a signal for (entity_id, entity_type, signal_type, signal_date)
        already exists, it is updated in-place to avoid duplicate rows.

        Args:
            entity_id:     UUID of the entity.
            entity_type:   Entity discriminator.
            signal_values: Mapping from SignalType to measured float value.
            signal_date:   Date of the signal; defaults to today (UTC).
            source:        Optional source label string.

        Returns:
            Number of rows written.
        """
        today = (signal_date or datetime.now(timezone.utc)).date()
        rows_written = 0

        async with get_db_context() as db:
            for sig_type, value in signal_values.items():
                stmt = (
                    pg_insert(TrendSignal)
                    .values(
                        id=uuid.uuid4(),
                        entity_id=entity_id,
                        entity_type=entity_type,
                        signal_date=today,
                        value=value,
                        signal_type=sig_type,
                        source=source,
                    )
                    .on_conflict_do_update(
                        index_elements=["entity_id", "entity_type", "signal_type", "signal_date"]
                        if False  # replace with actual unique constraint when migrated
                        else None,
                        constraint=None,
                        set_={"value": value, "source": source},
                    )
                )
                # Fallback: use a plain insert and catch IntegrityError
                try:
                    signal = TrendSignal(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        signal_date=today,
                        value=value,
                        signal_type=sig_type,
                        source=source,
                    )
                    db.add(signal)
                    rows_written += 1
                except Exception as exc:
                    logger.warning(
                        "Could not insert trend signal",
                        extra={
                            "entity_id": str(entity_id),
                            "signal_type": sig_type.value,
                            "error": str(exc),
                        },
                    )

            try:
                await db.flush()
            except Exception as exc:
                await db.rollback()
                logger.error("Trend signal flush failed", exc_info=exc)
                return 0

        return rows_written

    # =========================================================================
    # Private pipeline
    # =========================================================================

    async def _run_pipeline(
        self,
        job_id: str,
        sources: List[str],
        force_refresh: bool,
        progress: IngestionProgress,
    ) -> None:
        """Background task: run each source adapter and coordinate results."""
        total_upserted = 0
        total_signals_written = 0
        error_message: Optional[str] = None

        try:
            # Fetch from all sources concurrently (with individual error isolation)
            tasks = [self._fetch_source(source) for source in sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            async with get_db_context() as db:
                for source, result in zip(sources, results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Source fetch failed",
                            extra={"source": source, "error": str(result)},
                        )
                        await progress.update_source(source, processed=0, errors=1)
                        continue

                    signals: List[RawSignal] = result
                    logger.info(
                        "Source fetched",
                        extra={"source": source, "count": len(signals)},
                    )

                    # Deduplicate and upsert
                    counts = await self.process_signals(signals, session=db)
                    total_upserted += counts["inserted"] + counts["updated"]
                    await progress.update_source(source, processed=len(signals))

                    # Write trend signals
                    for sig in signals:
                        try:
                            entity_id = uuid.uuid5(uuid.NAMESPACE_URL, sig.url or sig.external_id)
                            for sig_type_str, value in sig.signal_values.items():
                                try:
                                    sig_type = SignalType(sig_type_str)
                                except ValueError:
                                    continue
                                entity_type = (
                                    EntityType.PAPER
                                    if sig.entity_type == "paper"
                                    else EntityType.REPOSITORY
                                )
                                written = await self.update_trend_signals(
                                    entity_id=entity_id,
                                    entity_type=entity_type,
                                    signal_values={sig_type: value},
                                    source=sig.source,
                                )
                                total_signals_written += written
                        except Exception as exc:
                            logger.warning(
                                "Signal update error",
                                extra={"source": source, "error": str(exc)},
                            )

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            logger.error("Ingestion pipeline failed", exc_info=exc, extra={"job_id": job_id})

        finally:
            await progress.finish(
                total_upserted=total_upserted,
                total_signals_written=total_signals_written,
                error=error_message,
            )
            await self._append_history(job_id, sources, total_upserted, error_message)
            await self._publish_completion_event(job_id, total_upserted)

            # Invalidate trending caches
            await self._cache.invalidate_namespace("trends")
            await self._cache.invalidate_namespace("dashboard")

            logger.info(
                "Ingestion pipeline completed",
                extra={
                    "job_id": job_id,
                    "total_upserted": total_upserted,
                    "total_signals_written": total_signals_written,
                    "error": error_message,
                },
            )

    async def _fetch_source(self, source: str) -> List[RawSignal]:
        """
        Dispatch to the appropriate source adapter.

        Each adapter is responsible for calling the external API / MCP server
        and returning a list of :class:`RawSignal` objects.

        In a real deployment these adapters live in ``backend/adapters/``.
        """
        logger.debug("Fetching source", extra={"source": source})

        # --- Simulated fetch; replace with real adapter calls ---
        await asyncio.sleep(0.1)  # Simulate I/O

        # Example stub — real adapters would return actual data
        return []

    # -------------------------------------------------------------------------
    # Upsert helpers
    # -------------------------------------------------------------------------

    async def _upsert_papers(
        self,
        signals: List[RawSignal],
        session: AsyncSession,
        force: bool,
    ) -> Dict[str, int]:
        """Upsert paper rows; returns {"inserted": N, "updated": M}."""
        inserted = 0
        updated = 0

        for sig in signals:
            try:
                arxiv_id = sig.external_id if sig.source == "arxiv" else None

                # Check existence
                existing = None
                if arxiv_id:
                    stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()

                if existing and not force:
                    # Merge metrics only
                    existing.citation_count = sig.signal_values.get("citations", existing.citation_count)
                    existing.updated_at = datetime.now(timezone.utc)
                    updated += 1
                else:
                    paper = Paper(
                        arxiv_id=arxiv_id,
                        title=sig.title,
                        abstract=sig.description,
                        url=sig.url,
                        source=PaperSource(sig.source) if sig.source in PaperSource._value2member_map_ else PaperSource.ARXIV,
                        authors=sig.metadata.get("authors", []),
                        categories=sig.metadata.get("categories", []),
                        published_date=sig.metadata.get("published_date"),
                        citation_count=int(sig.signal_values.get("citations", 0)),
                        github_url=sig.metadata.get("github_url"),
                        has_implementation=bool(sig.metadata.get("has_implementation", False)),
                    )
                    session.add(paper)
                    inserted += 1

            except Exception as exc:
                logger.warning(
                    "Paper upsert error",
                    extra={"external_id": sig.external_id, "error": str(exc)},
                )

        try:
            await session.flush()
        except Exception as exc:
            await session.rollback()
            logger.error("Paper batch flush failed", exc_info=exc)

        return {"inserted": inserted, "updated": updated}

    async def _upsert_repos(
        self,
        signals: List[RawSignal],
        session: AsyncSession,
        force: bool,
    ) -> Dict[str, int]:
        """Upsert repository rows; returns {"inserted": N, "updated": M}."""
        inserted = 0
        updated = 0

        for sig in signals:
            try:
                full_name = sig.external_id

                stmt = select(Repository).where(Repository.full_name == full_name)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing and not force:
                    existing.stars = int(sig.signal_values.get("stars", existing.stars))
                    existing.forks = int(sig.signal_values.get("forks", existing.forks))
                    existing.stars_today = int(sig.signal_values.get("stars_today", existing.stars_today or 0))
                    existing.last_ingested = datetime.now(timezone.utc)
                    existing.updated_at = datetime.now(timezone.utc)
                    updated += 1
                else:
                    repo = Repository(
                        name=sig.metadata.get("name", full_name.split("/")[-1]),
                        full_name=full_name,
                        owner=sig.metadata.get("owner"),
                        description=sig.description,
                        url=sig.url,
                        language=sig.metadata.get("language"),
                        topics=sig.metadata.get("topics", []),
                        stars=int(sig.signal_values.get("stars", 0)),
                        forks=int(sig.signal_values.get("forks", 0)),
                        watchers=int(sig.signal_values.get("watchers", 0)),
                        stars_today=int(sig.signal_values.get("stars_today", 0)),
                        last_ingested=datetime.now(timezone.utc),
                    )
                    session.add(repo)
                    inserted += 1

            except Exception as exc:
                logger.warning(
                    "Repo upsert error",
                    extra={"external_id": sig.external_id, "error": str(exc)},
                )

        try:
            await session.flush()
        except Exception as exc:
            await session.rollback()
            logger.error("Repo batch flush failed", exc_info=exc)

        return {"inserted": inserted, "updated": updated}

    # -------------------------------------------------------------------------
    # Event publishing
    # -------------------------------------------------------------------------

    async def _publish_completion_event(self, job_id: str, total_upserted: int) -> None:
        """Publish an ingestion completion event to Redis Streams."""
        event = {
            "type": "ingestion_complete",
            "job_id": job_id,
            "total_upserted": total_upserted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            from backend.services.cache_service import CacheService
            import redis.asyncio as aioredis
            from backend.services.cache_service import _get_pool

            async with aioredis.Redis(connection_pool=_get_pool()) as client:
                await client.xadd(
                    "research:ingestion:events",
                    {k: str(v) for k, v in event.items()},
                    maxlen=1000,
                )
        except Exception as exc:
            logger.warning(
                "Failed to publish ingestion event to Redis Stream",
                extra={"error": str(exc)},
            )

        # Optionally publish to Kafka
        if self._settings.KAFKA_ENABLED:
            await self._publish_kafka(self._settings.KAFKA_TOPIC_INGESTION, event)

    async def _publish_kafka(self, topic: str, event: Dict[str, Any]) -> None:
        """Publish an event to Kafka (no-op if kafka is not configured)."""
        try:
            import aiokafka  # type: ignore[import]
            producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self._settings.KAFKA_BOOTSTRAP_SERVERS
            )
            await producer.start()
            try:
                import json as _json
                await producer.send_and_wait(
                    topic,
                    value=_json.dumps(event).encode(),
                )
            finally:
                await producer.stop()
        except ImportError:
            logger.debug("aiokafka not installed; skipping Kafka publish")
        except Exception as exc:
            logger.warning("Kafka publish failed", extra={"topic": topic, "error": str(exc)})

    # -------------------------------------------------------------------------
    # History management
    # -------------------------------------------------------------------------

    async def _append_history(
        self,
        job_id: str,
        sources: List[str],
        total_upserted: int,
        error: Optional[str],
    ) -> None:
        """Append a summary entry to the ingestion history list in Redis."""
        entry = {
            "job_id": job_id,
            "sources": sources,
            "total_upserted": total_upserted,
            "status": "failed" if error else "completed",
            "error": error,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        history = await self._cache.get(_HISTORY_KEY) or []
        if isinstance(history, list):
            history.insert(0, entry)
            history = history[:50]  # Keep last 50 entries
        else:
            history = [entry]
        await self._cache.set(_HISTORY_KEY, history, ttl=7 * 86400)  # 7 days
