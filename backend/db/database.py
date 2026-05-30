"""
AI Research Intelligence Platform — Database Configuration
Async SQLAlchemy setup with PostgreSQL via asyncpg driver.
"""
import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# ─── Engine ─────────────────────────────────────────────────────────────────
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,          # Test connections before use
    pool_recycle=3600,            # Recycle connections every hour
    pool_timeout=30,
    connect_args={
        "statement_cache_size": 0,  # Disable prepared statement caching for pgbouncer compatibility
    },
)

# ─── Session Factory ─────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ─── Dependency ──────────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─── Table Management ────────────────────────────────────────────────────────
async def create_tables() -> None:
    """
    Create all database tables defined in ORM models.
    Imports models to ensure they are registered with Base.metadata.
    """
    # Import all models to register them with Base.metadata
    from backend.models import paper, repository, topic, trend_signal, insight  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created successfully")


async def drop_tables() -> None:
    """Drop all database tables. USE WITH CAUTION — data will be lost."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        logger.warning("⚠️  All database tables dropped")


async def check_db_health() -> bool:
    """
    Check if the database connection is healthy.
    Returns True if connected, False otherwise.
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
