"""
AI Research Intelligence Platform — FastAPI Main Application
Production-grade FastAPI app with full middleware, routing, and lifecycle management.
"""
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from backend.core.config import get_settings
from backend.db.database import create_tables, engine
from backend.api import trends, papers, repos, forecasts, ingestion, topics, insights
from backend.api.websocket import router as ws_router

# Configure module-level logger
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup initialization and graceful shutdown.
    """
    # ─── Startup ───────────────────────────────────────────────────────────
    logger.info("🚀 Starting AI Research Intelligence Platform...")

    # Initialize database tables
    try:
        await create_tables()
        logger.info("✅ Database tables initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    # Initialize Qdrant collections
    try:
        from vector_db.qdrant_client import QdrantService
        qdrant = QdrantService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        await qdrant.setup_collections()
        logger.info("✅ Qdrant vector collections initialized")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant initialization failed (non-fatal): {e}")

    # Initialize Redis connection
    try:
        from backend.services.cache_service import CacheService
        cache = CacheService(redis_url=settings.redis_url)
        await cache.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.warning(f"⚠️  Redis initialization failed (non-fatal): {e}")

    logger.info(f"🌐 API running at http://{settings.app_host}:{settings.app_port}")
    logger.info(f"📚 API docs at http://{settings.app_host}:{settings.app_port}/docs")

    yield  # Application runs here

    # ─── Shutdown ──────────────────────────────────────────────────────────
    logger.info("🔄 Shutting down AI Research Intelligence Platform...")

    # Close database connections
    await engine.dispose()
    logger.info("✅ Database connections closed")

    logger.info("👋 Shutdown complete")


# ─── FastAPI Application ────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Intelligence Platform API",
    description="""
    ## AI Research Intelligence Platform

    A Bloomberg Terminal for AI Research Trends. This API powers:

    - **Trend Discovery** — Real-time ranked AI research trends
    - **Paper Intelligence** — arXiv, HuggingFace, PapersWithCode aggregation
    - **GitHub Tracking** — Trending AI repositories with momentum signals
    - **Forecasting** — Prophet-powered trend momentum forecasting
    - **Topic Extraction** — BERTopic-powered emerging topic discovery
    - **AI Insights** — LLM-generated contextual summaries
    - **Live Updates** — WebSocket real-time data stream
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ─── Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all HTTP requests with timing and correlation ID."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Add request ID to state for downstream use
    request.state.request_id = request_id

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.1f}ms)"
    )

    # Add headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"

    return response


# ─── Exception Handlers ─────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions gracefully."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "An internal server error occurred",
            "path": str(request.url.path),
        },
    )


# ─── Routers ────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(trends.router, prefix=API_PREFIX, tags=["Trends"])
app.include_router(papers.router, prefix=API_PREFIX, tags=["Papers"])
app.include_router(repos.router, prefix=API_PREFIX, tags=["Repositories"])
app.include_router(forecasts.router, prefix=API_PREFIX, tags=["Forecasts"])
app.include_router(ingestion.router, prefix=API_PREFIX, tags=["Ingestion"])
app.include_router(topics.router, prefix=API_PREFIX, tags=["Topics"])
app.include_router(insights.router, prefix=API_PREFIX, tags=["Insights"])
app.include_router(ws_router, tags=["WebSocket"])


# ─── Core Endpoints ─────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "description": "Bloomberg Terminal for AI Research Trends",
        "docs": "/docs",
        "health": "/health",
        "api": {
            "trends": f"{API_PREFIX}/trends",
            "papers": f"{API_PREFIX}/papers",
            "repos": f"{API_PREFIX}/repos",
            "forecasts": f"{API_PREFIX}/forecasts",
            "topics": f"{API_PREFIX}/topics",
            "insights": f"{API_PREFIX}/insights",
            "ingestion": f"{API_PREFIX}/ingestion",
        },
        "websocket": "/ws/live-feed",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns status of all critical dependencies.
    """
    health = {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.app_env,
        "services": {},
    }

    # Check PostgreSQL
    try:
        from backend.db.database import check_db_health
        db_ok = await check_db_health()
        health["services"]["postgres"] = "healthy" if db_ok else "unhealthy"
    except Exception as e:
        health["services"]["postgres"] = f"error: {str(e)}"
        health["status"] = "degraded"

    # Check Redis
    try:
        from backend.services.cache_service import CacheService
        cache = CacheService(redis_url=settings.redis_url)
        redis_ok = await cache.ping()
        health["services"]["redis"] = "healthy" if redis_ok else "unhealthy"
    except Exception as e:
        health["services"]["redis"] = f"error: {str(e)}"
        health["status"] = "degraded"

    # Check Qdrant
    try:
        from vector_db.qdrant_client import QdrantService
        qdrant = QdrantService(host=settings.qdrant_host, port=settings.qdrant_port)
        qdrant_ok = await qdrant.health_check()
        health["services"]["qdrant"] = "healthy" if qdrant_ok else "unhealthy"
    except Exception as e:
        health["services"]["qdrant"] = f"error: {str(e)}"

    return health
