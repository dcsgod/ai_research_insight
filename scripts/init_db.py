"""
init_db.py — Database Initialization Script
===========================================
Creates all PostgreSQL tables via Alembic migrations, initializes
Qdrant vector collections, and seeds the database with sample data.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --skip-migrations
    python scripts/init_db.py --skip-seed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports work when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncpg
import structlog
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    OptimizersConfigDiff,
    HnswConfigDiff,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from alembic.config import Config as AlembicConfig
from alembic import command as alembic_command

# ---------------------------------------------------------------------------
# Bootstrap environment & logging
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("init_db")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://ai_user:ai_secret@localhost:5432/ai_research",
)
QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))

# Qdrant collection names and their embedding dimensions
QDRANT_COLLECTIONS: dict[str, dict[str, Any]] = {
    "papers": {
        "size": 768,
        "distance": Distance.COSINE,
        "description": "Semantic embeddings of research papers (title + abstract)",
    },
    "repositories": {
        "size": 768,
        "distance": Distance.COSINE,
        "description": "Semantic embeddings of GitHub repositories (name + description + README)",
    },
    "models": {
        "size": 768,
        "distance": Distance.COSINE,
        "description": "Semantic embeddings of HuggingFace models",
    },
    "topics": {
        "size": 768,
        "distance": Distance.COSINE,
        "description": "Topic cluster centroid vectors",
    },
}

# ---------------------------------------------------------------------------
# Sample seed data
# ---------------------------------------------------------------------------
SAMPLE_PAPERS: list[dict[str, Any]] = [
    {
        "id": str(uuid4()),
        "arxiv_id": "2310.06825",
        "title": "Mistral 7B",
        "authors": json.dumps(["Albert Q. Jiang", "Alexandre Sablayrolles", "Arthur Mensch"]),
        "abstract": (
            "We introduce Mistral 7B, a language model with 7 billion parameters. "
            "Mistral 7B outperforms Llama 2 13B on all benchmarks, and Llama 1 34B on "
            "many benchmarks. It uses Grouped-Query Attention (GQA) for fast inference, "
            "coupled with Sliding Window Attention (SWA) to handle sequences of arbitrary length."
        ),
        "categories": json.dumps(["cs.CL", "cs.AI"]),
        "published_at": datetime(2023, 10, 10, tzinfo=timezone.utc).isoformat(),
        "citation_count": 4200,
        "github_url": "https://github.com/mistralai/mistral-src",
        "pdf_url": "https://arxiv.org/pdf/2310.06825",
        "trend_score": 0.92,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "arxiv_id": "2307.09288",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": json.dumps(["Hugo Touvron", "Louis Martin", "Kevin Stone"]),
        "abstract": (
            "We develop and release Llama 2, a collection of pretrained and fine-tuned "
            "large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. "
            "Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases."
        ),
        "categories": json.dumps(["cs.CL", "cs.AI"]),
        "published_at": datetime(2023, 7, 18, tzinfo=timezone.utc).isoformat(),
        "citation_count": 12800,
        "github_url": "https://github.com/facebookresearch/llama",
        "pdf_url": "https://arxiv.org/pdf/2307.09288",
        "trend_score": 0.88,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "arxiv_id": "2303.08774",
        "title": "GPT-4 Technical Report",
        "authors": json.dumps(["OpenAI"]),
        "abstract": (
            "We report the development of GPT-4, a large-scale, multimodal model which can "
            "accept image and text inputs and produce text outputs. While less capable than "
            "humans in many real-world scenarios, GPT-4 exhibits human-level performance on "
            "various professional and academic benchmarks."
        ),
        "categories": json.dumps(["cs.CL", "cs.CV"]),
        "published_at": datetime(2023, 3, 15, tzinfo=timezone.utc).isoformat(),
        "citation_count": 18500,
        "github_url": None,
        "pdf_url": "https://arxiv.org/pdf/2303.08774",
        "trend_score": 0.95,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "arxiv_id": "2312.11805",
        "title": "Gemini: A Family of Highly Capable Multimodal Models",
        "authors": json.dumps(["Gemini Team", "Google"]),
        "abstract": (
            "This report introduces a new family of multimodal models, Gemini, that demonstrate "
            "remarkable capabilities across image, audio, video, and text understanding. Gemini "
            "Ultra achieves human-expert performance on MMLU, and state-of-the-art results on "
            "30 of 32 widely-used academic benchmarks."
        ),
        "categories": json.dumps(["cs.CL", "cs.AI", "cs.CV"]),
        "published_at": datetime(2023, 12, 19, tzinfo=timezone.utc).isoformat(),
        "citation_count": 5600,
        "github_url": None,
        "pdf_url": "https://arxiv.org/pdf/2312.11805",
        "trend_score": 0.90,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "arxiv_id": "2401.10020",
        "title": "Mixtral of Experts",
        "authors": json.dumps(["Albert Q. Jiang", "Alexandre Sablayrolles"]),
        "abstract": (
            "We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. "
            "Mixtral has the same architecture as Mistral 7B, with the difference that each "
            "layer is composed of 8 feedforward blocks (experts). At each layer, for each token, "
            "a router network selects two experts to process the current state."
        ),
        "categories": json.dumps(["cs.CL", "cs.AI"]),
        "published_at": datetime(2024, 1, 8, tzinfo=timezone.utc).isoformat(),
        "citation_count": 2100,
        "github_url": "https://github.com/mistralai/mistral-src",
        "pdf_url": "https://arxiv.org/pdf/2401.10020",
        "trend_score": 0.85,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
]

SAMPLE_REPOS: list[dict[str, Any]] = [
    {
        "id": str(uuid4()),
        "github_id": "527692701",
        "name": "transformers",
        "full_name": "huggingface/transformers",
        "description": "🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.",
        "language": "Python",
        "stars": 132000,
        "forks": 26000,
        "topics": json.dumps(["nlp", "deep-learning", "transformers", "pytorch", "tensorflow"]),
        "license": "Apache-2.0",
        "homepage": "https://huggingface.co/docs/transformers",
        "trend_score": 0.97,
        "weekly_star_growth": 850,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "github_id": "578047369",
        "name": "ollama",
        "full_name": "ollama/ollama",
        "description": "Get up and running with Llama 3.1, Mistral, Gemma 2, and other large language models.",
        "language": "Go",
        "stars": 89000,
        "forks": 7200,
        "topics": json.dumps(["llm", "llama", "mistral", "local-ai"]),
        "license": "MIT",
        "homepage": "https://ollama.com",
        "trend_score": 0.94,
        "weekly_star_growth": 1200,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "github_id": "614977791",
        "name": "langchain",
        "full_name": "langchain-ai/langchain",
        "description": "🦜🔗 Build context-aware reasoning applications",
        "language": "Python",
        "stars": 92000,
        "forks": 14800,
        "topics": json.dumps(["llm", "ai", "agents", "rag"]),
        "license": "MIT",
        "homepage": "https://langchain.com",
        "trend_score": 0.91,
        "weekly_star_growth": 600,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "github_id": "652624030",
        "name": "vllm",
        "full_name": "vllm-project/vllm",
        "description": "A high-throughput and memory-efficient inference and serving engine for LLMs",
        "language": "Python",
        "stars": 27000,
        "forks": 3900,
        "topics": json.dumps(["llm", "inference", "serving", "gpu"]),
        "license": "Apache-2.0",
        "homepage": "https://vllm.ai",
        "trend_score": 0.93,
        "weekly_star_growth": 750,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "github_id": "670450741",
        "name": "instructor",
        "full_name": "jxnl/instructor",
        "description": "Structured outputs for LLMs, supports many providers.",
        "language": "Python",
        "stars": 8200,
        "forks": 620,
        "topics": json.dumps(["llm", "pydantic", "structured-outputs", "openai"]),
        "license": "MIT",
        "homepage": "https://python.useinstructor.com",
        "trend_score": 0.82,
        "weekly_star_growth": 380,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
]

SAMPLE_TOPICS: list[dict[str, Any]] = [
    {
        "id": str(uuid4()),
        "name": "Large Language Models",
        "slug": "large-language-models",
        "description": "Research and development of large-scale language models including GPT, LLaMA, and Mistral families.",
        "keywords": json.dumps(["llm", "gpt", "transformer", "pretraining", "instruction-tuning", "rlhf"]),
        "trend_score": 0.98,
        "paper_count": 4820,
        "repo_count": 12400,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "name": "Retrieval-Augmented Generation",
        "slug": "retrieval-augmented-generation",
        "description": "Combining retrieval systems with generative models to ground outputs in external knowledge.",
        "keywords": json.dumps(["rag", "retrieval", "vector-search", "knowledge-base", "grounding"]),
        "trend_score": 0.93,
        "paper_count": 1250,
        "repo_count": 3800,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "id": str(uuid4()),
        "name": "Multimodal AI",
        "slug": "multimodal-ai",
        "description": "AI systems that process and generate multiple modalities including text, images, audio, and video.",
        "keywords": json.dumps(["multimodal", "vision-language", "image-text", "video-llm", "audio-llm"]),
        "trend_score": 0.90,
        "paper_count": 980,
        "repo_count": 2100,
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
]


# ---------------------------------------------------------------------------
# Alembic migrations
# ---------------------------------------------------------------------------
async def run_migrations(skip: bool = False) -> None:
    """Apply Alembic migrations to upgrade the database schema to HEAD."""
    if skip:
        logger.info("Skipping Alembic migrations (--skip-migrations flag)")
        return

    alembic_ini = PROJECT_ROOT / "alembic.ini"
    if not alembic_ini.exists():
        logger.warning(
            "alembic.ini not found — skipping migrations",
            path=str(alembic_ini),
        )
        return

    logger.info("Running Alembic migrations...")
    alembic_cfg = AlembicConfig(str(alembic_ini))
    alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)

    # Run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: alembic_command.upgrade(alembic_cfg, "head")
    )
    logger.info("Migrations applied successfully")


# ---------------------------------------------------------------------------
# Qdrant collection setup
# ---------------------------------------------------------------------------
def init_qdrant_collections(client: QdrantClient) -> None:
    """Create all required Qdrant vector collections if they don't exist."""
    logger.info("Initializing Qdrant collections...")

    existing = {c.name for c in client.get_collections().collections}

    for collection_name, config in QDRANT_COLLECTIONS.items():
        if collection_name in existing:
            logger.info("Collection already exists, skipping", collection=collection_name)
            continue

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config["size"],
                distance=config["distance"],
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )
        logger.info(
            "Created Qdrant collection",
            collection=collection_name,
            size=config["size"],
            distance=str(config["distance"]),
        )

    logger.info("Qdrant collections initialized", total=len(QDRANT_COLLECTIONS))


# ---------------------------------------------------------------------------
# PostgreSQL seed data
# ---------------------------------------------------------------------------
async def seed_database(engine: Any, skip: bool = False) -> None:
    """Insert sample data into the database for development/testing."""
    if skip:
        logger.info("Skipping seed data (--skip-seed flag)")
        return

    logger.info("Seeding database with sample data...")

    # Use raw asyncpg for seeding — faster and simpler than ORM for bulk inserts
    dsn = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn: asyncpg.Connection = await asyncpg.connect(dsn)

    try:
        # --- Papers ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                arxiv_id TEXT UNIQUE,
                title TEXT NOT NULL,
                authors JSONB,
                abstract TEXT,
                categories JSONB,
                published_at TIMESTAMPTZ,
                citation_count INTEGER DEFAULT 0,
                github_url TEXT,
                pdf_url TEXT,
                trend_score FLOAT DEFAULT 0.0,
                embedding_id TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        paper_count = 0
        for paper in SAMPLE_PAPERS:
            existing = await conn.fetchrow(
                "SELECT id FROM papers WHERE arxiv_id = $1", paper["arxiv_id"]
            )
            if existing:
                logger.debug("Paper already seeded", arxiv_id=paper["arxiv_id"])
                continue
            await conn.execute(
                """
                INSERT INTO papers (id, arxiv_id, title, authors, abstract, categories,
                    published_at, citation_count, github_url, pdf_url, trend_score, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                """,
                paper["id"],
                paper["arxiv_id"],
                paper["title"],
                paper["authors"],
                paper["abstract"],
                paper["categories"],
                datetime.fromisoformat(paper["published_at"]),
                paper["citation_count"],
                paper.get("github_url"),
                paper.get("pdf_url"),
                paper["trend_score"],
                datetime.fromisoformat(paper["created_at"]),
            )
            paper_count += 1

        logger.info("Seeded papers", count=paper_count)

        # --- Repositories ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS repositories (
                id TEXT PRIMARY KEY,
                github_id TEXT UNIQUE,
                name TEXT NOT NULL,
                full_name TEXT UNIQUE,
                description TEXT,
                language TEXT,
                stars INTEGER DEFAULT 0,
                forks INTEGER DEFAULT 0,
                topics JSONB,
                license TEXT,
                homepage TEXT,
                trend_score FLOAT DEFAULT 0.0,
                weekly_star_growth INTEGER DEFAULT 0,
                embedding_id TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        repo_count = 0
        for repo in SAMPLE_REPOS:
            existing = await conn.fetchrow(
                "SELECT id FROM repositories WHERE github_id = $1", repo["github_id"]
            )
            if existing:
                logger.debug("Repo already seeded", full_name=repo["full_name"])
                continue
            await conn.execute(
                """
                INSERT INTO repositories (id, github_id, name, full_name, description,
                    language, stars, forks, topics, license, homepage, trend_score,
                    weekly_star_growth, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                """,
                repo["id"],
                repo["github_id"],
                repo["name"],
                repo["full_name"],
                repo["description"],
                repo["language"],
                repo["stars"],
                repo["forks"],
                repo["topics"],
                repo.get("license"),
                repo.get("homepage"),
                repo["trend_score"],
                repo["weekly_star_growth"],
                datetime.fromisoformat(repo["created_at"]),
            )
            repo_count += 1

        logger.info("Seeded repositories", count=repo_count)

        # --- Topics ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                description TEXT,
                keywords JSONB,
                trend_score FLOAT DEFAULT 0.0,
                paper_count INTEGER DEFAULT 0,
                repo_count INTEGER DEFAULT 0,
                embedding_id TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        topic_count = 0
        for topic in SAMPLE_TOPICS:
            existing = await conn.fetchrow(
                "SELECT id FROM topics WHERE slug = $1", topic["slug"]
            )
            if existing:
                logger.debug("Topic already seeded", slug=topic["slug"])
                continue
            await conn.execute(
                """
                INSERT INTO topics (id, name, slug, description, keywords,
                    trend_score, paper_count, repo_count, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """,
                topic["id"],
                topic["name"],
                topic["slug"],
                topic["description"],
                topic["keywords"],
                topic["trend_score"],
                topic["paper_count"],
                topic["repo_count"],
                datetime.fromisoformat(topic["created_at"]),
            )
            topic_count += 1

        logger.info("Seeded topics", count=topic_count)

    finally:
        await conn.close()

    logger.info(
        "Database seeding complete",
        papers=len(SAMPLE_PAPERS),
        repos=len(SAMPLE_REPOS),
        topics=len(SAMPLE_TOPICS),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main(args: argparse.Namespace) -> None:
    """Main initialization flow."""
    logger.info(
        "Starting AI Research Platform database initialization",
        database_url=DATABASE_URL.split("@")[-1],  # Omit credentials in logs
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
    )

    # 1. Run Alembic migrations
    await run_migrations(skip=args.skip_migrations)

    # 2. Initialize Qdrant collections (synchronous client is fine here)
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
        init_qdrant_collections(qdrant_client)
    except Exception as exc:
        logger.error(
            "Failed to connect to Qdrant — skipping collection init",
            error=str(exc),
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )

    # 3. Create SQLAlchemy async engine (needed for ORM table creation if no Alembic)
    engine = create_async_engine(DATABASE_URL, echo=False)
    try:
        # Import models so that Base.metadata knows about them
        try:
            from backend.models import Base  # type: ignore[import]
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("SQLAlchemy models created (or already exist)")
        except ImportError:
            logger.warning(
                "backend.models not importable — skipping SQLAlchemy table creation"
            )

        # 4. Seed sample data
        await seed_database(engine, skip=args.skip_seed)

    finally:
        await engine.dispose()

    logger.info("✅ Database initialization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize the AI Research Platform database and vector store."
    )
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Skip running Alembic migrations",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Skip inserting sample seed data",
    )
    parsed = parser.parse_args()
    asyncio.run(main(parsed))
