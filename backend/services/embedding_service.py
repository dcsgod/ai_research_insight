"""
embedding_service.py — Embedding API Service
=============================================
FastAPI service layer that wraps the ml_pipeline embedding module,
providing async embedding generation and vector similarity search
via Qdrant.

Responsibilities:
- Encode text (title + abstract / name + description) via SentenceTransformer
- Store embeddings in Qdrant with entity metadata as payload
- Retrieve similar items using vector search
- Expose semantic search across any collection
- Delegate heavy encoding to ml_pipeline.EmbeddingService

Collections:
- "papers"       → Paper title + abstract embeddings
- "repositories" → Repo name + description + topics embeddings
- "topics"       → Topic name + description embeddings
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger("embedding_service")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CollectionName = Literal["papers", "repositories", "models", "topics"]

# ---------------------------------------------------------------------------
# Qdrant collection → embedding dimension mapping
# ---------------------------------------------------------------------------
COLLECTION_DIMS: dict[str, int] = {
    "papers": 768,
    "repositories": 768,
    "models": 768,
    "topics": 768,
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Paper(BaseModel):
    """Minimal Paper representation needed for embedding."""

    id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    arxiv_id: str | None = None
    published_at: datetime | None = None
    citation_count: int = 0
    trend_score: float = 0.0


class Repository(BaseModel):
    """Minimal Repository representation needed for embedding."""

    id: str
    name: str
    full_name: str
    description: str | None = None
    language: str | None = None
    stars: int = 0
    topics: list[str] = Field(default_factory=list)
    homepage: str | None = None


class SearchResult(BaseModel):
    """A single result from a vector similarity search."""

    id: str
    collection: str
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Qdrant point payload (entity metadata)",
    )
    embedding_id: str | None = None


# ---------------------------------------------------------------------------
# EmbeddingAPIService
# ---------------------------------------------------------------------------
class EmbeddingAPIService:
    """
    Service class that bridges the FastAPI layer with ml_pipeline.EmbeddingService
    and the Qdrant vector database.

    Constructor dependencies (injected):
    - qdrant_client: qdrant_client.QdrantClient (or async variant)
    - model_name:    SentenceTransformer model identifier

    All public methods are async — heavy CPU encoding is run in an executor
    to avoid blocking the event loop.
    """

    DEFAULT_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    DEFAULT_LIMIT: int = 10
    MAX_LIMIT: int = 100

    def __init__(
        self,
        qdrant_client: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        self._qdrant = qdrant_client
        self._model_name = model_name or self.DEFAULT_MODEL
        self._log = structlog.get_logger("embedding_service")

        # Lazy-loaded inner ML pipeline service
        self._ml_service: Any | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    async def embed_paper(self, paper: Paper) -> str:
        """
        Encode a research paper and upsert its vector into the 'papers' collection.

        The embedding text is constructed as:
            "{title}. {abstract}"

        Args:
            paper: Paper model with at least id and title fields populated.

        Returns:
            embedding_id (str) — UUID assigned to this Qdrant point.
        """
        text = self._build_paper_text(paper)
        embedding_id = self._deterministic_embedding_id("paper", paper.id)

        self._log.debug(
            "Embedding paper",
            paper_id=paper.id,
            embedding_id=embedding_id,
            text_chars=len(text),
        )

        vector = await self._encode_text(text)

        payload = {
            "entity_type": "paper",
            "entity_id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "arxiv_id": paper.arxiv_id,
            "citation_count": paper.citation_count,
            "trend_score": paper.trend_score,
            "published_at": paper.published_at.isoformat() if paper.published_at else None,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
        }

        await self._upsert_vector(
            collection="papers",
            embedding_id=embedding_id,
            vector=vector,
            payload=payload,
        )

        return embedding_id

    async def embed_repo(self, repo: Repository) -> str:
        """
        Encode a repository and upsert its vector into the 'repositories' collection.

        The embedding text is constructed as:
            "{full_name}: {description}. Topics: {topics}"

        Args:
            repo: Repository model with at least id, full_name fields populated.

        Returns:
            embedding_id (str) — UUID assigned to this Qdrant point.
        """
        text = self._build_repo_text(repo)
        embedding_id = self._deterministic_embedding_id("repo", repo.id)

        self._log.debug(
            "Embedding repository",
            repo_id=repo.id,
            embedding_id=embedding_id,
            text_chars=len(text),
        )

        vector = await self._encode_text(text)

        payload = {
            "entity_type": "repository",
            "entity_id": repo.id,
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description,
            "language": repo.language,
            "stars": repo.stars,
            "topics": repo.topics,
            "homepage": repo.homepage,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
        }

        await self._upsert_vector(
            collection="repositories",
            embedding_id=embedding_id,
            vector=vector,
            payload=payload,
        )

        return embedding_id

    async def find_similar_papers(
        self,
        paper_id: str,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> list[Paper]:
        """
        Find papers semantically similar to a given paper using vector search.

        Looks up the source paper's vector in Qdrant and returns the nearest
        neighbors (excluding the source paper itself).

        Args:
            paper_id:        ID of the source paper.
            limit:           Maximum number of similar papers to return.
            score_threshold: Minimum cosine similarity threshold (0-1).

        Returns:
            List of Paper models sorted by similarity score (descending).
        """
        limit = max(1, min(limit, self.MAX_LIMIT))
        embedding_id = self._deterministic_embedding_id("paper", paper_id)

        self._log.info(
            "Finding similar papers",
            paper_id=paper_id,
            limit=limit,
            threshold=score_threshold,
        )

        results = await self._search_by_id(
            collection="papers",
            embedding_id=embedding_id,
            limit=limit + 1,  # +1 to exclude the query paper
            score_threshold=score_threshold,
        )

        papers: list[Paper] = []
        for result in results:
            if result.payload.get("entity_id") == paper_id:
                continue  # Skip the source paper
            papers.append(
                Paper(
                    id=result.payload.get("entity_id", result.id),
                    title=result.payload.get("title", "Unknown"),
                    abstract=result.payload.get("abstract"),
                    authors=result.payload.get("authors", []),
                    categories=result.payload.get("categories", []),
                    arxiv_id=result.payload.get("arxiv_id"),
                    citation_count=result.payload.get("citation_count", 0),
                    trend_score=result.payload.get("trend_score", 0.0),
                )
            )
            if len(papers) >= limit:
                break

        self._log.info(
            "Similar papers found",
            paper_id=paper_id,
            count=len(papers),
        )
        return papers

    async def semantic_search(
        self,
        query: str,
        collection: CollectionName = "papers",
        limit: int = 10,
        score_threshold: float = 0.3,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform a free-text semantic search across a Qdrant collection.

        Encodes the query string and performs an approximate nearest-neighbor
        search in the specified collection.

        Args:
            query:           Natural-language search query.
            collection:      Target Qdrant collection name.
            limit:           Maximum number of results.
            score_threshold: Minimum similarity score.
            filters:         Optional Qdrant filter dict for metadata filtering.

        Returns:
            List of SearchResult sorted by similarity score (descending).
        """
        if not query.strip():
            return []

        limit = max(1, min(limit, self.MAX_LIMIT))

        self._log.info(
            "Semantic search",
            query=query[:100],
            collection=collection,
            limit=limit,
        )

        vector = await self._encode_text(query)
        raw_results = await self._search_by_vector(
            collection=collection,
            vector=vector,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters,
        )

        results = [
            SearchResult(
                id=str(r.id),
                collection=collection,
                score=round(float(r.score), 4),
                payload=r.payload or {},
                embedding_id=str(r.id),
            )
            for r in raw_results
        ]

        self._log.info(
            "Semantic search complete",
            collection=collection,
            query=query[:50],
            results_found=len(results),
        )
        return results

    async def embed_batch_papers(self, papers: list[Paper]) -> list[str]:
        """
        Encode a batch of papers in one pass (more efficient than one-by-one).

        Args:
            papers: List of Paper models to embed.

        Returns:
            List of embedding IDs in the same order as input.
        """
        texts = [self._build_paper_text(p) for p in papers]
        vectors = await self._encode_texts_batch(texts)

        embedding_ids: list[str] = []
        for paper, vector in zip(papers, vectors):
            embedding_id = self._deterministic_embedding_id("paper", paper.id)
            payload = {
                "entity_type": "paper",
                "entity_id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "categories": paper.categories,
                "arxiv_id": paper.arxiv_id,
                "citation_count": paper.citation_count,
                "trend_score": paper.trend_score,
                "embedded_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._upsert_vector(
                collection="papers",
                embedding_id=embedding_id,
                vector=vector,
                payload=payload,
            )
            embedding_ids.append(embedding_id)

        self._log.info("Batch paper embedding complete", count=len(embedding_ids))
        return embedding_ids

    # -----------------------------------------------------------------------
    # Private: Encoding
    # -----------------------------------------------------------------------
    async def _get_ml_service(self) -> Any:
        """Lazy-initialize and return the inner ml_pipeline.EmbeddingService."""
        if self._ml_service is None:
            try:
                from ml_pipeline.embedding_service import EmbeddingService  # type: ignore[import]

                self._ml_service = EmbeddingService(model_name=self._model_name)
                await self._ml_service.initialize()
            except ImportError:
                self._log.warning(
                    "ml_pipeline.EmbeddingService not available — using stub encoder"
                )
                self._ml_service = _StubEmbeddingService(self._model_name)
        return self._ml_service

    async def _encode_text(self, text: str) -> list[float]:
        """Encode a single text string to a vector."""
        import asyncio

        svc = await self._get_ml_service()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, svc.encode_single, text)

    async def _encode_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of text strings to vectors."""
        import asyncio

        svc = await self._get_ml_service()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, svc.encode_batch, texts)

    # -----------------------------------------------------------------------
    # Private: Qdrant operations
    # -----------------------------------------------------------------------
    async def _upsert_vector(
        self,
        collection: str,
        embedding_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Upsert a single vector point into Qdrant."""
        if self._qdrant is None:
            self._log.debug(
                "No Qdrant client — skipping vector upsert",
                embedding_id=embedding_id,
            )
            return

        try:
            from qdrant_client.models import PointStruct  # type: ignore[import]

            self._qdrant.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=embedding_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            self._log.error(
                "Failed to upsert vector into Qdrant",
                embedding_id=embedding_id,
                collection=collection,
                error=str(exc),
            )
            raise

    async def _search_by_vector(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Search Qdrant by vector."""
        if self._qdrant is None:
            self._log.debug("No Qdrant client — returning empty search results")
            return []

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            def _sync_search() -> list[Any]:
                from qdrant_client.models import Filter  # type: ignore[import]

                qdrant_filter = None
                if filters:
                    # Build a simple must-match filter from dict
                    from qdrant_client.models import FieldCondition, MatchValue, Filter as QFilter

                    conditions = [
                        FieldCondition(key=k, match=MatchValue(value=v))
                        for k, v in filters.items()
                    ]
                    qdrant_filter = QFilter(must=conditions)

                return self._qdrant.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )

            return await loop.run_in_executor(None, _sync_search)
        except Exception as exc:
            self._log.error("Qdrant search failed", collection=collection, error=str(exc))
            return []

    async def _search_by_id(
        self,
        collection: str,
        embedding_id: str,
        limit: int,
        score_threshold: float,
    ) -> list[SearchResult]:
        """Search Qdrant using an existing point ID as the query vector."""
        if self._qdrant is None:
            return []

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            def _sync_retrieve_and_search() -> list[Any]:
                # First, retrieve the point's vector
                points = self._qdrant.retrieve(
                    collection_name=collection,
                    ids=[embedding_id],
                    with_vectors=True,
                )
                if not points:
                    return []

                vector = points[0].vector
                return self._qdrant.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                )

            raw_results = await loop.run_in_executor(None, _sync_retrieve_and_search)

            return [
                SearchResult(
                    id=str(r.id),
                    collection=collection,
                    score=round(float(r.score), 4),
                    payload=r.payload or {},
                    embedding_id=str(r.id),
                )
                for r in raw_results
            ]
        except Exception as exc:
            self._log.error(
                "Qdrant ID-based search failed",
                collection=collection,
                embedding_id=embedding_id,
                error=str(exc),
            )
            return []

    # -----------------------------------------------------------------------
    # Private: Text builders
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_paper_text(paper: Paper) -> str:
        """Build the text representation of a paper for embedding."""
        parts = [paper.title.strip()]
        if paper.abstract:
            parts.append(paper.abstract.strip()[:2000])  # Truncate very long abstracts
        if paper.categories:
            parts.append("Categories: " + ", ".join(paper.categories))
        return ". ".join(parts)

    @staticmethod
    def _build_repo_text(repo: Repository) -> str:
        """Build the text representation of a repository for embedding."""
        parts = [repo.full_name.replace("/", " ").strip()]
        if repo.description:
            parts.append(repo.description.strip()[:1000])
        if repo.topics:
            parts.append("Topics: " + ", ".join(repo.topics[:20]))
        if repo.language:
            parts.append(f"Language: {repo.language}")
        return ". ".join(parts)

    # -----------------------------------------------------------------------
    # Private: Deterministic ID generation
    # -----------------------------------------------------------------------
    @staticmethod
    def _deterministic_embedding_id(entity_prefix: str, entity_id: str) -> str:
        """
        Generate a deterministic UUID v5 from entity type + entity ID.
        Ensures the same entity always maps to the same Qdrant point.
        """
        # UUID v5 is deterministic: SHA-1 hash of namespace + name
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace
        name = f"{entity_prefix}:{entity_id}"
        return str(uuid.uuid5(namespace, name))


# ---------------------------------------------------------------------------
# Stub embedding service (used when ml_pipeline is not available)
# ---------------------------------------------------------------------------
class _StubEmbeddingService:
    """
    Fallback embedding service that produces random unit vectors.
    Used in development/testing when ml_pipeline is not installed.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.dim = 768

    async def initialize(self) -> None:
        pass  # No initialization needed

    def encode_single(self, text: str) -> list[float]:
        """Produce a deterministic pseudo-random vector from text hash."""
        import hashlib
        import math

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vector = []
        for i in range(self.dim):
            # Simple deterministic pseudo-random float in [-1, 1]
            val = math.sin(seed * (i + 1)) * math.cos(seed + i)
            vector.append(val)

        # Normalize to unit vector (cosine similarity requires unit vectors)
        norm = math.sqrt(sum(v ** 2 for v in vector))
        return [v / (norm + 1e-9) for v in vector]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode_single(t) for t in texts]
