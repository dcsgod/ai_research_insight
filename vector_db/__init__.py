"""Vector database integration for similarity search."""

from vector_db.qdrant_client import QdrantService, SearchResult

__all__ = [
    "QdrantService",
    "SearchResult",
]
