"""
AI Research Intelligence Platform — Qdrant Vector Database Client
Provides embedding storage and similarity search for papers, repos, and topics.
"""
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a vector similarity search."""
    id: str
    score: float
    payload: Dict[str, Any]


class QdrantService:
    """
    Async Qdrant vector database service.
    Manages embeddings for papers, repositories, and topics.
    """

    COLLECTIONS = {
        "papers": "paper_embeddings",
        "repos": "repo_embeddings",
        "topics": "topic_embeddings",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        vector_size: int = 384,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.vector_size = vector_size
        self._client: Optional[AsyncQdrantClient] = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            kwargs: Dict[str, Any] = {"host": self.host, "port": self.port}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = AsyncQdrantClient(**kwargs)
        return self._client

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def setup_collections(self) -> None:
        """
        Create all required vector collections if they don't exist.
        Uses Cosine distance with 384-dim vectors (all-MiniLM-L6-v2).
        """
        for name, collection_name in self.COLLECTIONS.items():
            try:
                existing = await self.client.get_collections()
                existing_names = [c.name for c in existing.collections]

                if collection_name not in existing_names:
                    await self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=self.vector_size,
                            distance=qdrant_models.Distance.COSINE,
                        ),
                        optimizers_config=qdrant_models.OptimizersConfigDiff(
                            indexing_threshold=20000,
                        ),
                        hnsw_config=qdrant_models.HnswConfigDiff(
                            m=16,
                            ef_construct=100,
                        ),
                    )
                    logger.info(f"✅ Created Qdrant collection: {collection_name}")
                else:
                    logger.debug(f"Collection already exists: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                raise

    async def upsert_embedding(
        self,
        collection: str,
        id: str,
        vector: np.ndarray,
        payload: Dict[str, Any],
    ) -> str:
        """
        Insert or update an embedding in a collection.
        Returns the point ID used.
        """
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, id))

        try:
            await self.client.upsert(
                collection_name=collection,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={**payload, "source_id": id},
                    )
                ],
            )
            return point_id
        except Exception as e:
            logger.error(f"Failed to upsert embedding for {id}: {e}")
            raise

    async def batch_upsert(
        self,
        collection: str,
        items: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Batch upsert multiple embeddings efficiently.
        Each item: {"id": str, "vector": np.ndarray, "payload": dict}
        Returns count of upserted items.
        """
        total = 0
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            points = [
                qdrant_models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, item["id"])),
                    vector=item["vector"].tolist(),
                    payload={**item["payload"], "source_id": item["id"]},
                )
                for item in batch
            ]
            try:
                await self.client.upsert(collection_name=collection, points=points)
                total += len(batch)
            except Exception as e:
                logger.error(f"Batch upsert failed for batch {i}: {e}")
        return total

    async def search_similar(
        self,
        collection: str,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Find the most similar vectors in a collection.
        Returns results sorted by similarity score descending.
        """
        query_filter = None
        if filter_payload:
            conditions = [
                qdrant_models.FieldCondition(
                    key=k,
                    match=qdrant_models.MatchValue(value=v),
                )
                for k, v in filter_payload.items()
            ]
            query_filter = qdrant_models.Filter(must=conditions)

        try:
            results = await self.client.search(
                collection_name=collection,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Similarity search failed in {collection}: {e}")
            return []

    async def get_embedding(
        self, collection: str, source_id: str
    ) -> Optional[np.ndarray]:
        """Retrieve a stored embedding vector by its source ID."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))
        try:
            results = await self.client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_vectors=True,
            )
            if results and results[0].vector:
                return np.array(results[0].vector)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve embedding {source_id}: {e}")
            return None

    async def delete_embedding(self, collection: str, source_id: str) -> bool:
        """Delete an embedding from a collection."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))
        try:
            await self.client.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding {source_id}: {e}")
            return False

    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        try:
            info = await self.client.get_collection(collection)
            return {
                "name": collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {collection}: {e}")
            return {"name": collection, "error": str(e)}

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client:
            await self._client.close()
            self._client = None
