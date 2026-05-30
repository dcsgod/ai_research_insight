"""
AI Research Intelligence Platform — BERTopic Topic Extractor
Extracts and clusters emerging topics from paper abstracts and repo descriptions.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TopicAssignment:
    """Result of assigning a document to a topic."""
    document_id: str
    topic_id: int
    probability: float
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    topic_name: str = ""


@dataclass
class TopicInfo:
    """Metadata about a discovered topic."""
    topic_id: int
    name: str
    keywords: List[Tuple[str, float]]
    document_count: int
    representative_docs: List[str] = field(default_factory=list)


class TopicExtractor:
    """
    BERTopic-based topic extraction for AI research documents.
    Uses SentenceTransformers embeddings + UMAP + HDBSCAN clustering.
    Falls back to simple TF-IDF clustering if BERTopic is unavailable.
    """

    def __init__(
        self,
        embedding_service=None,
        n_topics: str = "auto",
        min_topic_size: int = 5,
        model_cache_path: str = "./ml_pipeline/models/bertopic_model",
    ):
        self.embedding_service = embedding_service
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.model_cache_path = model_cache_path
        self._model = None
        self._topic_metadata: Dict[int, TopicInfo] = {}
        self._is_fitted = False

    def _get_model(self):
        """Lazily initialize BERTopic model."""
        if self._model is not None:
            return self._model
        try:
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer

            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_topic_size,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            vectorizer = CountVectorizer(
                stop_words="english",
                min_df=2,
                ngram_range=(1, 2),
            )
            self._model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                top_n_words=10,
                verbose=False,
                calculate_probabilities=True,
            )
            logger.info("BERTopic model initialized")
        except ImportError:
            logger.warning("BERTopic not available — using TF-IDF fallback")
            self._model = "tfidf_fallback"
        return self._model

    def _tfidf_fallback_fit(self, documents: List[str]) -> Tuple[List[int], List[float]]:
        """Simple TF-IDF + KMeans fallback when BERTopic unavailable."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        n_clusters = max(3, min(20, len(documents) // 10))
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        X = vectorizer.fit_transform(documents)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        probs = np.ones(len(labels)) * 0.7
        self._tfidf_vectorizer = vectorizer
        self._tfidf_kmeans = kmeans
        return labels.tolist(), probs.tolist()

    def fit(self, documents: List[str], document_ids: Optional[List[str]] = None) -> None:
        """Train the topic model on a corpus of documents."""
        if len(documents) < self.min_topic_size:
            logger.warning(f"Too few documents ({len(documents)}) to fit topic model")
            return

        logger.info(f"Fitting topic model on {len(documents)} documents...")
        model = self._get_model()

        if model == "tfidf_fallback":
            labels, probs = self._tfidf_fallback_fit(documents)
            self._store_topics_from_labels(documents, labels, document_ids)
        else:
            try:
                embeddings = None
                if self.embedding_service:
                    embeddings = self.embedding_service.encode(documents)

                topics, probs = model.fit_transform(documents, embeddings=embeddings)
                self._store_topic_metadata(model, documents)
                logger.info(f"BERTopic found {len(set(topics)) - 1} topics")
            except Exception as e:
                logger.error(f"BERTopic fitting failed: {e}, using fallback")
                labels, probs = self._tfidf_fallback_fit(documents)
                self._store_topics_from_labels(documents, labels, document_ids)

        self._is_fitted = True

    def _store_topic_metadata(self, model, documents: List[str]) -> None:
        """Extract and store topic metadata from a trained BERTopic model."""
        try:
            topic_info_df = model.get_topic_info()
            for _, row in topic_info_df.iterrows():
                tid = int(row["Topic"])
                if tid == -1:
                    continue
                keywords = model.get_topic(tid) or []
                name = "_".join([kw for kw, _ in keywords[:3]])
                self._topic_metadata[tid] = TopicInfo(
                    topic_id=tid,
                    name=name,
                    keywords=keywords,
                    document_count=int(row["Count"]),
                )
        except Exception as e:
            logger.error(f"Failed to store topic metadata: {e}")

    def _store_topics_from_labels(
        self,
        documents: List[str],
        labels: List[int],
        document_ids: Optional[List[str]],
    ) -> None:
        """Build topic metadata from cluster labels (TF-IDF fallback)."""
        from collections import Counter, defaultdict
        from sklearn.feature_extraction.text import TfidfVectorizer

        clusters: Dict[int, List[str]] = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(documents[i])

        for tid, docs in clusters.items():
            try:
                vec = TfidfVectorizer(max_features=10, stop_words="english")
                vec.fit(docs)
                keywords = [(w, 1.0) for w in vec.get_feature_names_out()]
                name = "_".join([kw for kw, _ in keywords[:3]])
                self._topic_metadata[tid] = TopicInfo(
                    topic_id=tid,
                    name=name,
                    keywords=keywords,
                    document_count=len(docs),
                )
            except Exception:
                pass

    def transform(
        self, documents: List[str], document_ids: Optional[List[str]] = None
    ) -> List[TopicAssignment]:
        """Assign topics to new documents using the trained model."""
        if not self._is_fitted:
            logger.warning("Model not fitted — fitting now")
            self.fit(documents, document_ids)

        if document_ids is None:
            document_ids = [str(i) for i in range(len(documents))]

        model = self._get_model()
        assignments = []

        try:
            if model == "tfidf_fallback":
                X = self._tfidf_vectorizer.transform(documents)
                labels = self._tfidf_kmeans.predict(X)
                for i, (doc_id, label) in enumerate(zip(document_ids, labels)):
                    info = self._topic_metadata.get(int(label))
                    assignments.append(TopicAssignment(
                        document_id=doc_id,
                        topic_id=int(label),
                        probability=0.7,
                        keywords=info.keywords[:5] if info else [],
                        topic_name=info.name if info else f"topic_{label}",
                    ))
            else:
                topics, probs = model.transform(documents)
                for doc_id, topic_id, prob in zip(document_ids, topics, probs):
                    info = self._topic_metadata.get(int(topic_id))
                    p = float(prob) if isinstance(prob, (int, float)) else 0.5
                    assignments.append(TopicAssignment(
                        document_id=doc_id,
                        topic_id=int(topic_id),
                        probability=p,
                        keywords=info.keywords[:5] if info else [],
                        topic_name=info.name if info else f"topic_{topic_id}",
                    ))
        except Exception as e:
            logger.error(f"Transform failed: {e}")

        return assignments

    def get_all_topics(self) -> List[TopicInfo]:
        """Return all discovered topics with metadata."""
        return list(self._topic_metadata.values())

    def extract_keywords(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get keyword-score pairs for a topic."""
        info = self._topic_metadata.get(topic_id)
        return info.keywords if info else []

    def get_topic_summary(self, topic_id: int) -> str:
        """Generate a human-readable topic summary from keywords."""
        keywords = self.extract_keywords(topic_id)
        if not keywords:
            return f"Topic {topic_id}"
        top_words = [kw for kw, _ in keywords[:5]]
        return " | ".join(top_words)

    def find_similar_topics(self, topic_id: int, top_n: int = 5) -> List[Tuple[int, float]]:
        """Find topics most similar to the given topic by keyword overlap."""
        source = self._topic_metadata.get(topic_id)
        if not source:
            return []
        source_kws = {kw for kw, _ in source.keywords}
        similarities = []
        for tid, info in self._topic_metadata.items():
            if tid == topic_id:
                continue
            other_kws = {kw for kw, _ in info.keywords}
            overlap = len(source_kws & other_kws) / max(len(source_kws | other_kws), 1)
            similarities.append((tid, overlap))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def get_visualization_data(self) -> Dict[str, Any]:
        """Export topic data formatted for frontend graph visualization."""
        nodes = []
        edges = []
        for tid, info in self._topic_metadata.items():
            nodes.append({
                "id": tid,
                "name": info.name,
                "size": info.document_count,
                "keywords": [kw for kw, _ in info.keywords[:5]],
            })
            similar = self.find_similar_topics(tid, top_n=3)
            for other_tid, score in similar:
                if score > 0.2:
                    edges.append({"source": tid, "target": other_tid, "weight": score})

        return {"nodes": nodes, "edges": edges}

    def save(self, path: Optional[str] = None) -> None:
        """Persist topic metadata to disk."""
        save_path = path or self.model_cache_path
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        meta_path = f"{save_path}_metadata.json"
        serializable = {
            str(k): {
                "topic_id": v.topic_id,
                "name": v.name,
                "keywords": v.keywords,
                "document_count": v.document_count,
            }
            for k, v in self._topic_metadata.items()
        }
        with open(meta_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Topic metadata saved to {meta_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load topic metadata from disk."""
        load_path = path or self.model_cache_path
        meta_path = f"{load_path}_metadata.json"
        if not os.path.exists(meta_path):
            logger.warning(f"No saved topic metadata at {meta_path}")
            return
        with open(meta_path) as f:
            data = json.load(f)
        self._topic_metadata = {
            int(k): TopicInfo(
                topic_id=v["topic_id"],
                name=v["name"],
                keywords=v["keywords"],
                document_count=v["document_count"],
            )
            for k, v in data.items()
        }
        self._is_fitted = True
        logger.info(f"Loaded {len(self._topic_metadata)} topics from {meta_path}")
