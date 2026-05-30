"""
SentenceTransformer embedding service for the AI Research Intelligence Platform.

Provides:
  - EmbeddingService: Encode texts using sentence-transformers.
  - Singleton/model-caching pattern so the model is loaded once per process.
  - Async encoding via thread-pool executor to avoid blocking the event loop.
  - Long-text handling: truncation with configurable max token length.
  - Cosine similarity utilities (single pair and batched).
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Lazy import so that the platform can start without the model downloaded.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. EmbeddingService will return zero vectors. "
        "Install with: pip install sentence-transformers"
    )


# ---------------------------------------------------------------------------
# Module-level model cache (shared across instances)
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}
_MODEL_LOCK = threading.Lock()


def _get_or_load_model(model_name: str, device: str) -> Optional["SentenceTransformer"]:
    """Return a cached model, loading it if necessary.

    Thread-safe: uses a module-level lock so that concurrent callers don't
    trigger duplicate loads.

    Args:
        model_name: Hugging Face model identifier or local path.
        device: Device string ("cpu", "cuda", "mps", etc.).

    Returns:
        Loaded SentenceTransformer model, or None if not available.
    """
    if not _ST_AVAILABLE:
        return None

    cache_key = f"{model_name}::{device}"
    with _MODEL_LOCK:
        if cache_key not in _MODEL_CACHE:
            logger.info("Loading sentence-transformer model '%s' on %s.", model_name, device)
            _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, device=device)
            logger.info("Model '%s' loaded successfully.", model_name)
        return _MODEL_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Encodes texts into dense vectors using a SentenceTransformer model.

    This service is designed to be instantiated once and shared across the
    application.  All heavy model-loading is deferred to the first call.

    Args:
        model_name: Hugging Face model ID (default: all-MiniLM-L6-v2,
                    which produces 384-dimensional embeddings).
        device: Torch device string.  "cpu" is safe everywhere; use "cuda"
                or "mps" when GPU is available.
        batch_size: Number of texts to encode per forward pass.
        max_seq_length: Maximum number of tokens.  Texts longer than this
                        are truncated (with overlap for long documents).
        normalize_embeddings: If True, outputs are L2-normalized (cosine
                              similarity then equals dot product).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._normalize = normalize_embeddings
        self._model: Optional["SentenceTransformer"] = None  # lazy load

        logger.info(
            "EmbeddingService configured: model=%s device=%s batch_size=%d",
            model_name, device, batch_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Optional["SentenceTransformer"]:
        """Load the model on first use (lazy initialization)."""
        if self._model is None:
            self._model = _get_or_load_model(self._model_name, self._device)
            if self._model is not None and hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self._max_seq_length
        return self._model

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 2048) -> str:
        """Hard-truncate text to avoid OOM on extremely long documents.

        A smarter approach (used in production) is to chunk with overlap and
        mean-pool the chunk embeddings.  This simpler truncation is safe for
        the common case.

        Args:
            text: Input text.
            max_chars: Maximum character count (not token count).

        Returns:
            Truncated string.
        """
        if len(text) <= max_chars:
            return text
        # Truncate at the nearest word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            truncated = truncated[:last_space]
        logger.debug("Text truncated from %d to %d chars.", len(text), len(truncated))
        return truncated

    def _make_zero_embedding(self) -> NDArray[np.float32]:
        """Return a zero vector of the expected embedding dimension."""
        # Use 384 for all-MiniLM-L6-v2; will be overridden after first real encode.
        return np.zeros(384, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public encoding API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: List[str],
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]:
        """Encode a list of texts into an embedding matrix.

        Args:
            texts: Input strings to encode.
            show_progress_bar: If True, display a tqdm progress bar.

        Returns:
            Float32 array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self._ensure_model()
        if model is None:
            logger.warning("Model not available; returning zero embeddings.")
            return np.stack([self._make_zero_embedding() for _ in texts])

        # Truncate long texts
        processed = [self._truncate_text(t) for t in texts]

        embeddings: NDArray[np.float32] = model.encode(
            processed,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        logger.debug("Encoded %d texts, shape=%s.", len(texts), embeddings.shape)
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> NDArray[np.float32]:
        """Encode a single text string.

        Args:
            text: Input string.

        Returns:
            1-D float32 embedding array.
        """
        return self.encode([text])[0]

    async def encode_async(
        self,
        texts: List[str],
        executor: Optional[object] = None,
    ) -> NDArray[np.float32]:
        """Encode texts asynchronously using a thread-pool executor.

        The SentenceTransformer model is synchronous; wrapping it in an
        executor prevents blocking the event loop during batch encoding.

        Args:
            texts: Input strings.
            executor: Optional ``concurrent.futures.Executor`` instance.
                      If None, the event loop's default executor is used.

        Returns:
            Float32 embedding array.
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            executor,
            lambda: self.encode(texts),
        )
        return embeddings

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_similarity(
        emb1: NDArray[np.float32],
        emb2: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            emb1: First embedding (1-D array).
            emb2: Second embedding (1-D array, must match shape of emb1).

        Returns:
            Cosine similarity in [-1, 1].
        """
        e1 = np.array(emb1, dtype=float)
        e2 = np.array(emb2, dtype=float)
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.clip(np.dot(e1 / n1, e2 / n2), -1.0, 1.0))

    @staticmethod
    def batch_similarity(
        query_emb: NDArray[np.float32],
        corpus_embs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between one query and a corpus matrix.

        Args:
            query_emb: 1-D query embedding, shape (D,).
            corpus_embs: 2-D corpus matrix, shape (N, D).

        Returns:
            1-D array of similarity scores, shape (N,), in [-1, 1].
        """
        q = np.array(query_emb, dtype=float)
        C = np.array(corpus_embs, dtype=float)

        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return np.zeros(len(C), dtype=np.float32)

        q_unit = q / q_norm

        # Row-wise norms of corpus
        c_norms = np.linalg.norm(C, axis=1, keepdims=True)
        c_norms = np.where(c_norms < 1e-10, 1e-10, c_norms)
        C_unit = C / c_norms

        scores = C_unit @ q_unit
        return np.clip(scores, -1.0, 1.0).astype(np.float32)

    def find_most_similar(
        self,
        query_text: str,
        corpus_texts: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """Encode a query and a corpus, return the top-k most similar items.

        Args:
            query_text: The query string.
            corpus_texts: List of candidate strings.
            top_k: Number of results to return.

        Returns:
            List of (index, similarity_score, text) tuples, sorted by score
            descending.
        """
        if not corpus_texts:
            return []

        all_texts = [query_text] + corpus_texts
        embeddings = self.encode(all_texts)

        query_emb = embeddings[0]
        corpus_embs = embeddings[1:]

        scores = self.batch_similarity(query_emb, corpus_embs)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (int(idx), float(scores[idx]), corpus_texts[idx])
            for idx in top_indices
        ]
        return results

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality of the loaded model."""
        model = self._ensure_model()
        if model is None:
            return 384  # default for all-MiniLM-L6-v2
        # SentenceTransformer exposes get_sentence_embedding_dimension()
        if hasattr(model, "get_sentence_embedding_dimension"):
            return model.get_sentence_embedding_dimension()
        return 384
