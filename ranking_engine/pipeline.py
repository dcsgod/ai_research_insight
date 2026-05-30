"""
X-style multi-stage ranking pipeline for the AI Research Intelligence Platform.

Stages (in order):
  1. CandidateRetrieval       – fetch N candidates from DB / vector store
  2. SignalFilter              – remove low-quality / spam candidates
  3. EmbeddingSimilarity       – optional re-rank by user-context similarity
  4. MomentumScoring           – apply TrendScorer
  5. EngagementBooster         – boost recently engaged items
  6. NoveltyBooster            – promote novel / diverse items
  7. ReRankingTransformer      – final re-ranking pass (blends all signals)
  8. FinalRanking              – sort and slice to requested limit

Each stage is a self-contained class implementing process().  Stages can be
added, removed, or swapped without touching the pipeline orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ranking_engine.candidate_retrieval import CandidateItem, CandidateRetriever
from ranking_engine.scorer import ScoringContext, TrendScore, TrendScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the RankingPipeline.

    Attributes:
        max_candidates: Upper bound on candidates entering the pipeline.
        min_quality_score: Items below this are filtered in SignalFilter.
        embedding_similarity_weight: Weight given to user-context similarity
            score when re-ranking (0 disables the stage).
        engagement_boost_factor: Multiplier applied to recently-engaged items.
        novelty_boost_factor: Multiplier applied to high-novelty items.
        novelty_threshold: novelty_score above which the boost is applied.
        reranking_blend: Dict mapping signal name → weight used in final blend.
        experiment_id: Optional A/B experiment identifier attached to logs.
        log_stage_scores: If True, log item scores after each stage.
    """

    max_candidates: int = 500
    min_quality_score: float = 0.05
    embedding_similarity_weight: float = 0.20
    engagement_boost_factor: float = 1.25
    novelty_boost_factor: float = 1.15
    novelty_threshold: float = 0.70
    reranking_blend: Dict[str, float] = field(
        default_factory=lambda: {
            "trend_score": 0.60,
            "recency": 0.20,
            "similarity": 0.20,
        }
    )
    experiment_id: Optional[str] = None
    log_stage_scores: bool = False


# ---------------------------------------------------------------------------
# Ranked item wrapper
# ---------------------------------------------------------------------------


@dataclass
class RankedItem:
    """Wraps a CandidateItem with all accumulated pipeline scores."""

    candidate: CandidateItem
    trend_score: Optional[TrendScore] = None
    similarity_score: float = 0.0    # cosine similarity to user context
    engagement_boost: float = 1.0
    novelty_boost: float = 1.0
    final_rank_score: float = 0.0

    # Per-stage score snapshots (for debugging / A/B analysis)
    stage_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return self.candidate.entity_id

    @property
    def entity_type(self) -> str:
        return self.candidate.entity_type


# ---------------------------------------------------------------------------
# Abstract stage base
# ---------------------------------------------------------------------------


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    name: str = "UnnamedStage"

    @abstractmethod
    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        """Transform and return the item list.

        Args:
            items: Current list of ranked items.
            config: Shared pipeline configuration.
            context: Runtime context dict (user embedding, etc.).

        Returns:
            Processed list (may be filtered/reordered/augmented).
        """
        ...

    def _log_items(self, items: List[RankedItem], config: PipelineConfig) -> None:
        if not config.log_stage_scores:
            return
        for item in items[:10]:
            logger.debug(
                "[%s] %s → final_rank_score=%.4f stage_scores=%s",
                self.name,
                item.entity_id,
                item.final_rank_score,
                item.stage_scores,
            )


# ---------------------------------------------------------------------------
# Stage 1: CandidateRetrievalStage
# ---------------------------------------------------------------------------


class CandidateRetrievalStage(PipelineStage):
    """Fetches initial candidates and wraps them in RankedItem."""

    name = "CandidateRetrieval"

    def __init__(self, retriever: CandidateRetriever) -> None:
        self._retriever = retriever

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        entity_type: str = context.get("entity_type", "all")
        limit = min(context.get("limit", 100), config.max_candidates)

        if entity_type == "paper":
            candidates = await self._retriever.retrieve_papers(limit * 3)
        elif entity_type == "repo":
            candidates = await self._retriever.retrieve_repos(limit * 3)
        elif entity_type == "topic":
            candidates = await self._retriever.retrieve_topics(limit * 3)
        else:
            candidates = await self._retriever.retrieve_all(limit * 3)

        ranked = [RankedItem(candidate=c) for c in candidates]
        logger.info("[%s] Retrieved %d candidates.", self.name, len(ranked))
        return ranked


# ---------------------------------------------------------------------------
# Stage 2: SignalFilter
# ---------------------------------------------------------------------------


class SignalFilter(PipelineStage):
    """Removes spam and low-quality candidates."""

    name = "SignalFilter"

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        before = len(items)
        filtered = [
            item for item in items
            if not item.candidate.is_spam
            and item.candidate.quality_score >= config.min_quality_score
        ]
        logger.info(
            "[%s] Filtered %d → %d items (removed %d).",
            self.name, before, len(filtered), before - len(filtered),
        )
        self._log_items(filtered, config)
        return filtered


# ---------------------------------------------------------------------------
# Stage 3: EmbeddingSimilarity
# ---------------------------------------------------------------------------


class EmbeddingSimilarity(PipelineStage):
    """Re-ranks items by cosine similarity to the user-context embedding."""

    name = "EmbeddingSimilarity"

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        user_embedding: Optional[NDArray[np.float32]] = context.get("user_embedding")

        if user_embedding is None or config.embedding_similarity_weight == 0:
            logger.debug("[%s] Skipped (no user embedding or weight=0).", self.name)
            return items

        user_emb = np.array(user_embedding, dtype=float)
        user_norm = np.linalg.norm(user_emb)
        if user_norm < 1e-10:
            return items
        user_unit = user_emb / user_norm

        for item in items:
            emb = item.candidate.embedding
            if emb is None:
                item.similarity_score = 0.0
                continue
            e = np.array(emb, dtype=float)
            e_norm = np.linalg.norm(e)
            if e_norm < 1e-10:
                item.similarity_score = 0.0
                continue
            sim = float(np.clip(np.dot(user_unit, e / e_norm), -1.0, 1.0))
            # Map from [-1,1] to [0,1]
            item.similarity_score = (sim + 1.0) / 2.0
            item.stage_scores["similarity"] = item.similarity_score

        logger.info("[%s] Computed similarity for %d items.", self.name, len(items))
        self._log_items(items, config)
        return items


# ---------------------------------------------------------------------------
# Stage 4: MomentumScoring
# ---------------------------------------------------------------------------


class MomentumScoring(PipelineStage):
    """Applies the TrendScorer to each candidate."""

    name = "MomentumScoring"

    def __init__(self, scorer: TrendScorer) -> None:
        self._scorer = scorer

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        # Build ScoringContext list
        contexts: List[ScoringContext] = []
        for item in items:
            c = item.candidate
            ctx = ScoringContext(
                entity_id=c.entity_id,
                entity_type=c.entity_type,
                github_stars=c.github_stars,
                github_forks=c.github_forks,
                github_watchers=c.github_watchers,
                github_stars_today=c.github_stars_today,
                citation_history=c.citation_history,
                upvotes=c.upvotes,
                comments=c.comments,
                shares=c.shares,
                signals_history=c.signals_history,
                timestamps=c.timestamps,
                embedding=c.embedding,
                metadata=c.metadata,
            )
            contexts.append(ctx)

        # Reference embeddings for novelty (collect non-None embeddings)
        known_embeddings: List[NDArray[np.float32]] = [
            item.candidate.embedding
            for item in items
            if item.candidate.embedding is not None
        ]

        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        scores: List[TrendScore] = await loop.run_in_executor(
            None,
            lambda: self._scorer.batch_score(contexts, known_embeddings or None),
        )

        score_map: Dict[str, TrendScore] = {s.entity_id: s for s in scores}

        for item in items:
            ts = score_map.get(item.entity_id)
            if ts:
                item.trend_score = ts
                item.final_rank_score = ts.final_score
                item.stage_scores["trend_score"] = ts.final_score

        logger.info("[%s] Scored %d items.", self.name, len(items))
        self._log_items(items, config)
        return items


# ---------------------------------------------------------------------------
# Stage 5: EngagementBooster
# ---------------------------------------------------------------------------


class EngagementBooster(PipelineStage):
    """Boosts items that have been recently engaged with (upvotes, comments, shares).

    Items that received engagement in the last 24 hours get a configurable
    multiplicative boost applied to their final_rank_score.
    """

    name = "EngagementBooster"
    RECENT_ENGAGEMENT_HOURS = 24

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        now = datetime.now(timezone.utc)
        cutoff_hours = self.RECENT_ENGAGEMENT_HOURS

        for item in items:
            c = item.candidate
            last_activity = c.last_activity_at
            is_recent = False
            if last_activity is not None:
                if last_activity.tzinfo is None:
                    last_activity = last_activity.replace(tzinfo=timezone.utc)
                hours_ago = (now - last_activity).total_seconds() / 3600
                is_recent = hours_ago <= cutoff_hours

            # Heuristic: significant engagement = upvotes > 10 or comments > 5
            has_engagement = c.upvotes > 10 or c.comments > 5

            if is_recent and has_engagement:
                item.engagement_boost = config.engagement_boost_factor
                item.final_rank_score *= config.engagement_boost_factor
                item.stage_scores["engagement_boost"] = config.engagement_boost_factor
            else:
                item.stage_scores["engagement_boost"] = 1.0

        logger.info("[%s] Applied engagement boosts.", self.name)
        self._log_items(items, config)
        return items


# ---------------------------------------------------------------------------
# Stage 6: NoveltyBooster
# ---------------------------------------------------------------------------


class NoveltyBooster(PipelineStage):
    """Promotes novel / diverse items to increase feed variety."""

    name = "NoveltyBooster"

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        for item in items:
            if item.trend_score is None:
                continue
            if item.trend_score.novelty_score >= config.novelty_threshold:
                item.novelty_boost = config.novelty_boost_factor
                item.final_rank_score *= config.novelty_boost_factor
                item.stage_scores["novelty_boost"] = config.novelty_boost_factor
            else:
                item.stage_scores["novelty_boost"] = 1.0

        logger.info("[%s] Applied novelty boosts.", self.name)
        self._log_items(items, config)
        return items


# ---------------------------------------------------------------------------
# Stage 7: ReRankingTransformer
# ---------------------------------------------------------------------------


class ReRankingTransformer(PipelineStage):
    """Final re-ranking pass that blends all accumulated signals.

    Computes a weighted combination of:
      - trend_score   (momentum + github + citations + community + novelty)
      - recency       (exponential decay by publication age)
      - similarity    (cosine similarity to user context)

    The blending weights are taken from ``PipelineConfig.reranking_blend``.
    """

    name = "ReRankingTransformer"

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        blend = config.reranking_blend
        w_trend = blend.get("trend_score", 0.60)
        w_recency = blend.get("recency", 0.20)
        w_sim = blend.get("similarity", 0.20)

        for item in items:
            trend = item.trend_score.final_score if item.trend_score else 0.0
            recency = item.candidate.recency_score
            similarity = item.similarity_score

            blended = (
                w_trend * trend
                + w_recency * recency
                + w_sim * similarity
            )
            # Apply boosts (already applied multiplicatively in earlier stages,
            # so we use the current final_rank_score as the base but re-blend
            # with pure signals to avoid double-boost)
            item.final_rank_score = float(np.clip(blended, 0.0, 1.0))
            # Re-apply boosts
            item.final_rank_score *= item.engagement_boost * item.novelty_boost
            item.final_rank_score = float(np.clip(item.final_rank_score, 0.0, 2.0))

            item.stage_scores["final_blended"] = item.final_rank_score

        logger.info("[%s] Re-ranking blend applied to %d items.", self.name, len(items))
        self._log_items(items, config)
        return items


# ---------------------------------------------------------------------------
# Stage 8: FinalRanking
# ---------------------------------------------------------------------------


class FinalRanking(PipelineStage):
    """Sorts and slices the final ranked list."""

    name = "FinalRanking"

    async def process(
        self,
        items: List[RankedItem],
        config: PipelineConfig,
        context: Dict[str, Any],
    ) -> List[RankedItem]:
        limit: int = context.get("limit", 20)
        items.sort(key=lambda x: x.final_rank_score, reverse=True)
        result = items[:limit]
        logger.info("[%s] Final ranking: returning top %d items.", self.name, len(result))
        return result


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class RankingPipeline:
    """Orchestrates the full X-style ranking pipeline.

    Stages are run sequentially.  Each stage receives the output of the
    previous stage.  The pipeline supports:
      - Pluggable stages (add/remove via ``stages`` attribute)
      - A/B experiments via ``experiment_id`` in config
      - Timing and score logging at each stage

    Example::

        retriever = CandidateRetriever(db_pool=pool)
        scorer = TrendScorer()
        pipeline = RankingPipeline(retriever=retriever, scorer=scorer)
        results = await pipeline.rank(
            entity_type="paper",
            limit=20,
            context={"user_embedding": user_emb},
        )
    """

    def __init__(
        self,
        retriever: CandidateRetriever,
        scorer: TrendScorer,
        config: Optional[PipelineConfig] = None,
        extra_stages: Optional[List[PipelineStage]] = None,
    ) -> None:
        """Initialize with required collaborators.

        Args:
            retriever: CandidateRetriever instance.
            scorer: TrendScorer instance.
            config: Optional PipelineConfig (defaults used if None).
            extra_stages: Additional stages to append after NoveltyBooster
                          and before ReRankingTransformer.
        """
        self.config = config or PipelineConfig()
        self._retriever = retriever
        self._scorer = scorer

        # Build the default stage sequence
        self.stages: List[PipelineStage] = [
            CandidateRetrievalStage(retriever),
            SignalFilter(),
            EmbeddingSimilarity(),
            MomentumScoring(scorer),
            EngagementBooster(),
            NoveltyBooster(),
            *(extra_stages or []),
            ReRankingTransformer(),
            FinalRanking(),
        ]

        exp = self.config.experiment_id
        logger.info(
            "RankingPipeline ready: %d stages, experiment_id=%s",
            len(self.stages),
            exp,
        )

    async def rank(
        self,
        entity_type: str = "all",
        limit: int = 20,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RankedItem]:
        """Run the full ranking pipeline and return the top items.

        Args:
            entity_type: Type of entities to rank ("paper", "repo", "topic", "all").
            limit: Number of items to return.
            context: Optional dict with runtime context, e.g.::
                     {"user_embedding": np.ndarray, "user_id": "u123"}

        Returns:
            Ordered list of RankedItem (highest score first).
        """
        run_id = str(uuid.uuid4())[:8]
        experiment = self.config.experiment_id or "default"

        # Merge caller context with pipeline-required fields
        runtime_ctx: Dict[str, Any] = {
            "entity_type": entity_type,
            "limit": limit,
            "run_id": run_id,
            "experiment_id": experiment,
            **(context or {}),
        }

        logger.info(
            "Pipeline run %s: entity_type=%s, limit=%d, experiment=%s",
            run_id, entity_type, limit, experiment,
        )

        items: List[RankedItem] = []
        total_start = time.perf_counter()

        for stage in self.stages:
            stage_start = time.perf_counter()
            try:
                items = await stage.process(items, self.config, runtime_ctx)
            except Exception as exc:
                logger.error(
                    "Pipeline run %s: stage %s failed: %s",
                    run_id, stage.name, exc, exc_info=True,
                )
                # Continue with the current items list rather than crashing
                # so the pipeline can still return partial results.
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            logger.debug(
                "Pipeline run %s: [%s] completed in %.1f ms, %d items.",
                run_id, stage.name, elapsed_ms, len(items),
            )

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "Pipeline run %s complete: %d results in %.1f ms.",
            run_id, len(items), total_ms,
        )
        return items

    def add_stage(self, stage: PipelineStage, position: int = -1) -> None:
        """Insert a stage at the given position (-1 = before FinalRanking).

        Args:
            stage: A PipelineStage instance.
            position: Index in self.stages to insert at. -1 inserts before
                      the last stage (FinalRanking).
        """
        if position == -1:
            self.stages.insert(-1, stage)
        else:
            self.stages.insert(position, stage)
        logger.info("Stage %s added at position %d.", stage.name, position)

    def remove_stage(self, stage_name: str) -> bool:
        """Remove a stage by name.

        Args:
            stage_name: The ``name`` attribute of the stage to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, stage in enumerate(self.stages):
            if stage.name == stage_name:
                self.stages.pop(i)
                logger.info("Stage %s removed.", stage_name)
                return True
        logger.warning("Stage %s not found; nothing removed.", stage_name)
        return False
