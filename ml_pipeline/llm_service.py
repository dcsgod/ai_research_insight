"""
AI Research Intelligence Platform — LLM Insight Generation Service
Generates AI-powered summaries about trending papers, repos, and topics.
Supports Ollama (local), OpenAI, and vLLM backends.
"""
import asyncio
import hashlib
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    VLLM = "vllm"


# ── Prompt Templates ──────────────────────────────────────────────────────────

INSIGHT_PROMPTS = {
    "paper": (
        "You are an AI research analyst. Generate a 2-3 sentence insight about this "
        "research paper's significance and momentum in the AI ecosystem.\n\n"
        "Paper: {title}\nAbstract: {abstract}\nTrend Score: {trend_score:.2f}\n\n"
        "Focus on: why it's gaining momentum, real-world applications, and what makes "
        "it stand out. Be specific and use active language. Do NOT start with 'This paper'."
    ),
    "repo": (
        "You are an AI research analyst. Generate a 2-3 sentence insight about this "
        "GitHub repository's growing impact in the AI ecosystem.\n\n"
        "Repository: {name}\nDescription: {description}\nStars: {stars:,}\n"
        "Stars Today: {stars_today}\nTrend Score: {trend_score:.2f}\n\n"
        "Focus on: what problem it solves, why developers love it, and its trajectory. "
        "Be specific and energetic. Do NOT start with 'This repo'."
    ),
    "topic": (
        "You are an AI research analyst. Generate a 2-3 sentence insight about this "
        "emerging AI topic and why it's rapidly gaining momentum.\n\n"
        "Topic: {name}\nKeywords: {keywords}\nPapers: {paper_count}, Repos: {repo_count}\n"
        "Momentum Score: {trend_score:.2f}\n\n"
        "Explain what's driving interest, key applications, and future outlook. "
        "Be insightful and forward-looking."
    ),
    "daily_summary": (
        "You are an AI research analyst writing the daily AI ecosystem briefing. "
        "Summarize today's top 3-5 AI trends in 3-4 sentences. Be concise, specific, "
        "and highlight the most important shifts happening.\n\n"
        "Top trends today:\n{trends_list}\n\n"
        "Write a crisp, Bloomberg-style market summary."
    ),
}


class LLMService:
    """
    Multi-provider LLM service for generating AI research insights.
    Defaults to Ollama for local inference, falls back to OpenAI.
    """

    def __init__(
        self,
        provider: str = "ollama",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.provider = LLMProvider(provider)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, str] = {}  # In-memory cache (replace with Redis in prod)

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    async def _generate_ollama(self, prompt: str) -> str:
        """Call Ollama local inference API."""
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 200,
            },
        }
        response = await self.http_client.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    async def _generate_openai(self, prompt: str) -> str:
        """Call OpenAI API (gpt-4o-mini by default)."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250,
            "temperature": 0.7,
        }
        response = await self.http_client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _generate(self, prompt: str) -> str:
        """Generate text using configured provider with retry logic."""
        cache_key = self._cache_key(prompt)
        if cache_key in self._cache:
            return self._cache[cache_key]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == LLMProvider.OLLAMA:
                    result = await self._generate_ollama(prompt)
                elif self.provider == LLMProvider.OPENAI:
                    result = await self._generate_openai(prompt)
                else:
                    result = await self._generate_ollama(prompt)

                self._cache[cache_key] = result
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        # Graceful fallback — return a templated insight
        logger.error(f"All LLM attempts failed: {last_error}")
        return self._fallback_insight()

    def _fallback_insight(self) -> str:
        return (
            "This topic is showing strong momentum in the AI research community, "
            "with growing adoption across multiple domains and increasing community engagement."
        )

    async def generate_paper_insight(self, paper: Dict[str, Any]) -> str:
        """Generate insight for a research paper."""
        prompt = INSIGHT_PROMPTS["paper"].format(
            title=paper.get("title", "Unknown"),
            abstract=(paper.get("abstract") or "")[:500],
            trend_score=paper.get("trend_score", 0.0),
        )
        return await self._generate(prompt)

    async def generate_repo_insight(self, repo: Dict[str, Any]) -> str:
        """Generate insight for a GitHub repository."""
        prompt = INSIGHT_PROMPTS["repo"].format(
            name=repo.get("full_name", repo.get("name", "Unknown")),
            description=(repo.get("description") or "No description")[:300],
            stars=repo.get("stars", 0),
            stars_today=repo.get("stars_today", 0),
            trend_score=repo.get("trend_score", 0.0),
        )
        return await self._generate(prompt)

    async def generate_topic_insight(self, topic: Dict[str, Any]) -> str:
        """Generate insight for a topic cluster."""
        keywords = ", ".join(topic.get("keywords", [])[:8])
        prompt = INSIGHT_PROMPTS["topic"].format(
            name=topic.get("name", "Unknown"),
            keywords=keywords,
            paper_count=topic.get("paper_count", 0),
            repo_count=topic.get("repo_count", 0),
            trend_score=topic.get("trend_score", 0.0),
        )
        return await self._generate(prompt)

    async def generate_insight(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Generate insight for any entity type."""
        if entity_type == "paper":
            return await self.generate_paper_insight(entity)
        elif entity_type == "repo":
            return await self.generate_repo_insight(entity)
        elif entity_type == "topic":
            return await self.generate_topic_insight(entity)
        return self._fallback_insight()

    async def generate_daily_summary(self, top_trends: List[Dict[str, Any]]) -> str:
        """Generate a daily AI ecosystem market summary."""
        trends_list = "\n".join(
            f"- {t.get('title', t.get('name', 'Unknown'))} (score: {t.get('trend_score', 0):.2f})"
            for t in top_trends[:5]
        )
        prompt = INSIGHT_PROMPTS["daily_summary"].format(trends_list=trends_list)
        return await self._generate(prompt)

    async def generate_topic_analysis(self, topic: Dict[str, Any]) -> str:
        """Generate detailed topic analysis."""
        return await self.generate_topic_insight(topic)

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
