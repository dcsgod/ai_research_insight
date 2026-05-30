"""
Reddit MCP Server for AI Research Intelligence Platform.

Exposes four MCP tools:
  - get_hot_posts       – hot posts from a single subreddit
  - get_trending_topics – aggregated trending topics across multiple subreddits
  - get_rising_posts    – rising posts from a single subreddit
  - analyze_sentiment   – simple rule-based sentiment analysis of post titles

Uses the Reddit OAuth2 "application-only" (client credentials) flow so that
no user account is required.  Set the following environment variables:

    REDDIT_CLIENT_ID     – OAuth2 application client ID
    REDDIT_CLIENT_SECRET – OAuth2 application client secret
    REDDIT_USER_AGENT    – User-Agent string (required by Reddit API ToS)

Results are normalised to TopicSignalSchema / NormalizedSignal objects.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import httpx

from .base_server import (
    BaseServer,
    MCPAuthError,
    MCPFetchError,
    MCPRateLimitError,
    NormalizedSignal,
    TopicSignalSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

DEFAULT_SUBREDDITS: List[str] = [
    "MachineLearning",
    "LocalLLaMA",
    "artificial",
    "datascience",
    "deeplearning",
]

# Words that indicate positive/negative sentiment
_POSITIVE_WORDS: Set[str] = {
    "breakthrough", "amazing", "great", "excellent", "impressive", "new",
    "release", "launch", "milestone", "sota", "best", "efficient",
    "fast", "open", "free", "win", "success", "achieve", "improve",
}
_NEGATIVE_WORDS: Set[str] = {
    "bug", "issue", "problem", "fail", "broken", "slow", "bad", "error",
    "vulnerability", "concern", "risk", "danger", "deprecated", "remove",
    "drop", "ban", "block", "restrict",
}

# Simple stopwords for keyword extraction
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "or", "and", "but", "not", "this", "that", "it", "its",
    "i", "you", "we", "they", "he", "she", "my", "your", "our",
    "their", "what", "how", "why", "when", "where", "who", "which",
}

MAX_RESULTS_CAP = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_to_dt(ts: Optional[float]) -> Optional[datetime]:
    """Convert a Unix UTC timestamp to an aware datetime."""
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract the most informative words from a piece of text using a simple
    frequency-based approach after stopword removal.
    """
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq: Dict[str, int] = {}
    for token in tokens:
        if token not in _STOPWORDS:
            freq[token] = freq.get(token, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_tokens[:top_n]]


def _classify_sentiment(title: str) -> str:
    """
    Rule-based sentiment classification for a post title.

    Returns "positive", "negative", or "neutral".
    """
    tokens = set(re.findall(r"\b[a-z]+\b", title.lower()))
    pos_hits = len(tokens & _POSITIVE_WORDS)
    neg_hits = len(tokens & _NEGATIVE_WORDS)
    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Reddit MCP Server
# ---------------------------------------------------------------------------


class RedditMCPServer(BaseServer):
    """
    MCP server wrapping the Reddit REST API v1 via OAuth2 client credentials.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        super().__init__(server_name="reddit")
        self._client_id: str = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self._client_secret: str = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self._user_agent: str = (
            user_agent
            or os.getenv("REDDIT_USER_AGENT", "AI-Research-Platform/1.0 (MCP)")
        )
        self._timeout = timeout

        self._access_token: str = ""
        self._token_expires_at: float = 0.0

        if not self._client_id or not self._client_secret:
            self._logger.warning(
                "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set – "
                "Reddit API calls will fail until credentials are provided."
            )

        # Register tools
        self._register_tool("get_hot_posts", self._tool_get_hot_posts)
        self._register_tool("get_trending_topics", self._tool_get_trending_topics)
        self._register_tool("get_rising_posts", self._tool_get_rising_posts)
        self._register_tool("analyze_sentiment", self._tool_analyze_sentiment)

        self._logger.info("RedditMCPServer initialised")

    # ------------------------------------------------------------------
    # OAuth2 token management
    # ------------------------------------------------------------------

    async def _ensure_token(self) -> None:
        """Obtain or refresh the OAuth2 access token if needed."""
        if self._access_token and time.monotonic() < self._token_expires_at - 60:
            return  # Token still valid

        if not self._client_id or not self._client_secret:
            raise MCPAuthError(
                "Reddit client credentials not configured",
                server_name=self.server_name,
            )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                REDDIT_TOKEN_URL,
                auth=(self._client_id, self._client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": self._user_agent},
            )

            if response.status_code == 401:
                raise MCPAuthError(
                    "Reddit returned 401 – invalid client credentials",
                    server_name=self.server_name,
                    status_code=401,
                )

            response.raise_for_status()
            token_data = response.json()
            self._access_token = token_data.get("access_token", "")
            expires_in = float(token_data.get("expires_in", 3600))
            self._token_expires_at = time.monotonic() + expires_in
            self._logger.info(
                "Reddit token acquired (expires in %.0fs)", expires_in
            )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    async def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Authenticated GET to Reddit OAuth API with retry."""
        await self._ensure_token()
        url = f"{REDDIT_OAUTH_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "User-Agent": self._user_agent,
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            for attempt in range(1, 4):
                try:
                    response = await client.get(url, headers=headers, params=params)

                    if response.status_code == 401:
                        # Token may have expired; clear and retry once
                        self._access_token = ""
                        await self._ensure_token()
                        headers["Authorization"] = f"Bearer {self._access_token}"
                        continue

                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise MCPRateLimitError(
                            f"Reddit rate limit (429). Retry after {retry_after}s.",
                            server_name=self.server_name,
                            status_code=429,
                        )

                    if response.status_code >= 500:
                        if attempt < 3:
                            await asyncio.sleep(2.0 ** (attempt - 1))
                            continue
                        raise MCPFetchError(
                            f"Reddit returned HTTP {response.status_code}",
                            server_name=self.server_name,
                            status_code=response.status_code,
                        )

                    response.raise_for_status()
                    return response.json()

                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    if attempt == 3:
                        raise MCPFetchError(
                            f"Failed to connect to Reddit after 3 attempts: {exc}",
                            server_name=self.server_name,
                        ) from exc
                    await asyncio.sleep(2.0 ** (attempt - 1))

        raise MCPFetchError("Exhausted retries", server_name=self.server_name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    async def fetch_data(self, **kwargs: Any) -> Any:
        if "subreddits" in kwargs:
            return await self._tool_get_trending_topics(**kwargs)
        subreddit = kwargs.get("subreddit", "MachineLearning")
        return await self._tool_get_hot_posts(subreddit=subreddit)

    def normalize(self, raw: Any) -> List[NormalizedSignal]:
        """
        Convert Reddit API listing data into NormalizedSignal objects.

        ``raw`` may be:
          - a Reddit listing dict with ``data.children``
          - a list of already-extracted post dicts
          - a list of TopicSignalSchema (from analyze_sentiment)
        """
        if raw is None:
            return []

        # Already normalised TopicSignalSchema objects (from analyze_sentiment)
        if isinstance(raw, list) and raw and isinstance(raw[0], TopicSignalSchema):
            return [NormalizedSignal.from_topic(t) for t in raw]

        # List of raw post dicts
        if isinstance(raw, list):
            posts = raw
        elif isinstance(raw, dict) and "data" in raw:
            listing_data = raw["data"]
            posts = [child["data"] for child in listing_data.get("children", [])]
        else:
            self._logger.warning("Unexpected raw type from Reddit: %s", type(raw))
            return []

        signals: List[NormalizedSignal] = []
        for post in posts:
            try:
                topic = self._post_to_topic(post)
                if topic:
                    signals.append(NormalizedSignal.from_topic(topic))
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.warning("Skipping Reddit post due to error: %s", exc)

        self._logger.debug("Normalised %d Reddit topics", len(signals))
        return signals

    # ------------------------------------------------------------------
    # Data mapping
    # ------------------------------------------------------------------

    def _post_to_topic(self, post: Dict[str, Any]) -> Optional[TopicSignalSchema]:
        """Convert a raw Reddit post dict to TopicSignalSchema."""
        if not isinstance(post, dict):
            return None

        post_id = post.get("id", "")
        title = post.get("title", "")
        if not post_id or not title:
            return None

        url = post.get("url", "")
        permalink = post.get("permalink", "")
        if not url and permalink:
            url = f"https://www.reddit.com{permalink}"

        score = int(post.get("score", 0))
        num_comments = int(post.get("num_comments", 0))
        upvote_ratio = float(post.get("upvote_ratio", 0.0))
        subreddit = post.get("subreddit", "")
        created_utc = _ts_to_dt(post.get("created_utc"))

        keywords = _extract_keywords(title)
        sentiment = _classify_sentiment(title)

        return TopicSignalSchema(
            id=post_id,
            title=title,
            score=score,
            url=url,
            subreddit=subreddit,
            created_utc=created_utc,
            num_comments=num_comments,
            upvote_ratio=upvote_ratio,
            keywords=keywords,
            sentiment=sentiment,
            source="reddit",
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _tool_get_hot_posts(
        self,
        subreddit: str = "MachineLearning",
        limit: int = 25,
    ) -> Any:
        """
        Fetch hot posts from a subreddit.

        Args:
            subreddit: Name of the subreddit (without the "r/" prefix).
            limit:     Maximum number of posts to return.

        Returns:
            Reddit listing dict with post data.
        """
        limit = min(limit, MAX_RESULTS_CAP)
        self._logger.info("get_hot_posts: subreddit=%r limit=%d", subreddit, limit)
        return await self._get(f"/r/{subreddit}/hot", params={"limit": limit})

    async def _tool_get_trending_topics(
        self,
        subreddits: Optional[List[str]] = None,
        time_filter: str = "week",
        limit: int = 10,
    ) -> Any:
        """
        Aggregate top posts from multiple subreddits to surface trending topics.

        Args:
            subreddits:  List of subreddit names. Defaults to the standard AI set.
            time_filter: Reddit time filter – "hour", "day", "week", "month", "year", "all".
            limit:       Posts per subreddit.

        Returns:
            Combined list of raw post dicts sorted by score descending.
        """
        if not subreddits:
            subreddits = DEFAULT_SUBREDDITS

        valid_filters = {"hour", "day", "week", "month", "year", "all"}
        time_filter = time_filter if time_filter in valid_filters else "week"
        limit = min(limit, MAX_RESULTS_CAP)

        self._logger.info(
            "get_trending_topics: subreddits=%s time_filter=%r limit=%d",
            subreddits,
            time_filter,
            limit,
        )

        tasks = [
            self._get(
                f"/r/{sr}/top",
                params={"limit": limit, "t": time_filter},
            )
            for sr in subreddits
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_posts: List[Dict[str, Any]] = []
        for sub, result in zip(subreddits, results):
            if isinstance(result, Exception):
                self._logger.warning("Failed to fetch r/%s: %s", sub, result)
                continue
            if isinstance(result, dict) and "data" in result:
                for child in result["data"].get("children", []):
                    all_posts.append(child.get("data", {}))

        # Sort by score descending
        all_posts.sort(key=lambda p: p.get("score", 0), reverse=True)
        return all_posts

    async def _tool_get_rising_posts(
        self,
        subreddit: str = "MachineLearning",
        limit: int = 25,
    ) -> Any:
        """
        Fetch rising posts from a subreddit.

        Args:
            subreddit: Name of the subreddit.
            limit:     Maximum number of posts to return.

        Returns:
            Reddit listing dict with post data.
        """
        limit = min(limit, MAX_RESULTS_CAP)
        self._logger.info("get_rising_posts: subreddit=%r limit=%d", subreddit, limit)
        return await self._get(f"/r/{subreddit}/rising", params={"limit": limit})

    async def _tool_analyze_sentiment(
        self, posts: List[Dict[str, Any]]
    ) -> List[TopicSignalSchema]:
        """
        Perform simple rule-based sentiment analysis on a list of post dicts.

        This tool operates entirely in-process (no network calls) and enriches
        each post with a sentiment label and extracted keywords.

        Args:
            posts: List of raw Reddit post dicts (as returned by the API).

        Returns:
            List of TopicSignalSchema objects with sentiment and keywords populated.
        """
        self._logger.info("analyze_sentiment: processing %d posts", len(posts))
        results: List[TopicSignalSchema] = []
        for post in posts:
            try:
                topic = self._post_to_topic(post)
                if topic:
                    results.append(topic)
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.warning("analyze_sentiment: skipping post: %s", exc)
        return results
