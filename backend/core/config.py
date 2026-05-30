"""
AI Research Intelligence Platform — Backend Core Configuration
Loads all settings from environment variables with validation.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = Field(default="AI Research Intelligence Platform")
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    debug: bool = Field(default=False)
    secret_key: str = Field(default="change-me-in-production-please")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+asyncpg://airesearch:strongpassword123@localhost:5432/ai_research_platform"
    )
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="ai_research_platform")
    postgres_user: str = Field(default="airesearch")
    postgres_password: str = Field(default="strongpassword123")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    cache_ttl: int = Field(default=3600)

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_collection_papers: str = Field(default="paper_embeddings")
    qdrant_collection_repos: str = Field(default="repo_embeddings")
    qdrant_collection_topics: str = Field(default="topic_embeddings")

    # --- Kafka ---
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_topic_ingestion: str = Field(default="ai-signals-ingestion")
    kafka_topic_rankings: str = Field(default="ai-signals-rankings")
    kafka_consumer_group: str = Field(default="ai-research-platform")

    # --- GitHub API ---
    github_token: Optional[str] = Field(default=None)
    github_api_url: str = Field(default="https://api.github.com")

    # --- Reddit API ---
    reddit_client_id: Optional[str] = Field(default=None)
    reddit_client_secret: Optional[str] = Field(default=None)
    reddit_user_agent: str = Field(default="AIResearchPlatform/1.0")

    # --- HuggingFace ---
    huggingface_api_key: Optional[str] = Field(default=None)
    huggingface_api_url: str = Field(default="https://huggingface.co/api")

    # --- PapersWithCode ---
    paperswithcode_api_url: str = Field(default="https://paperswithcode.com/api/v1")

    # --- ArXiv ---
    arxiv_api_url: str = Field(default="http://export.arxiv.org/api/query")
    arxiv_max_results: int = Field(default=50)

    # --- LLM ---
    llm_provider: str = Field(default="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")

    # --- Embeddings ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: int = Field(default=32)

    # --- Celery ---
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # --- Scheduling ---
    ingestion_interval_minutes: int = Field(default=30)
    ranking_update_interval_minutes: int = Field(default=15)
    forecast_update_interval_hours: int = Field(default=6)

    # --- CORS ---
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a recognized value."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
