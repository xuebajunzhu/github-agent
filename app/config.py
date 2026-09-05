"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_list(value: object) -> object:
    """Accept JSON arrays, comma-separated strings, or real lists for list fields."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # -- App --
    app_name: str = "github-agent"
    debug: bool = False
    secret_key: str = "please-change-me"
    admin_token: str = "change-me-admin-token"
    # Public deployment guards
    rate_limit_per_minute: int = 120
    max_upload_mb: int = 10
    log_file: Optional[str] = "./logs/agent.log"

    # -- Database --
    database_url: str = "sqlite:///./github_agent.db"

    # -- GitHub OAuth / API --
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None
    github_redirect_uri: str = "http://127.0.0.1:8000/auth/github/callback"
    github_token: Optional[str] = None
    github_api_base: str = "https://api.github.com"
    github_search_keywords: List[str] = Field(
        default_factory=lambda: ["machine learning", "web framework", "developer tools"]
    )
    github_language: Optional[str] = None
    github_min_stars: int = 100
    github_per_page: int = 20
    github_max_repos_per_keyword: int = 20
    github_max_analyze_per_run: int = 30
    github_fetch_readme: bool = True
    github_readme_max_chars: int = 2000
    github_max_retries: int = 3
    github_backoff_seconds: float = 2.0

    # -- Security --
    fernet_key: Optional[str] = None  # Fernet key for encrypting stored GitHub tokens

    # -- LLM analysis --
    llm_provider: str = "openai"  # openai | anthropic | trained | local | rule
    llm_fallback_providers: List[str] = Field(
        default_factory=lambda: ["trained", "local", "rule"]
    )
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-haiku-latest"
    local_model_path: str = "microsoft/phi-2"
    trained_model_path: str = "./models/repo-analyzer"
    enable_llm_rerank: bool = False

    # -- Embedder --
    embedder_provider: str = "sentence_transformers"  # sentence_transformers | openai | trained | tfidf
    embedder_fallback_providers: List[str] = Field(
        default_factory=lambda: ["trained", "tfidf"]
    )
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_api_key: Optional[str] = None

    # -- Vector store --
    vector_store_backend: str = "chroma"  # chroma | memory
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "project_embeddings"

    # -- Scheduler / 24-7 autonomous loop --
    scheduler_enabled: bool = True
    fetch_interval_minutes: int = 1440
    refresh_interval_minutes: int = 360
    refresh_batch_size: int = 50
    # Each scheduled fetch consumes the next slice of the keyword list and
    # rotates, so a 24/7 deployment sweeps all keywords over time. 0 = all.
    keywords_per_fetch: int = 0
    # Patrol: short-interval self-healing sweep over the queue.
    patrol_enabled: bool = True
    patrol_interval_minutes: int = 10
    patrol_requeue_failed: bool = True
    patrol_reconcile_vectors: bool = True
    patrol_reconcile_limit: int = 2000
    # Parallel analysis threads (analysis is CPU/IO mixed; 1-4 sensible).
    analysis_workers: int = 2

    # -- Self-evolution (unattended retrain/promote/rollback loop) --
    evolution_enabled: bool = True
    evolution_interval_minutes: int = 1440
    evolution_train_epochs: int = 2
    evolution_min_new_samples: int = 200
    evolution_max_train_samples: int = 12000
    evolution_min_topic_freq: int = 40
    evolution_max_reembed: int = 500
    evolution_autorollback: bool = True

    # -- Speech (offline voice in/out) --
    asr_provider: str = "faster_whisper"  # faster_whisper | none
    asr_model_path: str = "./models/whisper-small"
    asr_compute_type: str = "int8"
    asr_language: Optional[str] = None  # None = auto-detect (zh/en)
    tts_provider: str = "sherpa_onnx"  # sherpa_onnx | none
    tts_model_dir: str = "./models/tts-melo-zh-en"
    tts_speaker_id: int = 0
    tts_speed: float = 1.0
    voice_reply_max_results: int = 3

    # -- CORS --
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator(
        "github_search_keywords",
        "llm_fallback_providers",
        "cors_origins",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: object) -> object:
        return _split_list(value)

    @property
    def is_github_oauth_configured(self) -> bool:
        return bool(self.github_client_id and self.github_client_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
