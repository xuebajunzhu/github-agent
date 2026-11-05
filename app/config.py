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
    # Trust CF-Connecting-IP / X-Forwarded-For (loopback clients only) when
    # deployed behind a tunnel/reverse proxy, so limits apply per real client.
    trust_proxy_headers: bool = False
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
    # Give up on a project after this many failed analysis attempts (patrol
    # keeps requeueing failures; without a cap a permanently failing project
    # retries forever and can starve the queue).
    analysis_max_attempts: int = 5

    # -- Security --
    fernet_key: Optional[str] = None  # Fernet key for encrypting stored GitHub tokens
    # Optional Redis (REDIS_URL, e.g. redis://localhost:6379/0). When empty,
    # every piece of infrastructure runs on in-process defaults (rate limiter,
    # job locks, caches) with zero extra services; when set, they become
    # shared across processes and survive restarts.
    redis_url: Optional[str] = None
    # Optional bearer token protecting the MCP endpoint (/mcp). Unset = open
    # (rate-limited); set it before exposing MCP publicly.
    mcp_auth_token: Optional[str] = None

    # -- LLM analysis --
    llm_provider: str = "openai"  # openai | anthropic | trained | local | rule
    llm_fallback_providers: List[str] = Field(
        default_factory=lambda: ["trained", "local", "rule"]
    )
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # Circuit breaker (Design-for-Reliability): after this many consecutive
    # failures an analyzer stops being called for the cooldown window -- a
    # revoked key or exhausted quota must not pay the retry tax on every call.
    llm_breaker_threshold: int = 3
    llm_breaker_cooldown_seconds: int = 300
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-haiku-latest"
    # DeepSeek (OpenAI-compatible endpoint; no embeddings API, so it is LLM-only)
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    local_model_path: str = "microsoft/phi-2"
    trained_model_path: str = "./models/repo-analyzer"
    enable_llm_rerank: bool = False
    # Which reranker to use when enable_llm_rerank is on:
    #   llm  = the primary analyzer (e.g. DeepSeek) rewrites ranking + reason
    #   bge  = local cross-encoder bge-reranker-v2-m3 (zero cost, offline,
    #          strong relevance ordering, no natural-language reason)
    #   laya = local 421M decision model (zero cost, offline; weakly
    #          discriminative on this domain -- experimental)
    rerank_provider: str = "llm"  # llm | bge | laya
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    # Hybrid retrieval: fuse dense (BGE-M3) ranking with a BM25 lexical
    # ranking via Reciprocal Rank Fusion -- exact keyword hits that embeddings
    # blur get pulled back in. Zero extra dependencies.
    hybrid_retrieval: bool = True

    # -- Feedback loop (external votes actually change behavior) --
    # Downvoted projects are multiplied by (1 - penalty * down_ratio), i.e. a
    # project rejected by every voter loses at most this fraction of its
    # score; upvoted projects gain up to `feedback_upvote_bonus`.
    feedback_in_ranking: bool = True
    feedback_penalty_weight: float = 0.25
    feedback_upvote_bonus: float = 0.10
    # Projects with >= threshold downvotes get re-analyzed by the patrol
    # (fresh pipeline, e.g. after README fetching was enabled).
    patrol_requeue_downvote_threshold: int = 2
    patrol_requeue_downvote_limit: int = 10
    # Recent zero-result queries are expanded into extra GitHub search
    # keywords (via the primary LLM, done by the GapScout job) and joined
    # into the rotating keyword list by the ingest pipeline.
    dynamic_keyword_limit: int = 8
    dynamic_keywords_file: str = "./dynamic_keywords.json"
    gap_scout_enabled: bool = True
    gap_scout_interval_minutes: int = 60
    gap_scout_max_queries: int = 10

    # -- Notifications (optional webhook on patrol/evolution failure, model
    # promotion/rollback). Auto-detects WeCom bot / ServerChan / Telegram /
    # generic JSON by URL. Empty = disabled.
    notify_webhook_url: Optional[str] = None
    notify_telegram_chat_id: Optional[str] = None
    notify_cooldown_minutes: int = 30

    # -- Backups (SQLite via the backup API + chroma dir tarball) --
    backup_enabled: bool = True
    backup_hour_utc: int = 19  # 03:00 Beijing time
    backup_retention: int = 7
    backup_dir: str = "./backups"

    # -- Embedder --
    embedder_provider: str = "sentence_transformers"  # sentence_transformers | openai | trained | tfidf
    embedder_fallback_providers: List[str] = Field(
        default_factory=lambda: ["trained", "tfidf"]
    )
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    # BGE-M3 (EMBEDDER_PROVIDER=bge-m3): 1024-dim multilingual dense embeddings
    bge_m3_model_name: str = "BAAI/bge-m3"
    bge_m3_max_len: int = 1024
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
    # query_history only grows; patrol prunes rows older than this many days.
    query_history_retention_days: int = 180

    # Problem research (POST /api/research): problem -> direct solutions +
    # similar/substitutable projects. Costs several LLM calls + GitHub quota.
    research_max_candidates: int = 40
    research_cache_hours: int = 24
    research_min_stars: int = 30
    research_keywords: int = 4
    # Parallel analysis threads (analysis is CPU/IO mixed; 1-4 sensible).
    analysis_workers: int = 2

    # -- Self-evolution (unattended retrain/promote/rollback loop) --
    evolution_enabled: bool = True
    evolution_interval_minutes: int = 1440
    evolution_train_epochs: int = 2
    evolution_train_timeout_seconds: int = 7200  # hard stop for the training subprocess
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
