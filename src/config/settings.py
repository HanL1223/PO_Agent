"""
Configuration Managemebnt with Pydantic setting

Pydantic setting validation environment variables at application start up

We group related settings into sub-classes (LLMSettings, JiraSettings, etc.)

"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field,field_validator
from pydantic_settings import BaseSettings,SettingsConfigDict

logger = logging.getLogger(__name__)


#Setting subclass
#Each sbuclass groups realted configuration with its own env_prefix
#This prevents naming collisions

class LLMSettings(BaseSettings):
    """
    Configurateion for the LLM provider - using Claude here

    ENV VARS
    LLM_MODEL=claude-sonnet-4-6
    LLM_TEMPERATURE=0.3
    LLM_MAX_TOKENS=4096
    """

    model_config = SettingsConfigDict(env_prefix = "LLM_",extra = "ignore")

    model: str = Field(
        default = "claude-sonnet-4-6", #using 4.6
        description="Anthropic model identifier. See https://docs.anthropic.com/en/docs/models",
    )
    temperature:float = Field(
        default = 0.3,
        ge=0.0,
        le= 1.0,
        description=(
            "Controls randomness. Lower = more deterministic."
        ),
    )
    max_tokens: int = Field(
        default = 4096,
        ge=256,
        le=8192,
        description = "Maximun tokens in the genereated response"
    )


class EmbeddingSettings(BaseSettings):
    """
    Configuration for the embedding model.

    We use a HuggingFace model (BAAI/bge-small-en-v1.5)
    instead of a cloud API for embeddings. Reasons:
      1. No per-request cost (runs locally or on the server)
      2. Faster inference (no network round-trip)
      3. Deterministic (same input always produces same embedding)
      4. Privacy (document content never leaves your infrastructure)

    The bge-small model has 384 dimensions and excellent quality for its size.
    For higher quality, upgrade to bge-base (768 dims) or bge-large (1024 dims).

    ENV VARS:
        EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
        EMBEDDING_DIMENSION=384
    """

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model ID for embeddings.",
    )
    dimension: int = Field(
        default=384,
        description="Embedding vector dimension. Must match the model.",
    )

class ChromaSettings(BaseSettings):
    """
    Configuration for ChromaDB vector store.

    EDUCATIONAL NOTE: ChromaDB runs in-process (embedded mode) which means
    no separate server is needed. Data persists to disk at persist_directory.
    In production on AWS, this directory would be an EFS mount for durability.

    ENV VARS:
        CHROMA_PERSIST_DIR=./data/chroma
        CHROMA_COLLECTION_PREFIX=jira
    """

    model_config = SettingsConfigDict(env_prefix="CHROMA_", extra="ignore")

    persist_dir: str = Field(
        default="./data/chroma",
        description="Directory for ChromaDB persistence.",
    )
    collection_prefix: str = Field(
        default="jira",
        description="Prefix for collection names. Collections are named {prefix}_{space}.",
    )

class JiraSettings(BaseSettings):
    """
    Configuration for Jira Cloud API intergration

    The 'spaces' field is a comma-separated list of Jira project keys.
    Each space gets its own vector index for retrieval, but the agent
    can search across all spaces or filter to a specific one.

    ENV VARS:
        JIRA_BASE_URL=https://company.atlassian.net
        JIRA_EMAIL=user@company.com
        JIRA_API_TOKEN=...
        JIRA_SPACES=CSCI,EDP
    """

    model_config = SettingsConfigDict(env_prefix = "JIRA_",extra = "ignore")

    base_url: str = Field(default="", description="Jira Cloud instance URL.")
    email: str = Field(default="", description="Atlassian account email.")
    api_token: str = Field(default="", description="Jira API token (not password).")
    spaces: str = Field(
        default="CSCI",
        description="Comma-separated Jira project keys to index.Default to F06 project",
    )

    @property
    def space_list(self) -> list[str]:
        return [s.strip().upper() for s in self.spaces.split(",") if s.strip()]
    
    @property
    def is_configured(self) -> bool:
        """Check if Jira integration has all required fields."""
        return bool(self.base_url and self.email and self.api_token)

class RAGSettings(BaseSettings):
    """
    Configuration for RAG retrieval behaviour

    These settings control the hybrid retrieval pipeline.
    - top_k: How many chunks to retrieve from each index
    - rrf_k: The k parameter in Reciprocal Rank Fusion (higher = less weight to rank)
    - dense_weight / sparse_weight: Relative importance of each retrieval method
    - chunk_size: Characters per chunk during indexing

    Tuning note:
    - Start with defaults and measure retrieval quality
    - If missing keyword matches: increase sparse_weight
    - If too many irrelevant results: decrease top_k or increase score_threshold
    - chunk_size 1024 works well for Jira tickets (1 section per chunk typically)

    ENV VARS:
        RAG_TOP_K=5
        RAG_CHUNK_SIZE=1024
        RAG_CHUNK_OVERLAP=128
    """
    model_config = SettingsConfigDict(prefix = "RAG_",extra = "ingore")

    top_K:int = Field(default = 5,ge = 1 , le=20)
    chunk_size: int = Field(default = 1024, ge = 256,le = 4096)
    chunk_overlap: int = Field(default = 128,ge = 0,le = 512)
    rrf_k:int =  Field(
        default = 60,
        description="RRF constant. Standard value is 60. Higher = less aggressive fusion.",
    )
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score. None = no filtering.",
    )

class CacheSettings(BaseSettings):
    """
    Configuration for the semantic cache layer

    Avoiding redundant LLM calls by storing generation results keyed by query embedidng
    if a sufficiently simliar queyr arrived, we return cached results instead of calling LLM

    BACKENDS:
    - 'memory': In-process dict cache (good for development, lost on restart)
    - 'disk': diskcache-based persistent cache (survives restarts)
    - 'redis': Distributed cache (for multi-instance deployments)

    ENV VARS:
        CACHE_BACKEND=disk
        CACHE_TTL_SECONDS=3600
    """

    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")

    backend: str = Field(default="disk", pattern="^(memory|disk|redis)$")
    ttl_seconds: int = Field(default=3600, ge=60)
    redis_url: str = Field(default="redis://localhost:6379/0")
    disk_dir: str = Field(default="./data/cache")


class ServerSettings(BaseSettings):
    """
    Configuration for the FastAPI server.

    ENV VARS:
        SERVER_HOST=0.0.0.0
        SERVER_PORT=8000
        SERVER_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
    """

    model_config = SettingsConfigDict(env_prefix="SERVER_", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173")
    env: str = Field(default="development", pattern="^(development|staging|production)$")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.env == "production"
    

#Main settings class

class Settings(BaseSettings):
    """
    
    Root configuration that aggreates all sub settings
    Use as a single entry point for all configuration e.g.
    ANTHROPIC_API_KEY -> settings.anthropic_api_key
    LLM_MODEL        -> settings.llm.model
    JIRA_BASE_URL    -> settings.jira.base_url
    RAG_TOP_K        -> settings.rag.top_k
    """
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore",
        case_sensitive = False
    )

    #API KEY
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude.")

    #Subsettings
    #as per above classes
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    jira: JiraSettings = Field(default_factory=JiraSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    #Feature flag
    #allows gradual roolout of capabilities
    # In production, it could be backed by a feature flag service (LaunchDarkly, etc.)

    enable_hybrid_retrieva: bool = Field(
        default = False,
        description="Use BM25 + dense fusion. False = dense only.",
    )
    enable_cache: bool = Field(default=True, description="Enable semantic caching.")

    enable_evaluation: bool = Field(
        default=False,
        description="Run evaluation on each generation (adds latency).",
    )

    #Compute properties
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    # --- Validation ---
    @field_validator("anthropic_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Warn if API key is not set. Do not fail - it may be set later."""
        if not v or v.startswith("sk-ant-placeholder"):
            import warnings

            warnings.warn(
                "ANTHROPIC_API_KEY is not set. LLM calls will fail. "
                "Set it in .env or as an environment variable.",
                stacklevel=2,
            )
        return v
    


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the cached singleton Settings instance.

    lru_cache(maxsize=1) ensures this function only creates
    one Settings object. Every subsequent call returns the same instance.
    This is Python's simplest singleton pattern.

    For testing, call get_settings.cache_clear() to reset.
    """
    return Settings()


# Module-level convenience instance
settings = get_settings()
