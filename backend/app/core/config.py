import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "InsightCore"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"

    # MySQL
    MYSQL_SERVER: str = os.getenv("MYSQL_SERVER", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "ragwebui")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "ragwebui")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "ragwebui")
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @property
    def get_database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        return (
            f"mysql+mysqlconnector://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

    # File storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")

    # LLM + Embeddings (OpenAI-compatible)
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "lmstudio")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "local-model")
    OPENAI_EMBEDDINGS_MODEL: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "local-embedding-model")
    # Dimension of the dense embedding model output. Must match OPENAI_EMBEDDINGS_MODEL.
    # qwen3-embedding-0.6b = 1024, text-embedding-3-small = 1536, text-embedding-ada-002 = 1536
    DENSE_EMBEDDING_DIM: int = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

    # Qdrant vector store
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

    # SPLADE sparse embedding model (FastEmbed / ONNX — CPU-optimised)
    SPLADE_MODEL: str = os.getenv("SPLADE_MODEL", "prithivida/Splade_PP_en_v1")
    # Directory where FastEmbed caches downloaded ONNX models.
    # Mount as a volume so the model survives container restarts.
    FASTEMBED_CACHE_DIR: str = os.getenv("FASTEMBED_CACHE_DIR", "/tmp/fastembed_cache")

    # ── Retrieval ──────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))

    # ── Chunking ────────────────────────────────────────────────────────────────
    # WARNING: changing these values after documents have been ingested creates
    # inconsistent chunk sizes across the knowledge base. If you change them,
    # delete and re-upload all existing documents to re-index with the new settings.
    #
    # CHUNK_SIZE: target chunk size in characters. Keep <= 1800 chars when using
    # SPLADE (prithivida/Splade_PP_en_v1) — BERT's 512-token limit means longer
    # chunks are silently truncated in the sparse leg (~4 chars/token for English).
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))
    # OVERLAP_PERCENTAGE: fraction of CHUNK_SIZE repeated at the start of the next
    # chunk (0.0–1.0). 0.20 = 20% overlap = 300 chars at CHUNK_SIZE=1500.
    OVERLAP_PERCENTAGE: float = float(os.getenv("OVERLAP_PERCENTAGE", "0.20"))

    @property
    def chunk_overlap(self) -> int:
        return int(self.CHUNK_SIZE * self.OVERLAP_PERCENTAGE)

    # RRF weights for each leg. Weights don't need to sum to 1; they are
    # relative multipliers on the RRF term 1/(k + rank).
    HYBRID_DENSE_WEIGHT: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5"))
    HYBRID_QDRANT_SPARSE_WEIGHT: float = float(os.getenv("HYBRID_QDRANT_SPARSE_WEIGHT", "0.3"))
    HYBRID_EXACT_WEIGHT: float = float(os.getenv("HYBRID_EXACT_WEIGHT", "0.2"))

    # Per-leg retrieval enable/disable.
    # Affects retrieval ONLY — ingestion always indexes all three pipelines
    # so re-enabling a leg later requires no re-indexing.
    RETRIEVAL_DENSE_ENABLED: bool = os.getenv("RETRIEVAL_DENSE_ENABLED", "true").lower() == "true"
    RETRIEVAL_QDRANT_SPARSE_ENABLED: bool = os.getenv("RETRIEVAL_QDRANT_SPARSE_ENABLED", "true").lower() == "true"
    RETRIEVAL_EXACT_ENABLED: bool = os.getenv("RETRIEVAL_EXACT_ENABLED", "true").lower() == "true"

    class Config:
        env_file = ".env"


settings = Settings()
