"""Centralized configuration loaded from environment variables.

All tunable knobs for the system — model names, thresholds, paths, and
service endpoints — live here so that the rest of the codebase reads
configuration from a single typed object rather than touching ``os.environ``
directly. See ``docs/ADR.md`` for the reasoning behind the defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration sourced from environment variables / ``.env``.

    Attributes:
        anthropic_api_key: API key used for answer generation.
        synth_model: Model used for the user-facing synthesis step.
        grader_model: Model used for decomposition and relevance grading.
        embedding_model: Local sentence-transformers model name.
        embedding_dim: Dimensionality of the embedding model's vectors.
        corpus_path: Path to the Kaggle arXiv metadata JSON file.
        arxiv_category: arXiv category the corpus is scoped to.
        max_papers: Cap on the number of papers ingested.
        ingest_batch_size: Batch size for embedding and upserting.
        top_k: Number of chunks retrieved per sub-query.
        score_threshold: Minimum cosine score for a chunk to be kept.
        max_subqueries: Maximum sub-queries a question is split into.
        max_retries: Maximum corrective re-retrievals per sub-query.
        arxiv_timeout: Timeout, in seconds, for arXiv API requests.
        qdrant_url: Qdrant service endpoint.
        qdrant_collection: Name of the Qdrant collection.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    synth_model: str = Field(
        default="claude-sonnet-4-6", alias="SYNTH_MODEL"
    )
    grader_model: str = Field(
        default="claude-haiku-4-5", alias="GRADER_MODEL"
    )

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    corpus_path: str = Field(
        default="_data/arxiv-metadata-oai-snapshot.json",
        alias="CORPUS_PATH",
    )
    arxiv_category: str = Field(default="cs.LG", alias="ARXIV_CATEGORY")
    max_papers: int = Field(default=50_000, alias="MAX_PAPERS")
    ingest_batch_size: int = Field(default=256, alias="INGEST_BATCH_SIZE")

    top_k: int = Field(default=5, alias="TOP_K")
    score_threshold: float = Field(default=0.3, alias="SCORE_THRESHOLD")

    max_subqueries: int = Field(default=3, alias="MAX_SUBQUERIES")
    max_retries: int = Field(default=1, alias="MAX_RETRIES")
    arxiv_timeout: float = Field(default=10.0, alias="ARXIV_TIMEOUT")

    qdrant_url: str = Field(
        default="http://localhost:6333", alias="QDRANT_URL"
    )
    qdrant_collection: str = Field(
        default="arxiv_papers", alias="QDRANT_COLLECTION"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    Returns:
        The process-wide settings object, constructed once and reused.
    """
    return Settings()
