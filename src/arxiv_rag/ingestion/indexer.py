"""Orchestrate the ingestion pipeline: load → embed → upsert.

Papers are streamed from the corpus, embedded locally in batches, and upserted
into the vector store. Batching keeps embedding throughput high and bounds
memory; upserts are idempotent (deterministic point IDs), so re-running
ingestion refreshes rather than duplicates.
"""

from typing import Iterator, Iterable, List

from arxiv_rag.retrieval import Embedder, VectorStore
from arxiv_rag.ingestion.corpus import load_papers
from arxiv_rag.logging_config import get_logger
from arxiv_rag.config import Settings
from arxiv_rag.domain import Paper

logger = get_logger(__name__)


def _batched(
    papers: Iterator[Paper], batch_size: int
) -> Iterable[List[Paper]]:
    """Group a stream of papers into fixed-size batches.

    Args:
        papers: An iterator of papers.
        batch_size: The maximum number of papers per batch.

    Yields:
        Lists of papers of length up to ``batch_size``.
    """
    batch: List[Paper] = []
    for paper in papers:
        batch.append(paper)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest(
    settings: Settings,
    embedder: Embedder,
    store: VectorStore,
) -> int:
    """Run the full ingestion pipeline.

    Loads papers for the configured category, embeds them in batches, and
    upserts them into the vector store. The collection is created if missing.

    Args:
        settings: Configuration (corpus path, category, caps, batch size).
        embedder: The embedder used to vectorize papers.
        store: The vector store to upsert into.

    Returns:
        The total number of papers ingested.
    """
    store.ensure_collection(embedder.dim)

    papers = load_papers(
        path=settings.corpus_path,
        category=settings.arxiv_category,
        max_papers=settings.max_papers,
    )

    total = 0
    for batch in _batched(papers, settings.ingest_batch_size):
        vectors = embedder.embed_documents(
            [paper.embedding_text() for paper in batch]
        )
        store.upsert(batch, vectors)
        total += len(batch)
        logger.info("ingested %d papers...", total)

    return total
