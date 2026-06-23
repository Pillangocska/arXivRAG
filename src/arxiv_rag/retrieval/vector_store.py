"""Vector storage behind a ``VectorStore`` protocol, backed by Qdrant.

Qdrant is chosen for first-class payload filtering (category, date, author)
alongside semantic search, and for local→prod parity (see ``docs/ADR.md``
section 4.4). The protocol keeps the agent and tools decoupled from Qdrant so
the store can be swapped without touching callers.
"""

from typing import Protocol, Sequence, Optional, List, Dict, Any
import uuid

from arxiv_rag.domain import Chunk, Paper


def _stable_point_id(arxiv_id: str) -> str:
    """Derive a deterministic Qdrant point ID from an arXiv ID.

    Qdrant point IDs must be UUIDs or unsigned integers, so the (string) arXiv
    ID is hashed into a deterministic UUID. Determinism makes upserts
    idempotent: re-ingesting the same paper overwrites its point rather than
    creating a duplicate.

    Args:
        arxiv_id: The arXiv identifier of the paper.

    Returns:
        A deterministic UUIDv5 string for use as a Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"arxiv:{arxiv_id}"))


class VectorStore(Protocol):
    """Protocol for a semantic store with metadata filtering."""

    def ensure_collection(self, dim: int) -> None:
        """Create the collection if it does not already exist.

        Args:
            dim: The dimensionality of vectors the collection will hold.
        """
        ...

    def upsert(
        self, papers: Sequence[Paper], vectors: Sequence[List[float]]
    ) -> None:
        """Insert or update papers and their vectors.

        Args:
            papers: The papers to store (their fields become payload).
            vectors: One vector per paper, in the same order.
        """
        ...

    def search(
        self,
        vector: List[float],
        top_k: int,
        score_threshold: float = 0.0,
        category: Optional[str] = None,
    ) -> List[Chunk]:
        """Search for the most similar papers to a query vector.

        Args:
            vector: The query embedding.
            top_k: Maximum number of results to return.
            score_threshold: Minimum score for a result to be returned.
            category: Optional category to restrict results to.

        Returns:
            Matching chunks, ordered by descending score.
        """
        ...

    def count(self) -> int:
        """Return the number of points currently in the collection.

        Returns:
            The point count, or ``0`` if the collection does not exist.
        """
        ...


class QdrantVectorStore:
    """A ``VectorStore`` backed by a Qdrant collection.

    Attributes:
        collection: The Qdrant collection name.
    """

    def __init__(self, url: str, collection: str) -> None:
        """Connect to Qdrant.

        Args:
            url: The Qdrant service endpoint.
            collection: The collection name to read from and write to.
        """
        from qdrant_client import QdrantClient

        self._client: QdrantClient = QdrantClient(url=url)
        self.collection: str = collection

    def ensure_collection(self, dim: int) -> None:
        """Create the collection with cosine distance if it is missing.

        Args:
            dim: The dimensionality of vectors the collection will hold.
        """
        from qdrant_client.models import VectorParams, Distance

        if self._client.collection_exists(self.collection):
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dim, distance=Distance.COSINE
            ),
        )

    def upsert(
        self, papers: Sequence[Paper], vectors: Sequence[List[float]]
    ) -> None:
        """Insert or update papers and their vectors in one batch.

        Args:
            papers: The papers to store.
            vectors: One vector per paper, in the same order.

        Raises:
            ValueError: If ``papers`` and ``vectors`` differ in length.
        """
        from qdrant_client.models import PointStruct

        if len(papers) != len(vectors):
            raise ValueError(
                "papers and vectors must have the same length "
                f"({len(papers)} != {len(vectors)})."
            )
        points = [
            PointStruct(
                id=_stable_point_id(paper.arxiv_id),
                vector=vector,
                payload=_paper_to_payload(paper),
            )
            for paper, vector in zip(papers, vectors)
        ]
        self._client.upsert(
            collection_name=self.collection, points=points
        )

    def search(
        self,
        vector: List[float],
        top_k: int,
        score_threshold: float = 0.0,
        category: Optional[str] = None,
    ) -> List[Chunk]:
        """Search the collection for the nearest papers to a query vector.

        Args:
            vector: The query embedding.
            top_k: Maximum number of results to return.
            score_threshold: Minimum score for a result to be returned.
            category: Optional category to restrict results to.

        Returns:
            Matching chunks, ordered by descending score.
        """
        from qdrant_client.models import (
            FieldCondition,
            MatchValue,
            Filter,
        )

        query_filter: Optional[Filter] = None
        if category is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="categories",
                        match=MatchValue(value=category),
                    )
                ]
            )

        hits = self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        return [_hit_to_chunk(hit) for hit in hits]

    def count(self) -> int:
        """Return the number of points in the collection.

        Returns:
            The point count, or ``0`` if the collection does not exist.
        """
        if not self._client.collection_exists(self.collection):
            return 0
        return self._client.count(self.collection).count


def _paper_to_payload(paper: Paper) -> Dict[str, Any]:
    """Convert a ``Paper`` into a Qdrant payload dict.

    Args:
        paper: The paper to convert.

    Returns:
        A payload dict carrying the paper's filterable and displayable fields.
    """
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "categories": paper.categories,
        "published": paper.published,
    }


def _hit_to_chunk(hit: Any) -> Chunk:
    """Convert a Qdrant search hit into a ``Chunk``.

    Args:
        hit: A Qdrant ``ScoredPoint`` returned from a search.

    Returns:
        The corresponding ``Chunk`` with its relevance score attached.
    """
    payload = hit.payload or {}
    title = payload.get("title", "")
    abstract = payload.get("abstract", "")
    return Chunk(
        arxiv_id=payload.get("arxiv_id", ""),
        title=title,
        text=f"{title}\n\n{abstract}".strip(),
        score=float(hit.score),
        source="vector",
        authors=payload.get("authors", []),
        published=payload.get("published"),
    )
