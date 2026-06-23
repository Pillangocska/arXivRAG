"""The vector-search tool: semantic retrieval over the local index.

This is the RAG path. It embeds the query locally and searches the vector
store, optionally constraining results to the configured category via Qdrant
payload filtering.
"""

from typing import List, Optional

from arxiv_rag.retrieval import Embedder, VectorStore
from arxiv_rag.logging_config import get_logger
from arxiv_rag.domain import Chunk

logger = get_logger(__name__)


def _is_connection_error(exc: BaseException) -> bool:
    """Report whether an exception chain stems from a failed connection.

    Qdrant being unreachable surfaces as a ``ConnectionError`` (e.g.
    ``ConnectionRefusedError`` / ``WinError 10061``) somewhere in the cause
    chain, often wrapped by the Qdrant client. Walking the chain lets the
    caller distinguish "the store is down" from other retrieval failures.

    Args:
        exc: The exception raised during retrieval.

    Returns:
        ``True`` if a ``ConnectionError`` appears in the exception chain.
    """
    seen: set = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ConnectionError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class VectorSearchTool:
    """A ``Tool`` that retrieves from the local vector store.

    Attributes:
        name: The routing label for this tool (``"vector"``).
    """

    name: str = "vector"

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        score_threshold: float = 0.0,
        category: Optional[str] = None,
    ) -> None:
        """Construct the tool.

        Args:
            embedder: The embedder used to vectorize queries.
            store: The vector store to search.
            score_threshold: Minimum cosine score for a result to be kept.
            category: Optional category filter applied to all searches.
        """
        self._embedder: Embedder = embedder
        self._store: VectorStore = store
        self._score_threshold: float = score_threshold
        self._category: Optional[str] = category

    def search(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve the most similar papers to a query.

        Args:
            query: The query to retrieve for.
            top_k: Maximum number of results to return.

        Returns:
            Matching chunks ordered by descending score; empty if the store
            is unreachable or returns nothing.
        """
        try:
            vector = self._embedder.embed_query(query)
            return self._store.search(
                vector=vector,
                top_k=top_k,
                score_threshold=self._score_threshold,
                category=self._category,
            )
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            if _is_connection_error(exc):
                logger.warning(
                    "vector store is unreachable - is Qdrant running? "
                    "Start it with `docker compose up -d qdrant` (or "
                    "`make up`). Falling back to no vector results."
                )
            else:
                logger.warning("retrieval failed: %s", exc)
            return []
