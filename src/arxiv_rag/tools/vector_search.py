"""The vector-search tool: semantic retrieval over the local index.

This is the RAG path. It embeds the query locally and searches the vector
store, optionally constraining results to the configured category via Qdrant
payload filtering.
"""

from typing import List, Optional

from arxiv_rag.retrieval import Embedder, VectorStore
from arxiv_rag.domain import Chunk


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
            print(f"[vector_search] retrieval failed: {exc}", flush=True)
            return []
