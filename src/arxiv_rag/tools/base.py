"""The ``Tool`` protocol shared by the retrieval tools.

Both retrieval paths — local vector search and the live arXiv API — expose the
same minimal contract so the agent can route a sub-query to either without
caring which one it is (see ``docs/ADR.md`` section 4.7).
"""

from typing import Protocol, List

from arxiv_rag.domain import Chunk


class Tool(Protocol):
    """Protocol for a retrieval tool that turns a query into context.

    Attributes:
        name: A short identifier matching the routing label
            (``"vector"`` or ``"arxiv"``).
    """

    name: str

    def search(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve context for a query.

        Implementations degrade gracefully: a failure returns an empty list
        rather than raising, so a single tool fault does not crash a query.

        Args:
            query: The (sub-)query to retrieve for.
            top_k: Maximum number of results to return.

        Returns:
            Retrieved chunks, ordered by descending relevance (may be empty).
        """
        ...
