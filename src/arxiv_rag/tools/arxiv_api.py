"""The arXiv API tool: live retrieval for current/structured queries.

This is the non-RAG path — it answers questions a static index cannot, such as
"what was published recently" or "papers by author X" (see ``docs/ADR.md``
section 1). It wraps the live arXiv API with a timeout and broad error
handling so that an API outage degrades to an empty result rather than
crashing the agent (graceful degradation, section 4.7).
"""

from typing import List, Any
import concurrent.futures

from arxiv_rag.logging_config import get_logger
from arxiv_rag.domain import Chunk

logger = get_logger(__name__)


class ArxivApiTool:
    """A ``Tool`` that retrieves live results from the arXiv API.

    Attributes:
        name: The routing label for this tool (``"arxiv"``).
    """

    name: str = "arxiv"

    def __init__(self, timeout: float = 10.0) -> None:
        """Construct the tool.

        Args:
            timeout: Maximum seconds to wait for an arXiv API response before
                giving up and returning no results.
        """
        self._timeout: float = timeout

    def _run_query(self, query: str, top_k: int) -> List[Chunk]:
        """Execute the live arXiv query (no timeout wrapper).

        Args:
            query: The query string passed to the arXiv API.
            top_k: Maximum number of results to fetch.

        Returns:
            The retrieved chunks, ordered as the API returned them.
        """
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=top_k,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return [
            self._result_to_chunk(result)
            for result in client.results(search)
        ]

    def search(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve live results from arXiv, with a timeout.

        The blocking arXiv call is run in a worker thread so a slow or hung
        request is bounded by ``timeout`` rather than stalling the agent. Any
        failure (timeout, network error, parse error) returns an empty list.

        Args:
            query: The query to retrieve for.
            top_k: Maximum number of results to return.

        Returns:
            Matching chunks (may be empty on any failure or timeout).
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                future = executor.submit(self._run_query, query, top_k)
                return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "query timed out after %ss", self._timeout
            )
            return []
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning("query failed: %s", exc)
            return []

    @staticmethod
    def _result_to_chunk(result: Any) -> Chunk:
        """Convert an ``arxiv.Result`` into a ``Chunk``.

        The live API does not return a similarity score, so the score is set
        to ``0.0``; the grader judges relevance for this path instead.

        Args:
            result: A result object from the ``arxiv`` library.

        Returns:
            The corresponding ``Chunk`` tagged with source ``"arxiv"``.
        """
        arxiv_id = result.get_short_id() if hasattr(
            result, "get_short_id"
        ) else str(getattr(result, "entry_id", ""))
        title = " ".join((result.title or "").split())
        abstract = " ".join((result.summary or "").split())
        published = (
            result.published.isoformat()
            if getattr(result, "published", None) is not None
            else None
        )
        return Chunk(
            arxiv_id=arxiv_id,
            title=title,
            text=f"{title}\n\n{abstract}".strip(),
            score=0.0,
            source="arxiv",
            authors=[a.name for a in getattr(result, "authors", [])],
            published=published,
        )
