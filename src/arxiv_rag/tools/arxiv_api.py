"""The arXiv API tool: live retrieval for current/structured queries.

This is the non-RAG path — it answers questions a static index cannot, such as
"what was published recently" or "papers by author X" (see ``docs/ADR.md``
section 1). It wraps the live arXiv API with a timeout and broad error
handling so that an API outage degrades to an empty result rather than
crashing the agent (graceful degradation, section 4.7).
"""

from typing import List, Any, Optional, Tuple
import concurrent.futures

from arxiv_rag.logging_config import get_logger
from arxiv_rag.domain import Chunk, ArxivQuery

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

    def _run_query(
        self,
        query: str,
        top_k: int,
        id_list: Optional[List[str]] = None,
        sort_by_recency: bool = False,
    ) -> List[Chunk]:
        """Execute the live arXiv query (no timeout wrapper).

        Args:
            query: The query string passed to the arXiv API (may use field
                prefixes such as ``au:`` or ``submittedDate:``).
            top_k: Maximum number of results to fetch.
            id_list: Specific arXiv ids to fetch; when given, the API returns
                exactly those papers.
            sort_by_recency: Sort by submission date (newest first) instead of
                relevance — used for "latest papers" style queries.

        Returns:
            The retrieved chunks, ordered as the API returned them.
        """
        import arxiv

        sort_by = (
            arxiv.SortCriterion.SubmittedDate
            if sort_by_recency
            else arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            id_list=id_list or [],
            max_results=top_k,
            sort_by=sort_by,
        )
        return [
            self._result_to_chunk(result)
            for result in client.results(search)
        ]

    def search(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve live results from arXiv by keyword, with a timeout.

        This is the plain ``Tool`` contract: a relevance keyword search. For
        author/id/recency lookups the agent calls ``structured_search``.

        Args:
            query: The query to retrieve for.
            top_k: Maximum number of results to return.

        Returns:
            Matching chunks (may be empty on any failure or timeout).
        """
        return self._search_bounded(query, top_k)

    def structured_search(
        self, arxiv_query: ArxivQuery, top_k: int
    ) -> List[Chunk]:
        """Retrieve live results using structured intent, with a timeout.

        Builds a field-scoped arXiv query from ``arxiv_query`` (e.g.
        ``au:LeCun`` and ``submittedDate:[20250101 TO 20251231]``) so author,
        id, and recency lookups hit the right results rather than degrading to
        a keyword match (see ``docs/ADR.md`` section 4.7).

        Args:
            arxiv_query: The structured lookup extracted during decomposition.
            top_k: Maximum number of results to return.

        Returns:
            Matching chunks (may be empty on any failure or timeout).
        """
        query, id_list, sort_by_recency = self._build_query(arxiv_query)
        return self._search_bounded(
            query, top_k, id_list=id_list, sort_by_recency=sort_by_recency
        )

    def _search_bounded(
        self,
        query: str,
        top_k: int,
        id_list: Optional[List[str]] = None,
        sort_by_recency: bool = False,
    ) -> List[Chunk]:
        """Run a query in a worker thread, bounded by ``timeout``.

        The blocking arXiv call is run off-thread so a slow or hung request is
        bounded rather than stalling the agent. Any failure (timeout, network
        error, parse error) returns an empty list (graceful degradation).

        Args:
            query: The query string passed to the arXiv API.
            top_k: Maximum number of results to return.
            id_list: Specific arXiv ids to fetch, if any.
            sort_by_recency: Whether to sort newest-first.

        Returns:
            Matching chunks (may be empty on any failure or timeout).
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                future = executor.submit(
                    self._run_query,
                    query,
                    top_k,
                    id_list,
                    sort_by_recency,
                )
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
    def _build_query(
        arxiv_query: ArxivQuery,
    ) -> Tuple[str, Optional[List[str]], bool]:
        """Translate structured intent into an arXiv query string.

        Maps the ``query_type`` to arXiv's field-prefix syntax: ``id`` becomes
        an ``id_list`` lookup; ``author`` adds an ``au:`` clause; year bounds
        add a ``submittedDate:[...]`` range; and ``recent`` flags newest-first
        sorting. Topical ``terms`` are AND-ed in as a free-text clause.

        Args:
            arxiv_query: The structured lookup to translate.

        Returns:
            A ``(query, id_list, sort_by_recency)`` triple for ``_run_query``.
            ``id_list`` is ``None`` unless this is an id lookup.
        """
        if arxiv_query.query_type == "id" and arxiv_query.ids:
            return "", list(arxiv_query.ids), False

        clauses: List[str] = []
        if arxiv_query.author:
            clauses.append(f'au:"{arxiv_query.author}"')
        if arxiv_query.terms:
            clauses.append(f"all:{arxiv_query.terms}")

        date_clause = ArxivApiTool._date_clause(
            arxiv_query.start_year, arxiv_query.end_year
        )
        if date_clause:
            clauses.append(date_clause)

        query = " AND ".join(clauses)
        # Sort newest-first for explicit recency queries and for any
        # date-bounded lookup (e.g. "papers by X in 2025"), where the user is
        # asking about a time window rather than topical relevance.
        sort_by_recency = (
            arxiv_query.query_type == "recent"
            or arxiv_query.start_year is not None
            or arxiv_query.end_year is not None
        )
        return query, None, sort_by_recency

    @staticmethod
    def _date_clause(
        start_year: Optional[int], end_year: Optional[int]
    ) -> Optional[str]:
        """Build a ``submittedDate`` range clause from year bounds.

        Args:
            start_year: Earliest submission year, if bounded.
            end_year: Latest submission year, if bounded.

        Returns:
            A ``submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]`` clause, or
            ``None`` if neither bound is given.
        """
        if start_year is None and end_year is None:
            return None
        low = start_year if start_year is not None else end_year
        high = end_year if end_year is not None else start_year
        start = f"{low:04d}0101000000"
        end = f"{high:04d}1231235959"
        return f"submittedDate:[{start} TO {end}]"

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
