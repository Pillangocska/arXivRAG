"""LangGraph node functions and their parsing helpers.

Each node is a pure-ish function from ``AgentState`` to a partial state update.
The nodes are grouped on an ``AgentNodes`` object that holds the agent's
dependencies (LLM client, tools, settings) so the graph wiring stays free of
globals and the whole agent is constructible with mocked dependencies in tests.

The corrective-retry bound lives in ``should_retry`` (consumed by the graph's
conditional edge), which is the test that proves the loop terminates (see
``docs/ADR.md`` sections 3 and 6).
"""

from typing import Dict, List, Any, Iterator, Optional
from contextlib import contextmanager
import time

from pydantic import ValidationError

from arxiv_rag.logging_config import get_logger
from arxiv_rag.llm.prompts import (
    build_synthesize_user,
    build_reformulate_user,
    build_decompose_user,
    build_grade_user,
    SYNTHESIZE_SYSTEM,
    REFORMULATE_SYSTEM,
    DECOMPOSE_SYSTEM,
    DECOMPOSE_SCHEMA,
    GRADE_SYSTEM,
    GRADE_SCHEMA,
)
from arxiv_rag.domain import AgentState, ArxivQuery, SubQuery, Chunk
from arxiv_rag.llm import LLMClient
from arxiv_rag.config import Settings
from arxiv_rag.tools import Tool

logger = get_logger(__name__)


@contextmanager
def _timed(stage: str, detail: str = "") -> Iterator[None]:
    """Log the start and elapsed time of a pipeline stage.

    Emits one ``INFO`` line when the stage begins and another when it ends,
    the latter carrying the wall-clock duration in seconds. Keeps per-stage
    timing logging to a single ``with`` block at each call site.

    Args:
        stage: The stage name shown in the log (e.g. ``"decompose"``).
        detail: Optional context appended to the start line (e.g. the
            sub-query being routed).

    Yields:
        Control to the wrapped block; timing is logged on exit.
    """
    suffix = f" ({detail})" if detail else ""
    logger.info("%s ...%s", stage, suffix)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s done in %.2fs", stage, elapsed)


def parse_subqueries(
    payload: Dict[str, Any], max_subqueries: int
) -> List[SubQuery]:
    """Parse the decomposition LLM output into ``SubQuery`` objects.

    Defensive against a model that returns too many sub-queries or an invalid
    route: the list is capped, and any route other than ``"arxiv"`` falls back
    to ``"vector"`` (the safe default, since the local index always exists).

    Args:
        payload: The JSON object returned by the decomposition call.
        max_subqueries: The configured cap on sub-query count.

    Returns:
        A non-empty list of parsed sub-queries (capped at ``max_subqueries``).
        Returns an empty list if the payload contains no usable sub-queries.
    """
    raw = payload.get("sub_queries", [])
    sub_queries: List[SubQuery] = []
    for item in raw[:max_subqueries]:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        route = item.get("route")
        route = route if route == "arxiv" else "vector"
        arxiv_query: Optional[ArxivQuery] = None
        if route == "arxiv":
            arxiv_query = _parse_arxiv_query(item.get("arxiv_query"), text)
        sub_queries.append(
            SubQuery(text=text, route=route, arxiv_query=arxiv_query)
        )
    return sub_queries


def _parse_arxiv_query(
    payload: Optional[Dict[str, Any]], text: str
) -> ArxivQuery:
    """Parse the structured arXiv intent for an arxiv-routed sub-query.

    Falls back to a plain keyword lookup over the sub-question text when the
    model omits or malforms the ``arxiv_query`` object, so the arXiv path
    always has a usable structured query.

    Args:
        payload: The ``arxiv_query`` object from the decomposition output,
            or ``None`` if the model did not emit one.
        text: The sub-question text, used as the keyword fallback terms.

    Returns:
        The parsed ``ArxivQuery`` (a keyword query over ``text`` on fallback).
    """
    if not isinstance(payload, dict):
        return ArxivQuery(query_type="keyword", terms=text)
    try:
        return ArxivQuery.model_validate(payload)
    except ValidationError:
        return ArxivQuery(query_type="keyword", terms=text)


def parse_grade(payload: Dict[str, Any]) -> str:
    """Parse the grading LLM output into a grade label.

    Args:
        payload: The JSON object returned by the grading call.

    Returns:
        ``"good"`` or ``"weak"``; defaults to ``"weak"`` when the value is
        missing or unrecognized (fail safe — a weak grade triggers a retry).
    """
    grade = payload.get("grade")
    return "good" if grade == "good" else "weak"


class AgentNodes:
    """Holds dependencies and implements each LangGraph node.

    Attributes:
        settings: Configuration (caps, top_k, model names).
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Dict[str, Tool],
        settings: Settings,
    ) -> None:
        """Construct the node set.

        Args:
            llm: The LLM client used for decompose, grade, reformulate,
                and synthesize.
            tools: Mapping from route label (``"vector"``/``"arxiv"``) to the
                tool that serves it.
            settings: Configuration shared across nodes.
        """
        self._llm: LLMClient = llm
        self._tools: Dict[str, Tool] = tools
        self.settings: Settings = settings

    def decompose(self, state: AgentState) -> Dict[str, Any]:
        """Split the question into routed sub-queries.

        Falls back to a single vector-routed sub-query (the whole question) if
        decomposition returns nothing usable, so the pipeline always has work
        to do.

        Args:
            state: The current agent state (must contain ``question``).

        Returns:
            A partial state update with ``sub_queries`` and a reset ``cursor``.
        """
        question = state["question"]
        system = DECOMPOSE_SYSTEM.format(
            max_subqueries=self.settings.max_subqueries
        )
        with _timed("decompose"):
            payload = self._llm.complete_json(
                model=self.settings.grader_model,
                system=system,
                user=build_decompose_user(question),
                schema=DECOMPOSE_SCHEMA,
            )
            sub_queries = parse_subqueries(
                payload, self.settings.max_subqueries
            )
            if not sub_queries:
                sub_queries = [SubQuery(text=question, route="vector")]
        logger.info("decompose -> %d sub-quer(ies)", len(sub_queries))
        return {"sub_queries": sub_queries, "cursor": 0}

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve context for the sub-query at the cursor.

        Routes to the tool tagged on the sub-query; an unknown route degrades
        to no results rather than raising.

        Args:
            state: The current agent state.

        Returns:
            A partial state update with the cursor sub-query's ``chunks`` set.
        """
        sub_queries = list(state["sub_queries"])
        cursor = state["cursor"]
        sub_query = sub_queries[cursor]

        tool = self._tools.get(sub_query.route)
        with _timed(
            "retrieve", f"[{cursor + 1}] route={sub_query.route}"
        ):
            chunks: List[Chunk] = self._retrieve_with(
                tool, sub_query
            )
        logger.info("retrieve -> %d hit(s)", len(chunks))
        sub_query.chunks = chunks
        sub_queries[cursor] = sub_query
        return {"sub_queries": sub_queries}

    def _retrieve_with(
        self, tool: Optional[Tool], sub_query: SubQuery
    ) -> List[Chunk]:
        """Retrieve for a sub-query, passing structured intent when present.

        Vector sub-queries (and any tool exposing only the plain ``Tool``
        contract) are searched by text. An arxiv-routed sub-query that carries
        an ``arxiv_query`` is retrieved through the structured path so the
        live API can apply author/date/id constraints rather than a keyword
        search (see ``docs/ADR.md`` section 4.7).

        Args:
            tool: The tool tagged on the sub-query, or ``None`` for an
                unknown route.
            sub_query: The sub-query being retrieved for.

        Returns:
            The retrieved chunks (empty on an unknown route or tool failure).
        """
        if tool is None:
            return []
        top_k = self.settings.top_k
        structured_search = getattr(tool, "structured_search", None)
        if sub_query.arxiv_query is not None and callable(structured_search):
            return structured_search(sub_query.arxiv_query, top_k)
        return tool.search(sub_query.text, top_k)

    def grade(self, state: AgentState) -> Dict[str, Any]:
        """Grade the relevance of the cursor sub-query's context.

        Empty context is graded ``"weak"`` without an LLM call (the verdict is
        unambiguous and the call would be wasted).

        Args:
            state: The current agent state.

        Returns:
            A partial state update with the cursor sub-query's ``grade`` set.
        """
        sub_queries = list(state["sub_queries"])
        cursor = state["cursor"]
        sub_query = sub_queries[cursor]

        if not sub_query.chunks:
            sub_query.grade = "weak"
        else:
            with _timed("grade", f"[{cursor + 1}]"):
                payload = self._llm.complete_json(
                    model=self.settings.grader_model,
                    system=GRADE_SYSTEM,
                    user=build_grade_user(
                        sub_query.text, sub_query.chunks
                    ),
                    schema=GRADE_SCHEMA,
                )
                sub_query.grade = parse_grade(payload)

        logger.info("grade -> [%d] %s", cursor + 1, sub_query.grade)
        sub_queries[cursor] = sub_query
        return {"sub_queries": sub_queries}

    def reformulate(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite the cursor sub-query and increment its retry count.

        This node only runs when ``should_retry`` permits it, so the retry
        increment here is what the bound is enforced against.

        Args:
            state: The current agent state.

        Returns:
            A partial state update with the cursor sub-query's ``text``
            rewritten and ``retries`` incremented.
        """
        sub_queries = list(state["sub_queries"])
        cursor = state["cursor"]
        sub_query = sub_queries[cursor]

        with _timed("reformulate", f"[{cursor + 1}] retry"):
            rewritten = self._llm.complete(
                model=self.settings.grader_model,
                system=REFORMULATE_SYSTEM,
                user=build_reformulate_user(sub_query.text),
            ).strip()
        if rewritten:
            sub_query.text = rewritten
            if sub_query.arxiv_query is not None:
                # Only keyword/recent queries depend on free-text terms, so
                # only they benefit from a rewrite. Author and id lookups are
                # already precise field queries — rephrasing them just dilutes
                # the field clause with a noisy ``all:`` blob, so leave their
                # structure intact and let the bounded retry re-run as-is.
                if sub_query.arxiv_query.query_type in ("keyword", "recent"):
                    sub_query.arxiv_query.terms = rewritten
        sub_query.retries += 1
        sub_queries[cursor] = sub_query
        return {"sub_queries": sub_queries}

    def advance(self, state: AgentState) -> Dict[str, Any]:
        """Move the cursor to the next sub-query.

        Args:
            state: The current agent state.

        Returns:
            A partial state update incrementing ``cursor``.
        """
        return {"cursor": state["cursor"] + 1}

    def synthesize(self, state: AgentState) -> Dict[str, Any]:
        """Compose the final cited answer from all retrieved context.

        Deduplicates chunks by arXiv id across sub-queries, and flags low
        confidence when any sub-query's context remained weak after retries.

        Args:
            state: The current agent state (all sub-queries graded).

        Returns:
            A partial state update with ``answer`` and ``low_confidence``.
        """
        sub_queries = state["sub_queries"]
        chunks: List[Chunk] = []
        seen: set = set()
        low_confidence = False
        for sub_query in sub_queries:
            if sub_query.grade == "weak":
                low_confidence = True
            for chunk in sub_query.chunks:
                if chunk.arxiv_id and chunk.arxiv_id in seen:
                    continue
                seen.add(chunk.arxiv_id)
                chunks.append(chunk)

        with _timed("synthesize", f"{len(chunks)} chunk(s)"):
            answer = self._llm.complete(
                model=self.settings.synth_model,
                system=SYNTHESIZE_SYSTEM,
                user=build_synthesize_user(
                    state["question"], chunks, low_confidence
                ),
                max_tokens=2048,
            )
        return {"answer": answer, "low_confidence": low_confidence}

    def should_retry(self, state: AgentState) -> str:
        """Decide the next step after grading the cursor sub-query.

        This is the conditional edge that bounds the corrective loop: a weak
        grade triggers a reformulation **only** while the sub-query's retry
        count is below ``max_retries``. Otherwise the cursor advances. The
        retry count living in state and being checked here is what guarantees
        termination (see ``docs/ADR.md`` section 3).

        Args:
            state: The current agent state.

        Returns:
            ``"reformulate"`` to retry, or ``"advance"`` to move on.
        """
        sub_query = state["sub_queries"][state["cursor"]]
        if (
            sub_query.grade == "weak"
            and sub_query.retries < self.settings.max_retries
        ):
            return "reformulate"
        return "advance"

    def has_more(self, state: AgentState) -> str:
        """Decide whether to process another sub-query or synthesize.

        Args:
            state: The current agent state (cursor already advanced).

        Returns:
            ``"retrieve"`` if sub-queries remain, else ``"synthesize"``.
        """
        if state["cursor"] < len(state["sub_queries"]):
            return "retrieve"
        return "synthesize"
