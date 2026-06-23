"""LangGraph node functions and their parsing helpers.

Each node is a pure-ish function from ``AgentState`` to a partial state update.
The nodes are grouped on an ``AgentNodes`` object that holds the agent's
dependencies (LLM client, tools, settings) so the graph wiring stays free of
globals and the whole agent is constructible with mocked dependencies in tests.

The corrective-retry bound lives in ``should_retry`` (consumed by the graph's
conditional edge), which is the test that proves the loop terminates (see
``docs/ADR.md`` sections 3 and 6).
"""

from typing import Dict, List, Any

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
from arxiv_rag.domain import AgentState, SubQuery, Chunk
from arxiv_rag.llm import LLMClient
from arxiv_rag.config import Settings
from arxiv_rag.tools import Tool


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
        sub_queries.append(SubQuery(text=text, route=route))
    return sub_queries


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
        chunks: List[Chunk] = (
            tool.search(sub_query.text, self.settings.top_k)
            if tool is not None
            else []
        )
        sub_query.chunks = chunks
        sub_queries[cursor] = sub_query
        return {"sub_queries": sub_queries}

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
            payload = self._llm.complete_json(
                model=self.settings.grader_model,
                system=GRADE_SYSTEM,
                user=build_grade_user(sub_query.text, sub_query.chunks),
                schema=GRADE_SCHEMA,
            )
            sub_query.grade = parse_grade(payload)

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

        rewritten = self._llm.complete(
            model=self.settings.grader_model,
            system=REFORMULATE_SYSTEM,
            user=build_reformulate_user(sub_query.text),
        ).strip()
        if rewritten:
            sub_query.text = rewritten
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
