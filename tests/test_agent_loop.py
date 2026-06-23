"""Unit tests for the corrective loop bound and routing logic.

These are the agent-workflow tests from ``docs/ADR.md`` section 6: that the
corrective retry fires on weak context, stops after the cap, and that
sub-queries reach the tool their route names.
"""

from arxiv_rag.agent.nodes import AgentNodes
from arxiv_rag.domain import SubQuery, AgentState
from tests.conftest import FakeTool, FakeLLM, make_chunk


def _nodes(llm, tools, settings) -> AgentNodes:
    """Construct ``AgentNodes`` from test doubles.

    Args:
        llm: A fake LLM client.
        tools: A route->tool mapping.
        settings: Test settings.

    Returns:
        The bound node set.
    """
    return AgentNodes(llm=llm, tools=tools, settings=settings)


def test_should_retry_fires_on_weak_then_stops(settings) -> None:
    """should_retry returns 'reformulate' once, then 'advance' at the cap."""
    nodes = _nodes(FakeLLM([], ["weak"]), {}, settings)
    sub = SubQuery(text="q", route="vector", grade="weak", retries=0)
    state: AgentState = {"sub_queries": [sub], "cursor": 0}

    # First weak grade, retries=0 < max_retries=1 -> retry
    assert nodes.should_retry(state) == "reformulate"

    # After one retry, retries=1 == max_retries -> advance (bounded)
    sub.retries = 1
    assert nodes.should_retry(state) == "advance"


def test_should_retry_advances_on_good(settings) -> None:
    """A good grade advances immediately without retrying."""
    nodes = _nodes(FakeLLM([], ["good"]), {}, settings)
    sub = SubQuery(text="q", route="vector", grade="good", retries=0)
    state: AgentState = {"sub_queries": [sub], "cursor": 0}
    assert nodes.should_retry(state) == "advance"


def test_reformulate_increments_retries_and_rewrites(settings) -> None:
    """reformulate rewrites the query text and bumps the retry count."""
    nodes = _nodes(FakeLLM([], ["weak"]), {}, settings)
    sub = SubQuery(text="original", route="vector", grade="weak")
    state: AgentState = {"sub_queries": [sub], "cursor": 0}
    update = nodes.reformulate(state)
    new_sub = update["sub_queries"][0]
    assert new_sub.retries == 1
    assert new_sub.text == "reformulated query"


def test_retrieve_routes_to_tagged_tool(settings) -> None:
    """A sub-query reaches exactly the tool its route names."""
    vector_tool = FakeTool("vector", [make_chunk("v1")])
    arxiv_tool = FakeTool("arxiv", [make_chunk("a1")])
    tools = {"vector": vector_tool, "arxiv": arxiv_tool}
    nodes = _nodes(FakeLLM([], ["good"]), tools, settings)

    sub = SubQuery(text="recent papers", route="arxiv")
    state: AgentState = {"sub_queries": [sub], "cursor": 0}
    update = nodes.retrieve(state)

    assert len(arxiv_tool.calls) == 1
    assert len(vector_tool.calls) == 0
    assert update["sub_queries"][0].chunks[0].arxiv_id == "a1"


def test_grade_empty_context_is_weak_without_llm(settings) -> None:
    """Empty retrieval is graded weak without spending an LLM grade call."""
    llm = FakeLLM([], ["good"])
    nodes = _nodes(llm, {}, settings)
    sub = SubQuery(text="q", route="vector", chunks=[])
    state: AgentState = {"sub_queries": [sub], "cursor": 0}
    update = nodes.grade(state)
    assert update["sub_queries"][0].grade == "weak"
    assert llm.grade_calls == 0


def test_has_more_controls_termination(settings) -> None:
    """has_more routes to retrieve while sub-queries remain, else synthesize."""
    nodes = _nodes(FakeLLM([], ["good"]), {}, settings)
    subs = [SubQuery(text="a", route="vector"), SubQuery(text="b", route="vector")]
    assert nodes.has_more({"sub_queries": subs, "cursor": 1}) == "retrieve"
    assert nodes.has_more({"sub_queries": subs, "cursor": 2}) == "synthesize"
