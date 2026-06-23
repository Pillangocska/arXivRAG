"""End-to-end agent tests with mocked LLM and tools (no network).

These exercise the full compiled LangGraph — decomposition, per-sub-query
routing, the corrective loop, and synthesis — proving the wiring terminates
and produces an answer for both the happy path and the weak-context path.
"""

from arxiv_rag.agent import RagAgent
from tests.conftest import FakeTool, FakeLLM, make_chunk


def test_end_to_end_good_context(settings) -> None:
    """A multi-part question routes to both tools and synthesizes an answer."""
    llm = FakeLLM(
        sub_queries=[
            {"text": "what does the transformer propose", "route": "vector"},
            {"text": "recent attention papers", "route": "arxiv"},
        ],
        grades=["good", "good"],
        answer="The Transformer uses attention (arXiv:1706.03762).",
    )
    tools = {
        "vector": FakeTool("vector", [make_chunk("1706.03762")]),
        "arxiv": FakeTool("arxiv", [make_chunk("2401.00001")]),
    }
    agent = RagAgent(llm=llm, tools=tools, settings=settings)

    state = agent.answer("What does the transformer propose, and what's new?")

    assert "arXiv:1706.03762" in state["answer"]
    assert len(state["sub_queries"]) == 2
    assert state["sub_queries"][0].route == "vector"
    assert state["sub_queries"][1].route == "arxiv"
    assert state["low_confidence"] is False


def test_end_to_end_weak_context_retries_once_then_finishes(settings) -> None:
    """Weak context triggers exactly one retry, then the run completes."""
    # Grade weak, weak (after retry still weak) -> loop must stop at the cap.
    llm = FakeLLM(
        sub_queries=[{"text": "obscure question", "route": "vector"}],
        grades=["weak", "weak"],
        answer="Insufficient context to answer fully.",
    )
    vector_tool = FakeTool("vector", [make_chunk("9999.00001")])
    agent = RagAgent(
        llm=llm, tools={"vector": vector_tool}, settings=settings
    )

    state = agent.answer("an obscure question")

    sub = state["sub_queries"][0]
    assert sub.retries == 1  # bounded by MAX_RETRIES=1
    # retrieved twice: initial attempt + one corrective retry
    assert len(vector_tool.calls) == 2
    assert state["low_confidence"] is True
    assert state["answer"]
