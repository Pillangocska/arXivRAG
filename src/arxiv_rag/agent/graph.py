"""LangGraph wiring and the high-level ``RagAgent`` facade.

The graph realizes the state machine from ``docs/ADR.md`` section 3:

    decompose -> retrieve -> grade -> (reformulate -> retrieve | advance)
    advance -> (retrieve | synthesize) -> END

The cyclic, conditional shape — grade can route a sub-query back through
retrieval — is exactly what motivates LangGraph over a linear chain (ADR 4.6).
"""

from typing import Dict

from langgraph.graph import StateGraph, START, END

from arxiv_rag.agent.nodes import AgentNodes
from arxiv_rag.domain import AgentState
from arxiv_rag.llm import LLMClient
from arxiv_rag.config import Settings
from arxiv_rag.tools import Tool


def build_graph(nodes: AgentNodes):
    """Build and compile the agent's LangGraph state machine.

    Args:
        nodes: The node implementations bound to the agent's dependencies.

    Returns:
        A compiled LangGraph application ready to ``invoke``.
    """
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("decompose", nodes.decompose)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("reformulate", nodes.reformulate)
    graph.add_node("advance", nodes.advance)
    graph.add_node("synthesize", nodes.synthesize)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "grade")

    # Corrective loop: a weak grade routes back to reformulate (bounded by
    # retry count), otherwise the cursor advances.
    graph.add_conditional_edges(
        "grade",
        nodes.should_retry,
        {"reformulate": "reformulate", "advance": "advance"},
    )
    graph.add_edge("reformulate", "retrieve")

    # After advancing, either process the next sub-query or synthesize.
    graph.add_conditional_edges(
        "advance",
        nodes.has_more,
        {"retrieve": "retrieve", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


class RagAgent:
    """High-level facade over the compiled agent graph.

    Attributes:
        nodes: The bound node implementations.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Dict[str, Tool],
        settings: Settings,
    ) -> None:
        """Construct the agent and compile its graph.

        Args:
            llm: The LLM client for the agent's reasoning steps.
            tools: Mapping from route label to retrieval tool.
            settings: Configuration shared across the agent.
        """
        self.nodes: AgentNodes = AgentNodes(llm, tools, settings)
        self._app = build_graph(self.nodes)

    def answer(self, question: str) -> AgentState:
        """Run the full pipeline for a question.

        Args:
            question: The user's question.

        Returns:
            The final agent state, including ``answer``, ``sub_queries``
            (with their routes, grades, and retrieved context), and
            ``low_confidence``.
        """
        initial: AgentState = {"question": question, "cursor": 0}
        return self._app.invoke(initial)
