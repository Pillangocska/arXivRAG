"""Shared test fixtures and fakes.

These fakes implement the protocols (``LLMClient``, ``Tool``, ``VectorStore``,
``Embedder``) so the agent and pipeline can be exercised without any network,
API key, or running services — the deterministic-test requirement in
``docs/ADR.md`` section 6.
"""

from typing import List, Dict, Any
import pytest

from arxiv_rag.config import Settings
from arxiv_rag.domain import Chunk


@pytest.fixture
def settings() -> Settings:
    """Return settings with small caps suitable for tests.

    Returns:
        A ``Settings`` instance with deterministic, test-friendly values.
    """
    return Settings(
        ANTHROPIC_API_KEY="test-key",
        TOP_K=3,
        MAX_SUBQUERIES=3,
        MAX_RETRIES=1,
        ARXIV_CATEGORY="cs.LG",
    )


class FakeTool:
    """A ``Tool`` returning preset chunks, recording every query it sees.

    Attributes:
        name: The route label this tool answers.
        calls: The list of ``(query, top_k)`` calls made, in order.
    """

    def __init__(self, name: str, results: List[Chunk]) -> None:
        """Construct the fake tool.

        Args:
            name: The route label (``"vector"`` or ``"arxiv"``).
            results: The chunks to return from every ``search`` call.
        """
        self.name: str = name
        self._results: List[Chunk] = results
        self.calls: List[tuple] = []

    def search(self, query: str, top_k: int) -> List[Chunk]:
        """Return the preset results and record the call.

        Args:
            query: The query searched for.
            top_k: The requested result count.

        Returns:
            The preset chunk list.
        """
        self.calls.append((query, top_k))
        return list(self._results)


class FakeLLM:
    """A scripted ``LLMClient`` for deterministic agent tests.

    The fake returns canned decomposition and grade payloads and a fixed
    synthesis string, while recording how many times it was asked to grade so
    tests can assert on the corrective loop's behavior.

    Attributes:
        grade_calls: Number of ``complete_json`` grade calls made.
        grades: The sequence of grades to return, one per grade call.
    """

    def __init__(
        self,
        sub_queries: List[Dict[str, str]],
        grades: List[str],
        answer: str = "A cited answer (arXiv:1234.5678).",
    ) -> None:
        """Construct the scripted LLM.

        Args:
            sub_queries: The sub-query list the decompose call returns.
            grades: Grades returned by successive grade calls (last repeats).
            answer: The synthesis text returned by ``complete``.
        """
        self._sub_queries = sub_queries
        self._grades = grades
        self._answer = answer
        self.grade_calls: int = 0

    def complete(
        self, model: str, system: str, user: str, max_tokens: int = 1024
    ) -> str:
        """Return a reformulation or the synthesis answer.

        Args:
            model: Ignored (recorded by real client, not needed here).
            system: Used to distinguish reformulate from synthesize.
            user: Ignored.
            max_tokens: Ignored.

        Returns:
            A reformulated query for reformulate prompts, else the answer.
        """
        if "Rewrite" in system or "reformulat" in system.lower():
            return "reformulated query"
        return self._answer

    def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return scripted decomposition or grade payloads.

        Args:
            model: Ignored.
            system: Used to distinguish decompose from grade.
            user: Ignored.
            schema: Ignored.
            max_tokens: Ignored.

        Returns:
            A decomposition payload for decompose prompts, else a grade
            payload (advancing through ``grades``).
        """
        if "planner" in system.lower():
            return {"sub_queries": self._sub_queries}
        # grade call
        idx = min(self.grade_calls, len(self._grades) - 1)
        grade = self._grades[idx]
        self.grade_calls += 1
        return {"grade": grade}


def make_chunk(arxiv_id: str = "1234.5678", score: float = 0.9) -> Chunk:
    """Build a simple chunk for use in tests.

    Args:
        arxiv_id: The chunk's arXiv id.
        score: The chunk's relevance score.

    Returns:
        A populated ``Chunk``.
    """
    return Chunk(
        arxiv_id=arxiv_id,
        title="A Paper",
        text="A Paper\n\nAn abstract about learning.",
        score=score,
        source="vector",
    )
