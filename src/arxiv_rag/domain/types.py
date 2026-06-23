"""Core domain types: Paper, Chunk, SubQuery, and the agent's shared state.

These types are the vocabulary the rest of the system speaks. ``Paper`` is the
ingested unit (one arXiv abstract); ``Chunk`` is a retrieved piece of context
with a relevance score; ``SubQuery`` is one decomposed part of a user question
tagged with the tool that should answer it; and ``AgentState`` is the typed
state that threads through every LangGraph node.
"""

from typing import List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field


Route = Literal["vector", "arxiv"]
"""Which source answers a sub-query: the local vector store or the live API."""

Grade = Literal["good", "weak"]
"""Relevance verdict assigned to a sub-query's retrieved context."""


class Paper(BaseModel):
    """A single arXiv paper as ingested into the vector store.

    The embedded text is ``title + abstract``; the remaining fields are stored
    as filterable Qdrant payload rather than embedded (see ``docs/ADR.md``
    section 4.5).

    Attributes:
        arxiv_id: The arXiv identifier, e.g. ``2103.00020``.
        title: Paper title.
        abstract: Paper abstract.
        authors: List of author names.
        categories: arXiv category tags, e.g. ``["cs.LG", "stat.ML"]``.
        published: Publication date as an ISO-8601 string, if known.
    """

    arxiv_id: str
    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    published: Optional[str] = None

    def embedding_text(self) -> str:
        """Return the text to embed for this paper.

        Returns:
            The title and abstract concatenated, which is what the retriever
            embeds and matches queries against.
        """
        return f"{self.title}\n\n{self.abstract}".strip()


class Chunk(BaseModel):
    """A retrieved unit of context with its source and relevance score.

    For this corpus a chunk corresponds to a whole paper (no sub-document
    chunking — see ``docs/ADR.md`` section 4.5), but the type is kept distinct
    from ``Paper`` so the retrieval contract does not change if chunking is
    later introduced.

    Attributes:
        arxiv_id: Identifier of the source paper.
        title: Title of the source paper.
        text: The retrieved text (title + abstract).
        score: Relevance score in ``[0, 1]``; higher is more relevant.
        source: Which tool produced the chunk (``"vector"`` or ``"arxiv"``).
        authors: Author names of the source paper.
        published: Publication date of the source paper, if known.
    """

    arxiv_id: str
    title: str
    text: str
    score: float = 0.0
    source: Route = "vector"
    authors: List[str] = Field(default_factory=list)
    published: Optional[str] = None

    def citation(self) -> str:
        """Return a short, human-readable citation tag for this chunk.

        Returns:
            A string of the form ``"Title (arXiv:ID)"`` suitable for inline
            citation in a synthesized answer.
        """
        return f"{self.title} (arXiv:{self.arxiv_id})"


class SubQuery(BaseModel):
    """One decomposed part of a user question, with retrieval bookkeeping.

    A ``SubQuery`` carries everything the corrective-retrieval loop needs for
    that part of the question: the routed tool, the (possibly reformulated)
    query text, the retrieved context, the relevance grade, and the retry
    count that bounds the loop.

    Attributes:
        text: The sub-question text (reformulated on retry).
        route: The tool this sub-query is routed to.
        chunks: Context retrieved for this sub-query.
        grade: Relevance verdict for ``chunks`` (``None`` until graded).
        retries: Number of corrective re-retrievals performed so far.
    """

    text: str
    route: Route
    chunks: List[Chunk] = Field(default_factory=list)
    grade: Optional[Grade] = None
    retries: int = 0


class AgentState(TypedDict, total=False):
    """Typed state threaded through every LangGraph node.

    LangGraph requires a ``TypedDict`` (or dataclass) for its shared state.
    The corrective loop is bounded by inspecting ``sub_queries[i].retries``
    in the conditional edge (see ``docs/ADR.md`` section 3).

    Attributes:
        question: The original user question.
        sub_queries: The decomposed sub-queries with their retrieval state.
        cursor: Index of the sub-query currently being processed.
        answer: The final synthesized answer (set by the synthesize node).
        low_confidence: Whether the answer was produced despite weak context.
    """

    question: str
    sub_queries: List[SubQuery]
    cursor: int
    answer: str
    low_confidence: bool
