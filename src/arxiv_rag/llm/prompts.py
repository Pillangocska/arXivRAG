"""Prompts and output schemas for decomposition, grading, and synthesis.

Each agent LLM step has a stable system prompt (cached), a user-message
builder, and — for the structured steps — a JSON schema the output is
constrained to. The schemas double as the contract the parsing helpers in
``arxiv_rag.agent`` validate against.
"""

from typing import List, Dict, Any

from arxiv_rag.domain import Chunk


DECOMPOSE_SYSTEM = """\
You are a query planner for a research assistant over arXiv papers (computer \
science). Your job is to split a user's question into 1-{max_subqueries} \
focused sub-questions and route each one to the right source.

Two sources are available:
- "vector": semantic search over a local index of arXiv paper abstracts. Use \
this for questions about the *content* of the literature: what a paper claims, \
proposes, or shows; methods; results; comparisons of ideas.
- "arxiv": the live arXiv API. Use this for questions about *current state*: \
recent or latest papers, papers by a specific author, or lookups by arXiv id.

Rules:
- Emit the fewest sub-questions that fully cover the question. A simple, \
single-intent question yields exactly one sub-question.
- Each sub-question must be self-contained (resolve pronouns and references).
- Choose exactly one route per sub-question.
- Do not invent parts of the question that were not asked."""


DECOMPOSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "route": {
                        "type": "string",
                        "enum": ["vector", "arxiv"],
                    },
                },
                "required": ["text", "route"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sub_queries"],
    "additionalProperties": False,
}


GRADE_SYSTEM = """\
You grade whether retrieved context is relevant enough to answer a \
sub-question. Judge relevance only — not completeness or writing quality.

Return "good" if at least one retrieved item is on-topic and could support an \
answer to the sub-question. Return "weak" if the items are off-topic, only \
tangentially related, or empty.

Be decisive: this is a near-binary judgment, not a nuanced score."""


GRADE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["good", "weak"]},
        "reason": {"type": "string"},
    },
    "required": ["grade"],
    "additionalProperties": False,
}


REFORMULATE_SYSTEM = """\
A first retrieval attempt for a sub-question returned weak results. Rewrite \
the sub-question to retrieve better: vary the terminology, expand acronyms, \
or rephrase around the underlying concept. Keep the same information need. \
Return only the rewritten query text, with no preamble."""


SYNTHESIZE_SYSTEM = """\
You are a research assistant answering a question using retrieved arXiv \
context. Write a clear, accurate answer grounded strictly in the provided \
context.

Rules:
- Use only information supported by the context. Do not add facts from prior \
knowledge.
- Cite sources inline using the arXiv id in the form (arXiv:ID) immediately \
after the claim they support.
- Answer directly. Do not open with a preamble such as "Based on the \
provided context" or "Here is what can be said" — state the substantive \
answer from the first sentence.
- Do not add meta-commentary about the context or the answer itself (e.g. \
notes that an excerpt is brief, caveats that the full paper would be needed, \
or descriptions of what you are about to do). Every sentence should be a \
substantive, context-supported claim that answers the question.
- If the context genuinely cannot support part of the answer, say so in one \
short inline clause at the relevant point rather than in a separate section.
- Be concise and organized; address every part of the question."""


def build_decompose_user(question: str) -> str:
    """Build the decomposition user message.

    Args:
        question: The original user question.

    Returns:
        The user message presenting the question to decompose.
    """
    return f"Question:\n{question}"


def build_grade_user(sub_query: str, chunks: List[Chunk]) -> str:
    """Build the grading user message for one sub-query.

    Args:
        sub_query: The sub-question whose context is being graded.
        chunks: The retrieved context to grade.

    Returns:
        The user message pairing the sub-question with its context.
    """
    if not chunks:
        context = "(no results were retrieved)"
    else:
        context = "\n\n".join(
            f"[{i + 1}] {chunk.title}\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )
    return (
        f"Sub-question:\n{sub_query}\n\n"
        f"Retrieved context:\n{context}"
    )


def build_reformulate_user(sub_query: str) -> str:
    """Build the reformulation user message for a weak sub-query.

    Args:
        sub_query: The sub-question that retrieved weak context.

    Returns:
        The user message asking for a rewritten query.
    """
    return f"Original sub-question:\n{sub_query}"


def build_synthesize_user(
    question: str, chunks: List[Chunk], low_confidence: bool
) -> str:
    """Build the synthesis user message.

    Args:
        question: The original user question.
        chunks: All retrieved context across sub-queries.
        low_confidence: Whether some context remained weak after retries.

    Returns:
        The user message pairing the question with the aggregated context.
    """
    if not chunks:
        context = "(no relevant context was retrieved)"
    else:
        context = "\n\n".join(
            f"[{i + 1}] {chunk.title} (arXiv:{chunk.arxiv_id})\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )
    caveat = ""
    if low_confidence:
        caveat = (
            "\n\nNote: retrieval was weak for part of this question. Where "
            "the context does not support an answer, say so in one short "
            "inline clause; do not add a separate caveat section."
        )
    return f"Question:\n{question}\n\nContext:\n{context}{caveat}"
