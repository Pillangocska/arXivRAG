"""Unit tests for prompt parsing and user-message builders."""

from arxiv_rag.agent.nodes import parse_subqueries, parse_grade
from arxiv_rag.llm.prompts import build_grade_user, build_synthesize_user
from arxiv_rag.domain import Chunk


def test_parse_subqueries_caps_and_keeps_routes() -> None:
    """Sub-queries are capped and valid routes preserved."""
    payload = {
        "sub_queries": [
            {"text": "what does paper X claim", "route": "vector"},
            {"text": "recent papers on Y", "route": "arxiv"},
            {"text": "third", "route": "vector"},
            {"text": "fourth (over cap)", "route": "vector"},
        ]
    }
    subs = parse_subqueries(payload, max_subqueries=3)
    assert len(subs) == 3
    assert subs[0].route == "vector"
    assert subs[1].route == "arxiv"


def test_parse_subqueries_defaults_unknown_route_to_vector() -> None:
    """An unknown route falls back to the safe 'vector' default."""
    payload = {"sub_queries": [{"text": "q", "route": "nonsense"}]}
    subs = parse_subqueries(payload, max_subqueries=3)
    assert subs[0].route == "vector"


def test_parse_subqueries_skips_empty_text() -> None:
    """Sub-queries with blank text are dropped."""
    payload = {"sub_queries": [{"text": "  ", "route": "vector"}]}
    assert parse_subqueries(payload, max_subqueries=3) == []


def test_parse_grade_defaults_to_weak() -> None:
    """An unrecognized grade fails safe to 'weak' (triggers a retry)."""
    assert parse_grade({"grade": "good"}) == "good"
    assert parse_grade({"grade": "weak"}) == "weak"
    assert parse_grade({}) == "weak"
    assert parse_grade({"grade": "maybe"}) == "weak"


def test_build_grade_user_handles_no_results() -> None:
    """The grade prompt states explicitly when nothing was retrieved."""
    text = build_grade_user("a sub-question", [])
    assert "no results" in text.lower()


def test_build_synthesize_user_includes_citations_and_caveat() -> None:
    """Synthesis context carries arXiv ids and a low-confidence caveat."""
    chunk = Chunk(
        arxiv_id="1234.5678", title="T", text="body", source="vector"
    )
    text = build_synthesize_user("q", [chunk], low_confidence=True)
    assert "arXiv:1234.5678" in text
    assert "weak" in text.lower()
