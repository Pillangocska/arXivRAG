"""Unit tests for the arXiv API tool's error handling and result mapping."""

from datetime import datetime, timezone
from types import SimpleNamespace

from arxiv_rag.tools import ArxivApiTool


def _fake_result() -> SimpleNamespace:
    """Build an object shaped like ``arxiv.Result``.

    Returns:
        A stand-in result with the attributes the tool reads.
    """
    return SimpleNamespace(
        title="Attention Is All You Need",
        summary="We propose the Transformer, based solely on attention.",
        published=datetime(2017, 6, 12, tzinfo=timezone.utc),
        authors=[SimpleNamespace(name="Ashish Vaswani")],
        get_short_id=lambda: "1706.03762",
    )


def test_search_maps_results(monkeypatch) -> None:
    """A successful query maps results to chunks tagged source='arxiv'."""
    tool = ArxivApiTool(timeout=5.0)
    monkeypatch.setattr(
        tool,
        "_run_query",
        lambda *a, **k: [tool._result_to_chunk(_fake_result())],
    )
    chunks = tool.search("transformer attention", top_k=3)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.arxiv_id == "1706.03762"
    assert chunk.source == "arxiv"
    assert "Transformer" in chunk.text
    assert chunk.authors == ["Ashish Vaswani"]


def test_search_returns_empty_on_error(monkeypatch) -> None:
    """A query that raises degrades to an empty list, not an exception."""
    tool = ArxivApiTool(timeout=5.0)

    def _boom(*args, **kwargs):
        raise ConnectionError("arxiv is down")

    monkeypatch.setattr(tool, "_run_query", _boom)
    assert tool.search("anything", top_k=3) == []


def test_search_returns_empty_on_timeout(monkeypatch) -> None:
    """A query that exceeds the timeout returns an empty list."""
    import time

    tool = ArxivApiTool(timeout=0.1)

    def _slow(*args, **kwargs):
        time.sleep(1.0)
        return []

    monkeypatch.setattr(tool, "_run_query", _slow)
    assert tool.search("slow query", top_k=3) == []
