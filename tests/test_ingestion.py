"""Unit tests for corpus parsing and filtering (field mapping)."""

from arxiv_rag.ingestion.corpus import parse_record, filter_record


def _raw_record() -> dict:
    """Return a representative raw corpus record.

    Returns:
        A dict shaped like one line of the Kaggle arXiv dataset.
    """
    return {
        "id": "0704.1028",
        "title": "A neural network\n  approach to ordinal regression",
        "abstract": "  We present a neural network for ordinal\nregression. ",
        "authors": "Jianlin Cheng",
        "authors_parsed": [["Cheng", "Jianlin", ""]],
        "categories": "cs.LG cs.AI cs.NE",
        "versions": [{"version": "v1", "created": "Sun, 8 Apr 2007 17:36:00 GMT"}],
    }


def test_parse_record_maps_fields() -> None:
    """parse_record maps and normalizes every field."""
    paper = parse_record(_raw_record())
    assert paper is not None
    assert paper.arxiv_id == "0704.1028"
    # whitespace (incl. embedded newlines) is collapsed
    assert paper.title == "A neural network approach to ordinal regression"
    assert paper.abstract == "We present a neural network for ordinal regression."
    assert paper.categories == ["cs.LG", "cs.AI", "cs.NE"]
    assert paper.authors == ["Jianlin Cheng"]
    assert paper.published == "Sun, 8 Apr 2007 17:36:00 GMT"


def test_parse_record_prefers_parsed_authors() -> None:
    """authors_parsed [last, first] is rendered as 'First Last'."""
    record = _raw_record()
    record["authors_parsed"] = [["Hinton", "Geoffrey", ""], ["LeCun", "Yann", ""]]
    paper = parse_record(record)
    assert paper.authors == ["Geoffrey Hinton", "Yann LeCun"]


def test_parse_record_missing_required_returns_none() -> None:
    """A record missing title or abstract is rejected."""
    record = _raw_record()
    del record["abstract"]
    assert parse_record(record) is None


def test_embedding_text_is_title_and_abstract() -> None:
    """The embedded text is title + abstract (ADR 4.5)."""
    paper = parse_record(_raw_record())
    text = paper.embedding_text()
    assert paper.title in text
    assert paper.abstract in text


def test_filter_record_matches_category() -> None:
    """filter_record keeps papers in the target category and drops others."""
    paper = parse_record(_raw_record())
    assert filter_record(paper, "cs.LG") is True
    assert filter_record(paper, "hep-ph") is False
