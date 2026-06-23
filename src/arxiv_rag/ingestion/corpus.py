"""Load and filter the Kaggle arXiv metadata corpus.

The Kaggle dataset (``arxiv-metadata-oai-snapshot.json``) is a JSON-lines file
with one paper object per line. This module streams it line by line — the full
file is several gigabytes — parsing each record into a ``Paper``, filtering to
the configured category, and stopping once the paper cap is reached.

The parse and filter steps are pure functions so they can be unit-tested
without touching the filesystem (see ``docs/ADR.md`` section 6).
"""

from typing import Iterator, Optional, Dict, Any, List
import json


def _normalize_authors(record: Dict[str, Any]) -> List[str]:
    """Extract a clean list of author names from a raw record.

    The dataset provides authors both as a free-text ``authors`` string and as
    a structured ``authors_parsed`` list of ``[last, first, suffix]`` triples;
    the structured form is preferred when present.

    Args:
        record: A raw parsed JSON record from the corpus.

    Returns:
        A list of ``"First Last"`` author name strings (possibly empty).
    """
    parsed = record.get("authors_parsed")
    if isinstance(parsed, list) and parsed:
        names: List[str] = []
        for entry in parsed:
            if not isinstance(entry, list):
                continue
            last = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            full = f"{first} {last}".strip()
            if full:
                names.append(full)
        return names
    authors = record.get("authors")
    if isinstance(authors, str) and authors.strip():
        return [authors.strip()]
    return []


def _categories(record: Dict[str, Any]) -> List[str]:
    """Split the space-separated ``categories`` field into a list.

    Args:
        record: A raw parsed JSON record from the corpus.

    Returns:
        The list of category tags (possibly empty).
    """
    categories = record.get("categories")
    if isinstance(categories, str):
        return [c for c in categories.split() if c]
    return []


def _published(record: Dict[str, Any]) -> Optional[str]:
    """Extract a publication date string from a record's version history.

    The first version's ``created`` timestamp is used as the publication date.

    Args:
        record: A raw parsed JSON record from the corpus.

    Returns:
        The first-version creation timestamp, or ``None`` if unavailable.
    """
    versions = record.get("versions")
    if isinstance(versions, list) and versions:
        first = versions[0]
        if isinstance(first, dict):
            created = first.get("created")
            if isinstance(created, str):
                return created
    return None


def parse_record(record: Dict[str, Any]) -> Optional["Paper"]:
    """Parse one raw corpus record into a ``Paper``.

    Args:
        record: A raw parsed JSON object from the corpus.

    Returns:
        A ``Paper`` if the record has the required fields (id, title,
        abstract), otherwise ``None``.
    """
    from arxiv_rag.domain import Paper

    arxiv_id = record.get("id")
    title = record.get("title")
    abstract = record.get("abstract")
    if not arxiv_id or not title or not abstract:
        return None
    return Paper(
        arxiv_id=str(arxiv_id).strip(),
        title=" ".join(str(title).split()),
        abstract=" ".join(str(abstract).split()),
        authors=_normalize_authors(record),
        categories=_categories(record),
        published=_published(record),
    )


def filter_record(paper: "Paper", category: str) -> bool:
    """Decide whether a parsed paper belongs in the index.

    Args:
        paper: The parsed paper.
        category: The arXiv category the corpus is scoped to.

    Returns:
        ``True`` if the paper lists the target category, else ``False``.
    """
    return category in paper.categories


def load_papers(
    path: str, category: str, max_papers: int
) -> Iterator["Paper"]:
    """Stream papers from the corpus file, filtered and capped.

    The file is read lazily, one line at a time, so memory use stays flat
    regardless of corpus size. Malformed JSON lines are skipped rather than
    aborting the run.

    Args:
        path: Path to the JSON-lines corpus file.
        category: The arXiv category to keep.
        max_papers: Maximum number of papers to yield.

    Yields:
        Parsed ``Paper`` objects matching the category, up to ``max_papers``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    yielded = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper = parse_record(record)
            if paper is None:
                continue
            if not filter_record(paper, category):
                continue
            yield paper
            yielded += 1
            if yielded >= max_papers:
                return
