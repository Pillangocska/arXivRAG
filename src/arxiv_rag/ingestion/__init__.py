"""Offline ingestion: load and filter the corpus, embed, and index."""

from arxiv_rag.ingestion.corpus import load_papers, parse_record, filter_record
from arxiv_rag.ingestion.indexer import ingest

__all__ = ["load_papers", "parse_record", "filter_record", "ingest"]
