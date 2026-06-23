"""CLI entrypoint for the arXiv Hybrid RAG agent.

Two subcommands:

- ``ingest`` — build the index: stream the corpus, embed locally, upsert to
  Qdrant.
- ``ask`` — answer a question end-to-end and print the cited answer along with
  a short trace of how each sub-query was routed and graded.

Run via ``python -m arxiv_rag.app <subcommand>`` or the ``arxiv-rag`` script.
"""

from typing import Optional, List
import warnings
import argparse

# LangGraph's checkpoint package emits a LangChainPendingDeprecationWarning
# about ``allowed_objects`` at import time, before any of our code runs.
# Suppress just that message so the CLI output stays clean.
warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
    category=Warning,
)

from arxiv_rag.logging_config import configure_logging, get_logger
from arxiv_rag.config import get_settings, Settings
from arxiv_rag.domain import AgentState

logger = get_logger(__name__)


def _cmd_ingest(settings: Settings) -> int:
    """Run the ingestion pipeline.

    Args:
        settings: Application configuration.

    Returns:
        A process exit code (``0`` on success, ``1`` on a known failure).
    """
    from arxiv_rag.factory import build_embedder, build_store
    from arxiv_rag.ingestion import ingest

    logger.info(
        "Ingesting category '%s' (max %d) from %s",
        settings.arxiv_category,
        settings.max_papers,
        settings.corpus_path,
    )
    embedder = build_embedder(settings)
    store = build_store(settings)
    try:
        total = ingest(settings, embedder, store)
    except FileNotFoundError:
        logger.error(
            "Corpus not found at %s. Download the arXiv metadata JSON "
            "from Kaggle and set CORPUS_PATH.",
            settings.corpus_path,
        )
        return 1
    logger.info(
        "Done. Indexed %d papers into '%s'.",
        total,
        settings.qdrant_collection,
    )
    return 0


def _print_trace(state: AgentState) -> None:
    """Print a short trace of routing and grading for an answered question.

    Args:
        state: The final agent state returned by the agent.
    """
    print("\n--- trace ---", flush=True)
    for i, sub_query in enumerate(state.get("sub_queries", []), start=1):
        print(
            f"  [{i}] route={sub_query.route} "
            f"grade={sub_query.grade} retries={sub_query.retries} "
            f"hits={len(sub_query.chunks)}",
            flush=True,
        )
        print(f"      q: {sub_query.text}", flush=True)
    if state.get("low_confidence"):
        logger.warning(
            "low confidence: weak context for part of the query."
        )


def _cmd_ask(settings: Settings, question: str) -> int:
    """Answer a single question and print the result.

    Args:
        settings: Application configuration.
        question: The user's question.

    Returns:
        A process exit code (``0`` on success, ``1`` on a known failure).
    """
    from arxiv_rag.factory import build_agent

    if not settings.anthropic_api_key:
        logger.error(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file."
        )
        return 1

    agent = build_agent(settings)
    state = agent.answer(question)

    print("\n=== answer ===\n", flush=True)
    print(state.get("answer", "(no answer produced)"), flush=True)
    _print_trace(state)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured argument parser with ``ingest`` and ``ask``
        subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="arxiv-rag",
        description="Hybrid RAG + agentic research assistant over arXiv.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build the vector index.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question.")
    ask_parser.add_argument(
        "question", help="The question to answer."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A process exit code.
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "ingest":
        return _cmd_ingest(settings)
    if args.command == "ask":
        return _cmd_ask(settings, args.question)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
