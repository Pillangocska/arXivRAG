"""Generate a synthetic, corpus-derived evaluation set.

Sampling papers from the ingested corpus and asking an LLM to write a
question + reference answer for each gives a fast, scrape-free eval set whose
ground truth is anchored to real abstracts (see ``docs/ADR.md`` sections 5 and
6). The result is written as JSON for ``run_eval.py`` to consume.

Run via ``python -m eval.generate`` (optionally ``--size N``).
"""

from typing import List, Dict, Any
import argparse
import random
import json
import os

from arxiv_rag.logging_config import configure_logging, get_logger
from arxiv_rag.llm.client import AnthropicLLMClient
from arxiv_rag.ingestion.corpus import load_papers
from arxiv_rag.config import get_settings, Settings
from arxiv_rag.domain import Paper

logger = get_logger(__name__)


QGEN_SYSTEM = """\
You write evaluation questions for a RAG system over arXiv abstracts. Given a \
single paper's title and abstract, produce one specific, self-contained \
question that the abstract answers, and a concise reference answer grounded \
strictly in that abstract. The question must be answerable from the abstract \
alone and must not name the paper or its authors."""

QGEN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
    },
    "required": ["question", "reference_answer"],
    "additionalProperties": False,
}

DEFAULT_OUTPUT = "eval/eval_set.json"


def _reservoir_sample(
    settings: Settings, size: int, pool_cap: int
) -> List[Paper]:
    """Sample papers uniformly from the corpus via reservoir sampling.

    Reading the whole corpus to sample from it would be wasteful, so a bounded
    pool is streamed (``pool_cap`` papers) and ``size`` are sampled from it.

    Args:
        settings: Application configuration (corpus path, category).
        size: Number of papers to sample.
        pool_cap: Maximum number of papers to stream into the pool.

    Returns:
        A list of up to ``size`` sampled papers.
    """
    pool: List[Paper] = list(
        load_papers(
            path=settings.corpus_path,
            category=settings.arxiv_category,
            max_papers=pool_cap,
        )
    )
    random.seed(42)
    if len(pool) <= size:
        return pool
    return random.sample(pool, size)


def generate(
    settings: Settings, size: int, output: str
) -> List[Dict[str, str]]:
    """Generate and persist the synthetic evaluation set.

    Args:
        settings: Application configuration.
        size: Number of question/answer items to generate.
        output: Path to write the eval set JSON to.

    Returns:
        The list of generated items, each with ``question``,
        ``reference_answer``, and the source ``arxiv_id``.
    """
    llm = AnthropicLLMClient(api_key=settings.anthropic_api_key)
    papers = _reservoir_sample(settings, size, pool_cap=size * 50)

    items: List[Dict[str, str]] = []
    for i, paper in enumerate(papers, start=1):
        user = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
        try:
            payload = llm.complete_json(
                model=settings.grader_model,
                system=QGEN_SYSTEM,
                user=user,
                schema=QGEN_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001 - skip a bad generation
            logger.warning("[%d] skipped (%s)", i, exc)
            continue
        items.append(
            {
                "question": payload["question"],
                "reference_answer": payload["reference_answer"],
                "arxiv_id": paper.arxiv_id,
            }
        )
        logger.info("[%d/%d] generated", i, len(papers))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)
    logger.info("Wrote %d items to %s", len(items), output)
    return items


def main() -> int:
    """CLI entry point for eval-set generation.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic RAG evaluation set."
    )
    parser.add_argument(
        "--size", type=int, default=10, help="Number of items to generate."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output JSON path."
    )
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    if not settings.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return 1
    generate(settings, args.size, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
