"""Run the agent over the eval set and score it with Ragas.

For each question the agent produces an answer and retrieved context; Ragas
then scores faithfulness (the primary, grounding metric), answer relevance,
and context precision (see ``docs/ADR.md`` section 6). Results are written to
``eval/results/`` as JSON.

Run via ``python -m eval.run_eval`` after generating an eval set.
"""

from typing import List, Dict, Any
import argparse
import json
import os

from arxiv_rag.logging_config import configure_logging, get_logger
from arxiv_rag.config import get_settings, Settings
from arxiv_rag.factory import build_agent

logger = get_logger(__name__)

DEFAULT_INPUT = "eval/eval_set.json"
RESULTS_DIR = "eval/results"

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


def _collect_contexts(sub_queries: List[Any]) -> List[str]:
    """Flatten retrieved context across sub-queries into Ragas's format.

    Args:
        sub_queries: The agent's sub-queries, each carrying its chunks.

    Returns:
        A deduplicated list of retrieved-context strings.
    """
    seen: set = set()
    contexts: List[str] = []
    for sub_query in sub_queries:
        for chunk in sub_query.chunks:
            if chunk.arxiv_id in seen:
                continue
            seen.add(chunk.arxiv_id)
            contexts.append(chunk.text)
    return contexts


def _run_agent(
    settings: Settings, items: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Run the agent over every eval item, collecting answers and contexts.

    Args:
        settings: Application configuration.
        items: The eval items (question + reference answer).

    Returns:
        Per-item records with the question, answer, retrieved contexts, and
        reference answer — the input rows for Ragas.
    """
    agent = build_agent(settings)
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        state = agent.answer(item["question"])
        records.append(
            {
                "user_input": item["question"],
                "response": state.get("answer", ""),
                "retrieved_contexts": _collect_contexts(
                    state.get("sub_queries", [])
                ),
                "reference": item["reference_answer"],
            }
        )
        logger.info("[%d/%d] answered", i, len(items))
    return records


def _score(records: List[Dict[str, Any]], settings: Settings) -> Any:
    """Score the agent's outputs with Ragas metrics.

    Args:
        records: Per-item records produced by ``_run_agent``.
        settings: Application configuration (for the judge model).

    Returns:
        The Ragas ``EvaluationResult``.
    """
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        faithfulness,
    )
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import (
        HuggingFaceEmbeddings,
    )

    # Faithfulness asks the judge to emit one verdict per atomic statement in
    # the answer as a single JSON array; for long, multi-paragraph cited
    # answers that list can be sizeable. A low cap truncates the JSON
    # mid-generation, which Ragas surfaces as LLMDidNotFinishException and then
    # drops the row (skewing the average, or yielding NaN if every row fails).
    # Give the judge ample output room.
    judge = LangchainLLMWrapper(
        ChatAnthropic(
            model=settings.synth_model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
    )
    # Reuse the local embedding model for answer-relevance scoring so the
    # eval has no extra embedding-API dependency.
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    dataset = EvaluationDataset.from_list(records)
    return evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge,
        embeddings=embeddings,
    )


def _build_per_row(result: Any) -> List[Dict[str, Any]]:
    """Assemble a per-question record of inputs alongside each metric score.

    Joins the agent's question, answer, retrieved contexts, and reference with
    the per-row Ragas scores so a low aggregate can be traced to the specific
    questions dragging it down. A failed judge call surfaces as ``None`` for
    that metric rather than being hidden.

    Args:
        result: The Ragas ``EvaluationResult`` returned by ``_score``.

    Returns:
        One dict per eval item, sorted by faithfulness ascending (worst first)
        so the rows most worth inspecting appear at the top. Returns an empty
        list if the result cannot be converted to a DataFrame.
    """
    import math

    try:
        frame = result.to_pandas()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not build per-row dump: %s", exc)
        return []

    def _clean(value: Any) -> Any:
        """Coerce NaN floats to ``None`` so the JSON stays valid."""
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    rows: List[Dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        contexts = record.get("retrieved_contexts") or []
        rows.append(
            {
                "question": record.get("user_input"),
                "answer": record.get("response"),
                "reference": record.get("reference"),
                "retrieved_contexts": list(contexts),
                "n_contexts": len(contexts),
                "scores": {
                    metric: _clean(record.get(metric))
                    for metric in METRIC_NAMES
                },
            }
        )

    def _sort_key(row: Dict[str, Any]) -> float:
        """Order by faithfulness ascending; push missing scores to the top."""
        score = row["scores"].get("faithfulness")
        return float("-inf") if score is None else score

    rows.sort(key=_sort_key)
    return rows


def _print_per_row(rows: List[Dict[str, Any]]) -> None:
    """Print a compact per-question table, worst faithfulness first.

    Shows each question with its three metric scores so low-scoring rows are
    visible at a glance; the full answer and contexts live in ``per_row.json``.

    Args:
        rows: The per-row records from ``_build_per_row``.
    """
    if not rows:
        return

    def _fmt(value: Any) -> str:
        """Render a score as a fixed-width string, or ``--`` if missing."""
        return f"{value:.3f}" if isinstance(value, float) else "  -- "

    print("\n=== Per-question scores (worst faithfulness first) ===")
    print(f"  {'faith':>6} {'ans_rel':>8} {'ctx_prec':>9}  question")
    for row in rows:
        scores = row["scores"]
        question = (row.get("question") or "").replace("\n", " ")
        if len(question) > 80:
            question = question[:77] + "..."
        print(
            f"  {_fmt(scores.get('faithfulness')):>6} "
            f"{_fmt(scores.get('answer_relevancy')):>8} "
            f"{_fmt(scores.get('context_precision')):>9}  {question}",
            flush=True,
        )


def main() -> int:
    """CLI entry point for the evaluation run.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run the agent over an eval set and score with Ragas."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Eval set JSON path."
    )
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    if not settings.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return 1
    if not os.path.exists(args.input):
        logger.error(
            "Eval set not found at %s. Run `python -m eval.generate` first.",
            args.input,
        )
        return 1

    with open(args.input, "r", encoding="utf-8") as handle:
        items = json.load(handle)

    logger.info("Running agent over %d questions...", len(items))
    records = _run_agent(settings, items)

    logger.info("Scoring with Ragas...")
    result = _score(records, settings)
    scores = {k: float(v) for k, v in result._repr_dict.items()} if hasattr(
        result, "_repr_dict"
    ) else dict(result)

    # Each metric is averaged (via nanmean) over only the rows that scored
    # successfully; a row whose judge call failed becomes NaN and is silently
    # dropped from that metric's average. Count the rows that actually
    # contributed so a partial run is visible rather than masquerading as a
    # clean score over all ``n`` items.
    valid_counts: Dict[str, int] = {}
    if hasattr(result, "_scores_dict"):
        import math

        for metric, values in result._scores_dict.items():
            valid_counts[metric] = sum(
                1 for v in values if v is not None and not math.isnan(v)
            )

    per_row = _build_per_row(result)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = os.path.join(RESULTS_DIR, "scores.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "scores": scores,
                "n": len(items),
                "valid": valid_counts,
                "per_row": per_row,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Ragas scores ===", flush=True)
    for metric, value in scores.items():
        n_valid = valid_counts.get(metric, len(items))
        suffix = (
            f"  (over {n_valid}/{len(items)} rows)"
            if n_valid < len(items)
            else ""
        )
        print(f"  {metric}: {value:.3f}{suffix}", flush=True)
        if n_valid < len(items):
            logger.warning(
                "%s scored over only %d/%d rows; %d judge call(s) failed "
                "(likely truncated output).",
                metric,
                n_valid,
                len(items),
                len(items) - n_valid,
            )

    _print_per_row(per_row)

    logger.info(
        "Wrote results (with per-question detail) to %s", output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
