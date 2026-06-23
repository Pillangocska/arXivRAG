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

from arxiv_rag.config import get_settings, Settings
from arxiv_rag.factory import build_agent

DEFAULT_INPUT = "eval/eval_set.json"
RESULTS_DIR = "eval/results"


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
        print(f"  [{i}/{len(items)}] answered", flush=True)
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

    judge = LangchainLLMWrapper(
        ChatAnthropic(
            model=settings.synth_model,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
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

    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return 1
    if not os.path.exists(args.input):
        print(
            f"Eval set not found at {args.input}. "
            "Run `python -m eval.generate` first."
        )
        return 1

    with open(args.input, "r", encoding="utf-8") as handle:
        items = json.load(handle)

    print(f"Running agent over {len(items)} questions...", flush=True)
    records = _run_agent(settings, items)

    print("Scoring with Ragas...", flush=True)
    result = _score(records, settings)
    scores = {k: float(v) for k, v in result._repr_dict.items()} if hasattr(
        result, "_repr_dict"
    ) else dict(result)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = os.path.join(RESULTS_DIR, "scores.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {"scores": scores, "n": len(items)}, handle, indent=2
        )

    print("\n=== Ragas scores ===", flush=True)
    for metric, value in scores.items():
        print(f"  {metric}: {value:.3f}", flush=True)
    print(f"\nWrote results to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
