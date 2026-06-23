# Demo & Evaluation

This page shows what running the system looks like and how to read the
evaluation output. The example transcripts and scores below are illustrative
of the expected shape — regenerate them on your own machine with `make eval`
(scores depend on the corpus slice, the eval-set size, and the judge model).

## Prerequisites recap

1. `uv sync` — install dependencies.
2. Set `ANTHROPIC_API_KEY` in `.env`.
3. `make up` — start Qdrant in Docker.
4. Place the Kaggle corpus at `_data/arxiv-metadata-oai-snapshot.json`.
5. `make ingest` — build the index (filter to `cs.LG`, embed, upsert).

## Example: a hybrid, multi-part question

The defining case for this system is a question that bundles a *content* need
(answered by the local index) with a *current-state* need (answered by the
live arXiv API):

```bash
make run Q="What does the original transformer paper propose, and what recent papers improve on its attention mechanism?"
```

Expected shape of the output:

```
=== answer ===

The original Transformer paper proposes an architecture based entirely on
self-attention, dispensing with recurrence and convolutions ... (arXiv:1706.03762).
Recent work improves the attention mechanism along several lines: efficient /
sparse attention to reduce the quadratic cost ... (arXiv:2009.14794), ...

--- trace ---
  [1] route=vector grade=good retries=0 hits=5
      q: What does the original transformer paper propose?
  [2] route=arxiv grade=good retries=0 hits=5
      q: recent papers improving on transformer attention mechanisms
```

The trace shows the two behaviors the ADR calls out: **decomposition** (the
question was split into two sub-questions) and **routing** (the content
sub-question went to `vector`, the current-state one to `arxiv`).

## Example: the corrective loop

When the first retrieval for a sub-query grades `weak`, the agent reformulates
the query once and retries before synthesizing. A trace from that path looks
like:

```
--- trace ---
  [1] route=vector grade=weak retries=1 hits=4
      q: <reformulated query text>
  (!) low confidence: weak context for part of the query.
```

`retries=1` with `MAX_RETRIES=1` is the bound in action: the loop fires at most
once, then the system proceeds and flags low confidence rather than looping
(see `docs/ADR.md` section 4.7). The deterministic test
`test_agent_integration.py::test_end_to_end_weak_context_retries_once_then_finishes`
proves this terminates.

## Evaluation

Generate a synthetic, corpus-derived eval set and score the agent with Ragas:

```bash
make eval                  # = eval-generate (size 20) + eval-run
# or control the size:
make eval-generate EVAL_SIZE=50
make eval-run
```

`generate.py` samples papers from the ingested category and asks the grader
model to write a question + reference answer per paper. `run_eval.py` runs the
full agent over each question and scores the results.

Results are written to `eval/results/scores.json`. Illustrative shape:

```json
{
  "scores": {
    "faithfulness": 0.91,
    "answer_relevancy": 0.88,
    "context_precision": 0.79
  },
  "n": 20
}
```

How to read the metrics (see `docs/ADR.md` section 6):

- **Faithfulness** (primary) — is the answer grounded in the retrieved
  context? Low values indicate hallucination.
- **Answer relevance** — does the answer address the question?
- **Context precision** — were the retrieved chunks relevant? This isolates
  retrieval quality from generation quality.

Faithfulness is the primary guardrail because it works identically offline and
online (it needs only the answer + retrieved context, no ground truth), making
it suitable as both a pre-deploy gate and a live health signal.
```
