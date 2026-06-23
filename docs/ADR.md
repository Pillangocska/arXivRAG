# Architecture Decision Record — arXiv Hybrid RAG Agent

## 1. Context & problem

Researchers and practitioners need to answer questions over scientific literature that mix two fundamentally different information needs:

1. **What does the existing literature say?** - questions answered by the *content* of papers (claims, methods, results). This is a semantic-search / retrieval problem.
2. **What is the current state?** - questions about recent publications, papers by a given author, or lookups by ID. This is a *live structured-data* problem that a static index cannot answer.

A pure RAG system handles (1) and fails (2); a question that bundles both ("what does paper X claim, and what newer work improves on it?") cannot be answered by a single retrieval pass. This motivated a hybrid design where retrieval is one tool among several that an agent orchestrates.

**Domain:** scientific literature (arXiv). **Corpus:** the Kaggle arXiv metadata dataset (abstracts), scoped to a single category because of hardware constraints.

## 2. Approach chosen - Hybrid RAG + Agentic

The defining idea: paper content lives in a vector store and is served by RAG; current/structured state is served by the live arXiv API; an agent decides, per sub-question, which source is needed. The agent adds two behaviors a linear RAG pipeline lacks:

- **Query decomposition** - multi-part questions are split into sub-queries, each routed independently. This is what makes the hybrid design *necessary* rather than decorative: different sub-queries genuinely require different tools.
- **Corrective retrieval** - retrieved context is graded for relevance; weak context triggers a single, bounded re-retrieval before synthesis.

Decomposition is the backbone (it exercises both tools); corrective retrieval is a bounded safeguard layered inside each sub-query's retrieval. This hierarchy keeps the system implementable within the time budget while still demonstrating two agentic patterns.

## 3. System architecture TODO

```
Query → Decompose → Route ─┬─→ Vector search (Qdrant) ──┐
                           └─→ arXiv API (live) ─────────┤
                                                         ▼
                                              Grade relevance
                                                   │
                                    weak (≤1 retry) │ good
                                          ┌─────────┴────────┐
                                          ▼                  ▼
                                   Reformulate & retry   Synthesize → Answer
```

The agent is a LangGraph state machine. A typed `AgentState` threads through every node, carrying the sub-queries, per-sub-query retrieved context, grades, retry counts, and the accumulated answer. The retry count lives in state and is checked by the conditional edge, which is how the corrective loop is bounded.

**Nodes:** decompose → route → (vector_search | arxiv_api) → grade → conditional (synthesize | reformulate-and-retry) → synthesize.

## 4. Key technical decisions

### 4.1 LLM selection

The workload is tiered: different calls have different difficulty and frequency, so they use different models.

- **Decompose & grade → Claude Haiku 4.5.** Grading is a near-binary, high-frequency relevance judgment; decomposition needs reliable structured output but limited reasoning. Haiku is ~5x cheaper than Sonnet and adequate for both.
- **Synthesize → Claude Sonnet 4.6.** This is the user-facing, citation-bearing answer where faithfulness matters most. Sonnet is the quality/cost sweet spot; Opus was considered and rejected because synthesis over a handful of retrieved abstracts is grounding-bound, not reasoning-bound, and does not justify the premium.

Model names are configuration variables, so the provider/model is swappable without code changes.

*Rates (per million tokens, mid-2026):* Haiku 4.5 $1/$5, Sonnet 4.6 $3/$15, Opus $5/$25. Output is 5x input; batch is 50% off; prompt caching is up to 90% off cached input.

### 4.2 Local embeddings vs. API generation

Embeddings run **locally**; generation uses the **API**.

- Embeddings are quality-tolerant and high-volume — running them locally removes per-query embedding cost and keeps the retrieval path working even if the LLM API is unavailable or vica-versa.
- Generation is quality-sensitive, the synthesis step needs the faithfulness of a frontier model.

A fully-local LLM was considered. On a 32GB laptop an 8B quantized model fits comfortably alongside Docker and Qdrant, but (a) the 8B tier is well below Sonnet on faithful, grounded generation, and the 32B+ models that would close the gap strain 32GB and run slowly on CPU; (b) local CPU generation is ~20–60s per answer vs. a few seconds on the API. API generation was chosen for quality and latency. The architecture remains provider-agnostic, so a local model (e.g. via Ollama) is a config change (this is also the data-privacy / graceful-degradation story).

### 4.3 Embedding model

Two local candidates were shortlisted from the MTEB retrieval leaderboard and benchmarked on the corpus: **BGE-small-en-v1.5** (small, fast, strong, 384-dim) and **Qwen3-Embedding-0.6B** (newer, near the top of the open leaderboard, laptop-runnable). The final choice is decided by an MRR/NDCG measurement on a corpus-derived eval set, not by leaderboard rank alone — leaderboard scores don't guarantee performance on a specialized domain like scientific abstracts.

Domain-specific embedders (SPECTER2, SciNCL) were considered and set aside: they are trained on the citation graph for *document<->document* similarity, whereas RAG needs *short-query↔abstract* retrieval, a different task shape.

### 4.4 Vector database

**Qdrant.** The workload combines semantic search with structured metadata filtering (category, publication date, author), and Qdrant's payload filtering is first-class. It runs fully locally in Docker for the demo and the same client scales to a hosted deployment, so there is no dev -> prod rewrite.

Alternatives: Chroma (simpler, but weaker filtering/scale — a more junior default); pgvector (interesting given the structured side of the workload, but doesnt scale well); FAISS (fast but a library, not a database — no built-in metadata filtering, which the use case requires); Pinecone/Weaviate Cloud (rejected — cloud dependency violates the local constraint).

### 4.5 Chunking strategy

**No chunking — one abstract per record.** arXiv abstracts are short (~200–400 tokens), fit the embedding context window whole, and are self-contained semantic units authored to stand alone. Splitting them would fragment meaning (separating problem from method from result). The embedded text is `title + abstract`; `category`, `authors`, `date`, and `arxiv_id` are stored as filterable Qdrant payload, not embedded.

If the corpus were extended to full-text PDFs, chunking would become necessary, recursive/semantic chunking at ~512–1024 tokens with ~10–15% overlap on section boundaries, possibly with a parent-document retriever. This is documented as a future variant, not built.

### 4.6 Framework choice

**LangGraph.** The control flow is cyclic and conditional: the corrective step can route a sub-query back for re-retrieval, and that bounded loop is awkward to express in a linear (acyclic) chain but native to a state graph. LangGraph also provides explicit shared state (where the retry cap lives), an inspectable/visualizable graph for documentation, and clean composition of the decomposition and corrective behaviors.

A hand-rolled orchestration was considered. For a single linear RAG pass it would be the right, lighter choice; here it would mean writing a bespoke state machine and loop guard, which is more error-prone and less readable than a framework built for exactly this. The trade-off accepted: a heavier dependency in exchange for the cyclic-flow fit.

### 4.7 Agentic design

- **State:** `AgentState` (sub-queries, results, grades, retry counts, final answer).
- **Tools:** `vector_search(query, filters) → chunks` (RAG path); `arxiv_api(query_type, params) → papers` (live path, wrapped with timeout + error handling for graceful degradation).
- **Routing:** the tool for each sub-query is tagged during decomposition (cheaper than a separate routing LLM call). For arXiv-routed sub-queries the decomposition step also emits a structured `arxiv_query` (`query_type` of author/id/recent/keyword plus params), which the tool translates into arXiv field syntax — `au:"…"`, `id_list`, and a `submittedDate:[…]` range — rather than a plain keyword search, so author/date/id lookups hit the right results and recency queries sort newest-first.
- **Bounding:** sub-queries capped at ~3; corrective retry capped at 1; on exhausted retry the system proceeds and flags low confidence rather than looping.

## 5. Trade-offs considered

**Optimized for:** demonstrating thought process and architectural fit, with faithful grounded answers as the quality target. Cost is controlled via model tiering, local embeddings, and prompt caching; latency is acceptable but not the primary axis.

**Accepted trade-offs:**
- Sequential sub-query execution (simpler state handling) over parallel (lower latency).
- A single arXiv category (laptop-friendly index) over the full corpus.
- A synthetic eval set (fast, no scraping) over a hand-curated one (more realistic).
- LangGraph's dependency weight over a lighter hand-rolled loop.

**What I would improve with more time:**
- Parallelize independent sub-queries.
- Expand and harden the eval set with hand-written multi-hop questions.
- Add a hybrid local/API deployment (local grader, API synthesizer).
- A reranking stage between retrieval and grading.
- Scale the index (full corpus; production embedding/indexing pipeline).
- And many more...

**Production considerations identified:**
- Expose the agent behind a FastAPI service with async endpoints and a job queue for long-running multi-hop queries (out of scope here per the brief's guidance against non-trivial interfaces).
- Online monitoring: sample-based faithfulness on live traffic, corrective-retry rate and retrieval-score distributions as health signals, latency/cost per query.
- Config-driven model selection for per-environment cost/quality tuning.

## 6. How success is measured

Evaluation has two layers.

**Deterministic unit tests** — ingestion/field mapping, the arXiv API tool's error path, prompt parsing, and the retry-cap routing logic (the test that proves the loop is bounded). LLM and network calls are mocked.

**RAG metrics (Ragas)** over a synthetic, corpus-derived eval set:
- **Faithfulness** — is the answer grounded in retrieved context (hallucination)? Primary metric.
- **Answer relevance** — does the answer address the question?
- **Context precision** — were the retrieved chunks relevant (retrieval quality)?
- (Context recall noted as a useful addition.)

These localize failures: context precision/recall judge retrieval; faithfulness/answer-relevance judge generation.

**Agent-workflow tests** — routing correctness (query → expected tool), decomposition correctness (expected sub-query coverage), and corrective-loop behavior (fires on weak context, stops after one retry).
