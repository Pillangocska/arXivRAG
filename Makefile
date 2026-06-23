# Common tasks for the arXiv Hybrid RAG agent.
#
# Two ways to run:
#   - Host (default): uses `uv` directly on your machine. Fastest for dev.
#   - Containerized:  `make build` then the `*-docker` targets run inside the
#     app image alongside Qdrant via docker-compose.
#
# Examples:
#   make up                       # start Qdrant only
#   make up SERVICE=all           # start Qdrant and the app
#   make ingest                   # build the index (host)
#   make run Q="your question"    # ask a question (host)
#   make test                     # run the test suite
#   make eval                     # generate eval set + score with Ragas

Q ?=
EVAL_SIZE ?= 10
SERVICE ?= qdrant

.PHONY: help up down ingest run test eval eval-generate eval-run \
        build ingest-docker run-docker clean

help:
	@echo "Targets:"
	@echo "  up [SERVICE=all]  Start services (default: qdrant only)."
	@echo "  down              Stop all services."
	@echo "  ingest            Build the index on the host (uv)."
	@echo "  run Q=\"...\"       Ask a question on the host (uv)."
	@echo "  test              Run the test suite."
	@echo "  eval              Generate an eval set and score it with Ragas."
	@echo "  build             Build the app container image."
	@echo "  ingest-docker     Build the index inside the container."
	@echo "  run-docker Q=..   Ask a question inside the container."

# --- Services ---
SERVICE_ARGS = $(if $(filter all,$(SERVICE)),qdrant app,$(SERVICE))

up:
	docker compose up -d $(SERVICE_ARGS)

down:
	docker compose down

# --- Host (uv) workflow ---
ingest:
	uv run python -m arxiv_rag.app ingest

run:
	@if [ -z '$(Q)' ]; then echo 'Usage: make run Q="your question"'; exit 1; fi
	uv run python -m arxiv_rag.app ask "$(Q)"

test:
	uv run pytest -q

eval: eval-generate eval-run

eval-generate:
	uv run python -m eval.generate --size $(EVAL_SIZE)

eval-run:
	uv run python -m eval.run_eval

# --- Containerized workflow ---
build:
	docker compose build app

ingest-docker:
	docker compose run --rm --no-deps app ingest

run-docker:
	@if [ -z '$(Q)' ]; then echo 'Usage: make run-docker Q="your question"'; exit 1; fi
	docker compose run --rm --no-deps app ask "$(Q)"

clean:
	rm -rf .pytest_cache .ruff_cache eval/results
