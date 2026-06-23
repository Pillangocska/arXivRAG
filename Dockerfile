# syntax=docker/dockerfile:1

# Build the app image on the official uv + Python base for fast, reproducible
# installs. The corpus and Qdrant are external (mounted volume / compose
# service), so only the application and its dependencies live in the image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies first (cached layer) using only the manifest, so source
# changes don't invalidate the dependency install.
COPY pyproject.toml ./
RUN uv sync --no-install-project

# Copy the source and install the project itself.
COPY src ./src
COPY eval ./eval
RUN uv sync

# The embedding model is downloaded on first use into this cache; mounting it
# as a volume (see docker-compose.yml) avoids re-downloading across runs.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python", "-m", "arxiv_rag.app"]
CMD ["--help"]
