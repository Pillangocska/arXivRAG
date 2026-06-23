"""Composition root: build the concrete components from configuration.

Centralizing construction here keeps the wiring in one place and means the CLI
and ingestion entrypoints share the same dependency assembly. Each builder
returns the concrete type behind its protocol, so swapping an implementation
(a different embedder, store, or LLM provider) is a change confined to this
module.
"""

from typing import Dict

from arxiv_rag.retrieval import (
    SentenceTransformerEmbedder,
    QdrantVectorStore,
    VectorStore,
    Embedder,
)
from arxiv_rag.tools import VectorSearchTool, ArxivApiTool, Tool
from arxiv_rag.llm import AnthropicLLMClient, LLMClient
from arxiv_rag.agent import RagAgent
from arxiv_rag.config import Settings


def build_embedder(settings: Settings) -> Embedder:
    """Build the local embedder from configuration.

    Args:
        settings: Application configuration.

    Returns:
        A lazily-loading sentence-transformers embedder.
    """
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        expected_dim=settings.embedding_dim,
    )


def build_store(settings: Settings) -> VectorStore:
    """Build the Qdrant-backed vector store from configuration.

    Args:
        settings: Application configuration.

    Returns:
        A vector store connected to the configured Qdrant endpoint.
    """
    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )


def build_llm(settings: Settings) -> LLMClient:
    """Build the Anthropic LLM client from configuration.

    Args:
        settings: Application configuration.

    Returns:
        An Anthropic-backed LLM client.
    """
    return AnthropicLLMClient(api_key=settings.anthropic_api_key)


def build_tools(
    settings: Settings, embedder: Embedder, store: VectorStore
) -> Dict[str, Tool]:
    """Build the retrieval tools keyed by route label.

    Args:
        settings: Application configuration.
        embedder: The embedder used by vector search.
        store: The vector store searched by vector search.

    Returns:
        A mapping ``{"vector": ..., "arxiv": ...}`` of tools.
    """
    vector_tool = VectorSearchTool(
        embedder=embedder,
        store=store,
        score_threshold=settings.score_threshold,
        category=settings.arxiv_category,
    )
    arxiv_tool = ArxivApiTool(timeout=settings.arxiv_timeout)
    return {vector_tool.name: vector_tool, arxiv_tool.name: arxiv_tool}


def build_agent(settings: Settings) -> RagAgent:
    """Build a fully-wired agent ready to answer questions.

    Args:
        settings: Application configuration.

    Returns:
        A ``RagAgent`` with real embedder, store, tools, and LLM client.
    """
    embedder = build_embedder(settings)
    store = build_store(settings)
    llm = build_llm(settings)
    tools = build_tools(settings, embedder, store)
    return RagAgent(llm=llm, tools=tools, settings=settings)
