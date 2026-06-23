"""Local text embedding behind an ``Embedder`` protocol.

Embeddings run locally (no per-query API cost; retrieval works offline). The
``Embedder`` protocol keeps the rest of the system decoupled from the concrete
model, so swapping ``BAAI/bge-small-en-v1.5`` for another sentence-transformers
model — or a different backend entirely — is a single-class change (see
``docs/ADR.md`` sections 4.2 and 4.3).
"""

from typing import Protocol, Sequence, List
import threading


class Embedder(Protocol):
    """Protocol for turning text into dense vectors.

    Implementations must produce L2-normalized vectors so that a dot product
    equals cosine similarity, matching the ``COSINE`` distance configured on
    the vector store.
    """

    @property
    def dim(self) -> int:
        """The dimensionality of the produced vectors."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of documents.

        Args:
            texts: The document texts to embed.

        Returns:
            One vector per input text, in the same order.
        """
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: The query text to embed.

        Returns:
            The query's embedding vector.
        """
        ...


class SentenceTransformerEmbedder:
    """An ``Embedder`` backed by a local sentence-transformers model.

    The underlying model is loaded lazily on first use so that importing this
    module (e.g. in unit tests) does not pull multi-hundred-megabyte weights
    into memory. Loading is guarded by a lock to keep first use thread-safe.

    Attributes:
        model_name: The sentence-transformers model identifier.
    """

    def __init__(self, model_name: str, expected_dim: int) -> None:
        """Initialize the embedder without loading the model.

        Args:
            model_name: The sentence-transformers model to load on first use.
            expected_dim: The vector dimensionality declared in configuration,
                used to validate the loaded model.
        """
        self.model_name: str = model_name
        self._expected_dim: int = expected_dim
        self._model: object = None
        self._lock: threading.Lock = threading.Lock()

    def _ensure_model(self) -> object:
        """Load and cache the model on first call (thread-safe).

        Returns:
            The loaded ``SentenceTransformer`` instance.

        Raises:
            ValueError: If the loaded model's dimensionality does not match
                the configured ``expected_dim``.
        """
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            actual_dim = model.get_sentence_embedding_dimension()
            if actual_dim != self._expected_dim:
                raise ValueError(
                    f"Model '{self.model_name}' has dimension {actual_dim}, "
                    f"but EMBEDDING_DIM is set to {self._expected_dim}."
                )
            self._model = model
            return self._model

    @property
    def dim(self) -> int:
        """Return the configured vector dimensionality.

        Returns:
            The expected embedding dimension (does not force model loading).
        """
        return self._expected_dim

    def embed_documents(
        self, texts: Sequence[str]
    ) -> List[List[float]]:
        """Embed a batch of documents into normalized vectors.

        Args:
            texts: The document texts to embed.

        Returns:
            One L2-normalized vector (as a list of floats) per input text.
        """
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a normalized vector.

        Args:
            text: The query text to embed.

        Returns:
            The query's L2-normalized embedding vector.
        """
        return self.embed_documents([text])[0]
