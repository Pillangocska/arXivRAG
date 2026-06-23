"""Centralized logging setup for the arXiv Hybrid RAG agent.

Provides a single place to configure logging so every component logs through
the standard ``logging`` module with a consistent format that carries a
timestamp and the name of the component (module/class) doing the logging.

Modules obtain a logger via :func:`get_logger` instead of calling ``print``.
Entry points (the CLI, the eval scripts) call :func:`configure_logging` once
at startup to install the handler and format.
"""

from typing import Optional
import logging
import sys

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that log chatty INFO records (HTTP requests, model
# loading, arXiv paging). They are pinned to WARNING so our own INFO logs
# stay readable; raise an individual one to INFO when debugging it.
NOISY_LOGGERS: tuple = (
    "httpx",
    "httpcore",
    "arxiv",
    "datasets",
    "sentence_transformers",
    "urllib3",
)

_configured: bool = False


def configure_logging(level: int = logging.INFO) -> None:
    """Install the root logging handler and format.

    Idempotent: repeated calls do not add duplicate handlers. Output goes to
    ``stderr`` so it stays separate from any program output written to
    ``stdout``.

    Args:
        level: The minimum severity level to emit. Defaults to ``INFO``.
    """
    global _configured
    if _configured:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger for a component.

    The returned logger's name appears in every record it emits, so log lines
    identify the module or class doing the logging. Pass ``__name__`` from a
    module, or a class name for finer granularity.

    Args:
        name: The logger name. Defaults to the package root logger.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name if name is not None else "arxiv_rag")
