"""Provider-agnostic LLM client, with an Anthropic implementation.

The ``LLMClient`` protocol exposes the two shapes the agent needs: a plain
text completion (for synthesis) and a JSON completion validated against a
schema (for decomposition and grading). Keeping this behind a protocol means
the provider is a single-class swap (see ``docs/ADR.md`` section 4.1).

The Anthropic implementation uses prompt caching on the stable system-prompt
prefix to reduce repeated-context cost, and retries transient API failures
with exponential backoff.
"""

from typing import Protocol, Dict, Any

from tenacity import (
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    retry,
)


class LLMError(RuntimeError):
    """Raised when an LLM call fails after exhausting retries."""


class LLMClient(Protocol):
    """Protocol for the LLM operations the agent depends on."""

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> str:
        """Return a plain-text completion.

        Args:
            model: The model identifier to use.
            system: The system prompt (stable prefix; cached when supported).
            user: The user message.
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's text response.
        """
        ...

    def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return a completion constrained to a JSON schema.

        Args:
            model: The model identifier to use.
            system: The system prompt (stable prefix; cached when supported).
            user: The user message.
            schema: A JSON Schema the output must conform to.
            max_tokens: Maximum tokens to generate.

        Returns:
            The parsed JSON object.
        """
        ...


def _is_retryable(exc: BaseException) -> bool:
    """Return whether an Anthropic exception is worth retrying.

    Args:
        exc: The exception raised by the Anthropic SDK.

    Returns:
        ``True`` for rate limits, connection errors, and 5xx responses.
    """
    import anthropic

    if isinstance(
        exc, (anthropic.RateLimitError, anthropic.APIConnectionError)
    ):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


class AnthropicLLMClient:
    """An ``LLMClient`` backed by the Anthropic Messages API.

    The SDK already retries transient errors, but an explicit ``tenacity``
    policy is layered on so the backoff is uniform across the codebase and
    surfaces a single ``LLMError`` to callers on exhaustion.
    """

    def __init__(self, api_key: str) -> None:
        """Construct the client.

        Args:
            api_key: The Anthropic API key. If empty, the SDK falls back to
                the ``ANTHROPIC_API_KEY`` environment variable.
        """
        import anthropic

        self._client = (
            anthropic.Anthropic(api_key=api_key)
            if api_key
            else anthropic.Anthropic()
        )

    def _system_blocks(self, system: str) -> list:
        """Build a cacheable system-prompt block list.

        The stable system prefix is marked with ``cache_control`` so repeated
        calls (e.g. grading every sub-query) read it from cache rather than
        reprocessing it each time.

        Args:
            system: The system prompt text.

        Returns:
            A single-element list of content blocks with caching enabled.
        """
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _create_text(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        """Make one text Messages API call, retrying transient failures.

        Args:
            model: The model identifier.
            system: The system prompt.
            user: The user message.
            max_tokens: Maximum tokens to generate.

        Returns:
            The first text block of the response.

        Raises:
            Exception: Re-raised on retryable errors so ``tenacity`` retries;
                wrapped in ``LLMError`` on non-retryable errors.
        """
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system),
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below if final
            if _is_retryable(exc):
                raise
            raise LLMError(str(exc)) from exc
        return next(
            (b.text for b in response.content if b.type == "text"), ""
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _create_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Make one structured Messages API call via forced tool use.

        Structured output is obtained by defining a single tool whose
        ``input_schema`` is the requested schema and forcing the model to call
        it. This is portable across SDK versions and yields a validated,
        already-parsed input object — no brittle text-to-JSON parsing.

        Args:
            model: The model identifier.
            system: The system prompt.
            user: The user message.
            schema: The JSON Schema the tool input must conform to.
            max_tokens: Maximum tokens to generate.

        Returns:
            The tool-call input as a dict.

        Raises:
            LLMError: If the model returns no tool-use block.
            Exception: Re-raised on retryable errors so ``tenacity`` retries.
        """
        tool = {
            "name": "emit",
            "description": "Emit the structured result.",
            "input_schema": schema,
        }
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system),
                messages=[{"role": "user", "content": user}],
                tools=[tool],
                tool_choice={"type": "tool", "name": "emit"},
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below if final
            if _is_retryable(exc):
                raise
            raise LLMError(str(exc)) from exc
        for block in response.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise LLMError("Model returned no tool_use block for structured call.")

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> str:
        """Return a plain-text completion.

        Args:
            model: The model identifier to use.
            system: The system prompt.
            user: The user message.
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's text response.

        Raises:
            LLMError: If the call fails after exhausting retries.
        """
        try:
            return self._create_text(model, system, user, max_tokens)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(str(exc)) from exc

    def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return a completion constrained to a JSON schema.

        Args:
            model: The model identifier to use.
            system: The system prompt.
            user: The user message.
            schema: A JSON Schema the output must conform to.
            max_tokens: Maximum tokens to generate.

        Returns:
            The parsed JSON object.

        Raises:
            LLMError: If the call fails after exhausting retries.
        """
        try:
            return self._create_json(
                model, system, user, schema, max_tokens
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(str(exc)) from exc
