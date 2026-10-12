
"""Language model client with configuration checks and bounded retries."""

from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings, get_settings
from src.core.exceptions import LLMClientError, LLMResponseError
from src.utils.context_manager import ContextManager


RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class _EmptyModelResponseError(Exception):
    """Raised when the provider returns no usable text."""


class LLMClient:
    """Send token-validated chat-completion requests to an LLM provider."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.validate_llm_configuration()

        self.context_manager = context_manager or ContextManager(
            settings=self.settings
        )

        # Dependency injection allows tests to run without real API calls.
        self.client = client or OpenAI(
            api_key=self.settings.llm_api_key.get_secret_value(),
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=0,
        )

    def _request(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """Request a completion and validate its text content."""
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        if not response.choices:
            raise _EmptyModelResponseError(
                "The provider returned no completion choices."
            )

        content = response.choices[0].message.content

        if not isinstance(content, str) or not content.strip():
            raise _EmptyModelResponseError(
                "The provider returned empty text."
            )

        return content.strip()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> str:
        """Validate input size and generate text with bounded retries."""
        if not messages:
            raise ValueError("Messages must not be empty.")

        token_limit = (
            self.settings.max_output_tokens
            if max_tokens is None
            else max_tokens
        )

        if token_limit <= 0:
            raise ValueError("max_tokens must be positive.")

        # Reject oversized prompts before making any provider request.
        self.context_manager.validate_messages(messages)

        retrying = Retrying(
            stop=stop_after_attempt(self.settings.llm_max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(
                RETRYABLE_ERRORS + (_EmptyModelResponseError,)
            ),
            reraise=True,
        )

        try:
            return retrying(
                self._request,
                messages=messages,
                max_tokens=token_limit,
            )

        except _EmptyModelResponseError as exc:
            raise LLMResponseError(
                "The language model returned empty text after retries. "
                "Try again or select another available model."
            ) from exc

        except APIError as exc:
            raise LLMClientError(
                "The language model provider request failed. "
                "Check provider availability and configuration."
            ) from exc

