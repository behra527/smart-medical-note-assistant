"""Token estimation and input-budget validation."""

import json
from typing import Any

import tiktoken

from src.config import Settings, get_settings
from src.core.exceptions import ContextLengthError


class ContextManager:
    """Estimate prompt tokens and reject prompts exceeding the safe budget."""

    def __init__(
        self,
        settings: Settings | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.settings = settings or get_settings()

        if self.settings.context_safety_margin >= self.settings.max_input_tokens:
            raise ValueError(
                "CONTEXT_SAFETY_MARGIN must be smaller than MAX_INPUT_TOKENS."
            )

        self.encoding = tiktoken.get_encoding(encoding_name)

        self.input_token_limit = (
            self.settings.max_input_tokens
            - self.settings.context_safety_margin
        )

    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a string."""
        return len(self.encoding.encode(text))

    def count_message_tokens(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        """Estimate tokens for serialized chat messages and their content."""
        if not messages:
            raise ValueError("Messages must not be empty.")

        serialized_messages = json.dumps(
            messages,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return self.count_tokens(serialized_messages)

    def validate_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        """Return estimated token count or reject an oversized prompt."""
        token_count = self.count_message_tokens(messages)

        if token_count > self.input_token_limit:
            raise ContextLengthError(
                "The prompt exceeds the safe input-token budget. "
                f"Estimated tokens: {token_count}; "
                f"allowed tokens: {self.input_token_limit}. "
                "Reduce the source notes and try again. "
                "The application does not automatically truncate clinical notes."
            )

        return token_count
