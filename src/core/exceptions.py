"""Application-specific exceptions."""


class ApplicationError(Exception):
    """Base exception for expected application errors."""


class LLMClientError(ApplicationError):
    """Raised when the language model provider request fails."""


class LLMResponseError(LLMClientError):
    """Raised when the provider returns an unusable response."""


class ContextLengthError(ApplicationError):
    """Raised when a prompt exceeds the configured input-token budget."""
