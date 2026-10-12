
"""Centralized application configuration.

Loads settings from environment variables and .env, validates configuration,
and keeps credentials separate from application business logic.
"""

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # LLM provider settings
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for the configured LLM provider.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        min_length=1,
    )
    # Empty is permitted during development and testing.
    # A model is required only when an LLM request is made.
    llm_model: str = Field(default="")

    # Request reliability
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # Context management
    max_input_tokens: int = Field(default=6000, gt=0)
    max_output_tokens: int = Field(default=1500, gt=0)
    context_safety_margin: int = Field(default=500, ge=0)

    # Application settings
    app_env: Literal["development", "testing", "production"] = "development"
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"

    @field_validator("llm_model")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        """Remove accidental whitespace from a configured model name."""
        return value.strip()

    @field_validator("llm_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Require an absolute HTTP(S) URL with a hostname."""
        value = value.strip()
        parsed = urlparse(value)

        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                "LLM_BASE_URL must be an absolute HTTP or HTTPS URL."
            )

        if parsed.username or parsed.password:
            raise ValueError(
                "Do not embed credentials in LLM_BASE_URL; use environment variables."
            )

        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_token_budget(self) -> "Settings":
        """Ensure the configured token budgets are positive and usable."""
        if self.max_input_tokens <= 0:
            raise ValueError("MAX_INPUT_TOKENS must be positive.")

        if self.max_output_tokens <= 0:
            raise ValueError("MAX_OUTPUT_TOKENS must be positive.")

        # Input, output, and safety reserve must all fit within the
        # context window configured by the application.
        if (
            self.max_input_tokens
            + self.max_output_tokens
            + self.context_safety_margin
            <= 0
        ):
            raise ValueError("The total token budget must be positive.")

        return self

    @property
    def has_api_key(self) -> bool:
        """Return whether a non-empty API key has been configured."""
        return bool(self.llm_api_key.get_secret_value().strip())

    def validate_llm_configuration(self) -> None:
        """Require credentials and a model before sending an LLM request."""
        if not self.has_api_key:
            raise ValueError(
                "LLM_API_KEY is missing. Configure it in your local .env file."
            )

        if not self.llm_model:
            raise ValueError(
                "LLM_MODEL is missing. Configure a model supported by your provider."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear cached settings, primarily for tests."""
    get_settings.cache_clear()
