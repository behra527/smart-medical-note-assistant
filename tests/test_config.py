import pytest
from pydantic import SecretStr, ValidationError

from src.config import Settings


def test_default_configuration_without_env_file():
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.llm_timeout_seconds == 30.0
    assert settings.llm_max_retries == 3
    assert settings.max_input_tokens == 6000
    assert settings.max_output_tokens == 1500


def test_api_key_is_not_configured_by_default():
    settings = Settings(_env_file=None)

    assert settings.has_api_key is False


def test_api_key_is_detected():
    settings = Settings(
        _env_file=None,
        llm_api_key=SecretStr("test-key"),
        llm_model="test-model",
    )

    assert settings.has_api_key is True


def test_missing_api_key_raises_clear_error():
    settings = Settings(_env_file=None, llm_model="test-model")

    with pytest.raises(ValueError, match="LLM_API_KEY is missing"):
        settings.validate_llm_configuration()


def test_missing_model_raises_clear_error():
    settings = Settings(
        _env_file=None,
        llm_api_key=SecretStr("test-key"),
    )

    with pytest.raises(ValueError, match="LLM_MODEL is missing"):
        settings.validate_llm_configuration()


def test_negative_timeout_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_timeout_seconds=-1)


def test_retry_count_cannot_be_negative():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_max_retries=-1)


def test_token_budget_values_are_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_input_tokens=0)
