from unittest.mock import MagicMock

import pytest
from openai import APITimeoutError
from pydantic import SecretStr

from src.config import Settings
from src.core.exceptions import (
    ContextLengthError,
    LLMClientError,
    LLMResponseError,
)
from src.services.llm_client import LLMClient


def make_settings(**overrides):
    values = {
        "llm_api_key": SecretStr("test-key"),
        "llm_model": "test-model",
        "llm_max_retries": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_client(content="Generated draft", settings=None):
    fake_provider = MagicMock()
    fake_response = MagicMock()
    fake_response.choices[0].message.content = content
    fake_provider.chat.completions.create.return_value = fake_response

    client = LLMClient(
        settings=settings or make_settings(),
        client=fake_provider,
    )
    return client, fake_provider


def test_generate_returns_provider_text():
    client, _ = make_client("  Draft note  ")

    result = client.generate(
        [{"role": "user", "content": "Draft a note."}]
    )

    assert result == "Draft note"


def test_generate_uses_configured_model_and_token_limit():
    client, provider = make_client()

    client.generate(
        [{"role": "user", "content": "Draft a note."}],
        max_tokens=250,
    )

    provider.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "Draft a note."}],
        max_tokens=250,
        response_format={"type": "json_object"},
    )


def test_empty_messages_are_rejected():
    client, provider = make_client()

    with pytest.raises(ValueError, match="Messages must not be empty"):
        client.generate([])

    provider.chat.completions.create.assert_not_called()


def test_non_positive_token_limit_is_rejected():
    client, provider = make_client()

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        client.generate(
            [{"role": "user", "content": "Draft a note."}],
            max_tokens=0,
        )

    provider.chat.completions.create.assert_not_called()


def test_empty_provider_response_is_rejected():
    client, _ = make_client("  ")

    with pytest.raises(LLMResponseError, match="returned empty text"):
        client.generate([{"role": "user", "content": "Hello"}])


def test_missing_api_key_is_rejected_before_client_creation():
    settings = make_settings(llm_api_key=SecretStr(""))

    with pytest.raises(ValueError, match="LLM_API_KEY is missing"):
        LLMClient(settings=settings)


def test_missing_model_is_rejected_before_client_creation():
    settings = make_settings(llm_model="")

    with pytest.raises(ValueError, match="LLM_MODEL is missing"):
        LLMClient(settings=settings)


def test_transient_timeout_is_retried():
    fake_provider = MagicMock()
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "Recovered draft"

    fake_provider.chat.completions.create.side_effect = [
        APITimeoutError(request=None),
        fake_response,
    ]

    settings = make_settings(llm_max_retries=1)
    client = LLMClient(settings=settings, client=fake_provider)

    result = client.generate(
        [{"role": "user", "content": "Draft a note."}]
    )

    assert result == "Recovered draft"
    assert fake_provider.chat.completions.create.call_count == 2


def test_unexpected_provider_error_propagates():
    fake_provider = MagicMock()
    fake_provider.chat.completions.create.side_effect = RuntimeError(
        "Unexpected provider failure"
    )

    client = LLMClient(
        settings=make_settings(),
        client=fake_provider,
    )

    with pytest.raises(RuntimeError, match="Unexpected provider failure"):
        client.generate([{"role": "user", "content": "Hello"}])


def test_oversized_prompt_is_rejected_before_provider_request():
    settings = make_settings(
        max_input_tokens=1000,
        max_output_tokens=100,
        context_safety_margin=100,
    )
    client, provider = make_client(settings=settings)

    messages = [
        {
            "role": "user",
            "content": "clinical information " * 1000,
        }
    ]

    with pytest.raises(
        ContextLengthError,
        match="exceeds the safe input-token budget",
    ):
        client.generate(messages)

    provider.chat.completions.create.assert_not_called()

