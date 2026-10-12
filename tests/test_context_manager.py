import pytest

from src.config import Settings
from src.core.exceptions import ContextLengthError
from src.utils.context_manager import ContextManager


def make_settings(**overrides):
    values = {
        "max_input_tokens": 100,
        "max_output_tokens": 20,
        "context_safety_margin": 10,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_count_tokens_returns_positive_count():
    manager = ContextManager(settings=make_settings())

    assert manager.count_tokens("Patient reports headache.") > 0


def test_empty_messages_are_rejected():
    manager = ContextManager(settings=make_settings())

    with pytest.raises(ValueError, match="Messages must not be empty"):
        manager.count_message_tokens([])


def test_small_prompt_passes_budget_validation():
    manager = ContextManager(settings=make_settings())

    count = manager.validate_messages(
        [{"role": "user", "content": "Patient reports headache."}]
    )

    assert count > 0
    assert count <= manager.input_token_limit


def test_oversized_prompt_is_rejected():
    manager = ContextManager(settings=make_settings())

    messages = [
        {
            "role": "user",
            "content": "clinical information " * 500,
        }
    ]

    with pytest.raises(ContextLengthError, match="exceeds the safe input-token budget"):
        manager.validate_messages(messages)


def test_safety_margin_reduces_available_input_budget():
    manager = ContextManager(settings=make_settings())

    assert manager.input_token_limit == 90


def test_safety_margin_cannot_equal_input_budget():
    with pytest.raises(
        ValueError,
        match="must be smaller than MAX_INPUT_TOKENS",
    ):
        ContextManager(
            settings=make_settings(
                max_input_tokens=100,
                context_safety_margin=100,
            )
        )


def test_message_count_includes_prompt_instructions():
    manager = ContextManager(settings=make_settings())

    short_messages = [{"role": "user", "content": "Headache"}]
    longer_messages = [
        {
            "role": "system",
            "content": "Follow the clinical documentation safety rules.",
        },
        {
            "role": "user",
            "content": "Patient reports headache since yesterday.",
        },
    ]

    assert (
        manager.count_message_tokens(longer_messages)
        > manager.count_message_tokens(short_messages)
    )


def test_unicode_notes_can_be_counted():
    manager = ContextManager(settings=make_settings())

    assert manager.count_tokens("Patient reports درد.") > 0
