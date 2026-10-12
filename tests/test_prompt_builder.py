import pytest

from src.prompts.prompt_builder import PromptBuilder


def test_builder_returns_system_and_user_messages():
    messages = PromptBuilder().build_messages(
        "Patient reports headache since yesterday."
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_prompt_includes_source_notes():
    source = "Patient reports headache since yesterday."

    messages = PromptBuilder().build_messages(source)

    assert source in messages[1]["content"]


def test_prompt_includes_schema_and_safety_instructions():
    messages = PromptBuilder().build_messages("Patient reports headache.")

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert "Never invent symptoms" in system_prompt
    assert "missing_information" in user_prompt
    assert "chief_complaint" in user_prompt


def test_prompt_supports_dynamic_document_type_and_role():
    messages = PromptBuilder().build_messages(
        "Patient reports cough.",
        note_type="Follow-up Note",
        clinician_role="Nurse Practitioner",
    )

    assert "Follow-up Note" in messages[1]["content"]
    assert "Nurse Practitioner" in messages[1]["content"]


@pytest.mark.parametrize(
    "source",
    ["", "   ", "\n\t"],
)
def test_empty_source_notes_are_rejected(source):
    with pytest.raises(ValueError, match="Source notes must not be empty"):
        PromptBuilder().build_messages(source)


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        (
            {"note_type": " "},
            "Note type must not be empty",
        ),
        (
            {"clinician_role": " "},
            "Clinician role must not be empty",
        ),
    ],
)
def test_empty_document_metadata_is_rejected(kwargs, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        PromptBuilder().build_messages("Patient reports cough.", **kwargs)
