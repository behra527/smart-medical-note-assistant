import json
from unittest.mock import MagicMock

import pytest

from src.core.exceptions import ContextLengthError, LLMResponseError
from src.prompts.prompt_builder import PromptBuilder
from src.schemas.clinical_note import ClinicalNote
from src.services.note_generation_service import NoteGenerationService
from src.services.output_validator import OutputValidator


def make_valid_output():
    return json.dumps(
        {
            "chief_complaint": "Headache",
            "history_of_present_illness": (
                "The practitioner reports a headache since yesterday."
            ),
            "relevant_medical_history": "Not documented in supplied notes.",
            "symptoms": ["Headache"],
            "objective_findings": [],
            "assessment": "Headache, as documented by the practitioner.",
            "plan": [],
            "missing_information": [
                "Relevant medical history",
                "Objective findings",
                "Documented plan",
            ],
        }
    )


def make_service(raw_output=None):
    llm_client = MagicMock()
    llm_client.generate.return_value = (
        make_valid_output() if raw_output is None else raw_output
    )

    service = NoteGenerationService(
        llm_client=llm_client,
        prompt_builder=PromptBuilder(),
        output_validator=OutputValidator(),
    )
    return service, llm_client


def test_generate_returns_validated_clinical_note():
    service, _ = make_service()

    result = service.generate("Patient reports headache since yesterday.")

    assert isinstance(result, ClinicalNote)
    assert result.chief_complaint == "Headache"
    assert result.symptoms == ["Headache"]


def test_generate_sends_built_messages_to_llm():
    service, llm_client = make_service()

    service.generate(
        source_notes="Patient reports headache.",
        note_type="Progress Note",
        clinician_role="Physician",
    )

    llm_client.generate.assert_called_once()
    messages = llm_client.generate.call_args.args[0]

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Patient reports headache." in messages[1]["content"]
    assert "Progress Note" in messages[1]["content"]
    assert "Physician" in messages[1]["content"]


def test_generate_passes_custom_output_token_limit():
    service, llm_client = make_service()

    service.generate(
        source_notes="Patient reports headache.",
        max_tokens=300,
    )

    llm_client.generate.assert_called_once()
    assert llm_client.generate.call_args.kwargs == {"max_tokens": 300}


def test_generate_uses_default_output_token_limit_when_unspecified():
    service, llm_client = make_service()

    service.generate("Patient reports headache.")

    llm_client.generate.assert_called_once_with(
        llm_client.generate.call_args.args[0]
    )
    assert llm_client.generate.call_args.kwargs == {}


def test_empty_source_notes_are_rejected_before_llm_call():
    service, llm_client = make_service()

    with pytest.raises(ValueError, match="Source notes must not be empty"):
        service.generate("   ")

    llm_client.generate.assert_not_called()


def test_invalid_model_output_is_rejected():
    service, _ = make_service("This is not JSON.")

    with pytest.raises(LLMResponseError, match="not valid JSON"):
        service.generate("Patient reports headache.")


def test_context_length_error_propagates():
    service, llm_client = make_service()
    llm_client.generate.side_effect = ContextLengthError(
        "Prompt exceeds the safe input-token budget."
    )

    with pytest.raises(ContextLengthError):
        service.generate("Patient reports headache.")


def test_service_does_not_return_unvalidated_text():
    service, _ = make_service(
        json.dumps(
            {
                "chief_complaint": "Headache",
                "unexpected_field": "Unsupported data",
            }
        )
    )

    with pytest.raises(LLMResponseError):
        service.generate("Patient reports headache.")

def test_invalid_first_response_retries_and_recovers():
    service, llm_client = make_service("This is not JSON.")
    llm_client.generate.side_effect = [
        "This is not JSON.",
        make_valid_output(),
    ]

    result = service.generate("Patient reports headache since yesterday.")

    assert isinstance(result, ClinicalNote)
    assert result.chief_complaint == "Headache"
    assert llm_client.generate.call_count == 2

    repair_messages = llm_client.generate.call_args.args[0]
    assert "This is not JSON." in repair_messages[-2]["content"]
    assert "corrected response" in repair_messages[-1]["content"]
