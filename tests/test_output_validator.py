import json

import pytest

from src.core.exceptions import LLMResponseError
from src.schemas.clinical_note import ClinicalNote
from src.services.output_validator import OutputValidator


def valid_payload():
    return {
        "chief_complaint": "Headache",
        "history_of_present_illness": "Headache since yesterday.",
        "relevant_medical_history": "Not documented in the supplied notes.",
        "symptoms": ["Headache"],
        "objective_findings": [],
        "assessment": "No explicit assessment documented in the supplied notes.",
        "plan": [],
        "missing_information": ["Blood pressure not provided"],
    }


def test_valid_json_returns_clinical_note():
    raw_output = json.dumps(valid_payload())

    result = OutputValidator.validate(raw_output)

    assert isinstance(result, ClinicalNote)
    assert result.chief_complaint == "Headache"
    assert result.symptoms == ["Headache"]


def test_validated_output_preserves_all_fields():
    result = OutputValidator.validate(json.dumps(valid_payload()))

    assert set(result.model_dump().keys()) == set(valid_payload().keys())


def test_invalid_json_is_rejected():
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        OutputValidator.validate('{"chief_complaint": ')


def test_unexpected_fields_are_rejected():
    payload = valid_payload()
    payload["invented_field"] = "Unexpected data"

    with pytest.raises(LLMResponseError, match="required clinical note schema"):
        OutputValidator.validate(json.dumps(payload))


def test_missing_required_field_is_rejected():
    payload = valid_payload()
    del payload["symptoms"]

    with pytest.raises(LLMResponseError, match="required clinical note schema"):
        OutputValidator.validate(json.dumps(payload))


def test_blank_narrative_field_is_rejected():
    payload = valid_payload()
    payload["assessment"] = "   "

    with pytest.raises(LLMResponseError, match="required clinical note schema"):
        OutputValidator.validate(json.dumps(payload))


@pytest.mark.parametrize("raw_output", ["", " ", "\n"])
def test_empty_output_is_rejected(raw_output):
    with pytest.raises(LLMResponseError, match="response is empty"):
        OutputValidator.validate(raw_output)


def test_non_json_markdown_response_is_rejected():
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        OutputValidator.validate("Here is your clinical note.")
