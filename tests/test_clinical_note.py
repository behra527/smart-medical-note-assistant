import pytest
from pydantic import ValidationError

from src.schemas.clinical_note import ClinicalNote


def valid_payload():
    return {
        "chief_complaint": "Headache",
        "history_of_present_illness": "Not documented in the supplied notes.",
        "relevant_medical_history": "Not documented in the supplied notes.",
        "symptoms": [],
        "objective_findings": [],
        "assessment": "No explicit assessment documented in the supplied notes.",
        "plan": [],
        "missing_information": [],
    }


def test_valid_clinical_note_is_accepted():
    note = ClinicalNote(**valid_payload())

    assert note.chief_complaint == "Headache"
    assert note.symptoms == []


def test_all_expected_fields_are_present():
    note = ClinicalNote(**valid_payload())

    assert set(note.model_dump().keys()) == set(valid_payload().keys())


def test_unexpected_fields_are_rejected():
    payload = valid_payload()
    payload["unknown_field"] = "Unexpected information"

    with pytest.raises(ValidationError):
        ClinicalNote(**payload)


@pytest.mark.parametrize(
    "missing_field",
    [
        "chief_complaint",
        "history_of_present_illness",
        "relevant_medical_history",
        "symptoms",
        "objective_findings",
        "assessment",
        "plan",
        "missing_information",
    ],
)
def test_missing_fields_are_rejected(missing_field):
    payload = valid_payload()
    del payload[missing_field]

    with pytest.raises(ValidationError):
        ClinicalNote(**payload)


def test_empty_chief_complaint_is_rejected():
    payload = valid_payload()
    payload["chief_complaint"] = "   "

    with pytest.raises(ValidationError):
        ClinicalNote(**payload)


def test_blank_narrative_sections_are_rejected():
    payload = valid_payload()
    payload["assessment"] = "   "

    with pytest.raises(ValidationError):
        ClinicalNote(**payload)


def test_list_items_are_normalized():
    payload = valid_payload()
    payload["symptoms"] = ["  Headache  ", " ", "Nausea"]

    note = ClinicalNote(**payload)

    assert note.symptoms == ["Headache", "Nausea"]


def test_clinical_note_can_store_documented_findings():
    payload = valid_payload()
    payload["symptoms"] = ["Headache since yesterday"]
    payload["objective_findings"] = ["Temperature: 37.0 C"]
    payload["plan"] = ["Follow-up documented by practitioner"]
    payload["missing_information"] = ["Blood pressure not provided"]

    note = ClinicalNote(**payload)

    assert note.symptoms == ["Headache since yesterday"]
    assert note.objective_findings == ["Temperature: 37.0 C"]
    assert note.plan == ["Follow-up documented by practitioner"]
    assert note.missing_information == ["Blood pressure not provided"]