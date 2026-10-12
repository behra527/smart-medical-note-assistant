"""Pydantic schemas for structured clinical note drafts."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClinicalNote(BaseModel):
    """Structured clinical note draft from practitioner-provided notes."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    chief_complaint: str = Field(
        description="Primary reason for the clinical encounter.",
    )
    history_of_present_illness: str = Field(
        description="Relevant history explicitly supplied by the practitioner.",
    )
    relevant_medical_history: str = Field(
        description="Previously documented medical history.",
    )
    symptoms: list[str] = Field(
        description="Symptoms explicitly reported in the supplied notes.",
    )
    objective_findings: list[str] = Field(
        description="Documented measurements and examination findings.",
    )
    assessment: str = Field(
        description="Clinical assessment explicitly documented by the practitioner.",
    )
    plan: list[str] = Field(
        description="Documented treatment, recommendations, and follow-up.",
    )
    missing_information: list[str] = Field(
        description="Relevant information not supplied in the source notes.",
    )

    @field_validator(
        "chief_complaint",
        "history_of_present_illness",
        "relevant_medical_history",
        "assessment",
    )
    @classmethod
    def reject_empty_sections(cls, value: str) -> str:
        """Prevent blank narrative sections."""
        if not value:
            raise ValueError(
                "Narrative sections cannot be empty; use an explicit "
                "not-documented statement instead."
            )
        return value

    @field_validator(
        "symptoms",
        "objective_findings",
        "plan",
        "missing_information",
    )
    @classmethod
    def normalize_list_items(cls, values: list[str]) -> list[str]:
        """Remove blank list entries and normalize whitespace."""
        return [item.strip() for item in values if item.strip()]
