"""Coordinate prompt construction, model generation, and output validation."""

from src.core.exceptions import LLMResponseError
from src.prompts.prompt_builder import PromptBuilder
from src.schemas.clinical_note import ClinicalNote
from src.services.llm_client import LLMClient
from src.services.output_validator import OutputValidator


class NoteGenerationService:
    """Generate and validate clinical note drafts."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        output_validator: OutputValidator | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.output_validator = output_validator or OutputValidator()

    def _generate_raw(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None,
    ) -> str:
        """Generate model output while preserving the configured token limit."""
        if max_tokens is None:
            return self.llm_client.generate(messages)

        return self.llm_client.generate(messages, max_tokens=max_tokens)

    def generate(
        self,
        source_notes: str,
        note_type: str = "Clinical Note",
        clinician_role: str = "Practitioner",
        max_tokens: int | None = None,
    ) -> ClinicalNote:
        """Generate a validated draft with one correction attempt if needed."""
        messages = self.prompt_builder.build_messages(
            source_notes=source_notes,
            note_type=note_type,
            clinician_role=clinician_role,
        )

        raw_output = self._generate_raw(messages, max_tokens)

        try:
            note = self.output_validator.validate(raw_output)
        except LLMResponseError:
            # Retry only when the returned content fails output validation.
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed JSON or clinical-note "
                        "schema validation. Return a corrected response as "
                        "one valid JSON object matching the required schema. "
                        "Return JSON only, without Markdown fences or extra "
                        "text. Use only facts present in the original source "
                        "notes. Do not invent findings, diagnoses, history, "
                        "or treatment plans. Represent missing information "
                        "according to the original instructions."
                    ),
                },
            ]

            raw_output = self._generate_raw(repair_messages, max_tokens)
            note = self.output_validator.validate(raw_output)

        if not isinstance(note, ClinicalNote):
            raise LLMResponseError(
                "The output validator did not return a valid clinical note."
            )

        return note
