"""Parse and validate structured clinical note output."""

import json
import re

from pydantic import ValidationError

from src.core.exceptions import LLMResponseError
from src.schemas.clinical_note import ClinicalNote


class OutputValidator:
    """Validate model output before it reaches the application UI."""

    @staticmethod
    def validate(raw_output: str) -> ClinicalNote:
        """Parse model JSON and validate it against the clinical note schema."""
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise LLMResponseError("The model response is empty.")

        cleaned = raw_output.strip()

        # Remove a Markdown code fence when it wraps the complete response.
        cleaned = re.sub(
            r"\A```(?:json)?\s*|\s*```\Z",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        try:
            return ClinicalNote.model_validate_json(cleaned)
        except ValidationError as original_error:
            # Some models add explanatory text before or after their JSON.
            # Extract one JSON object, then apply the same strict schema.
            decoder = json.JSONDecoder()
            parsed_object = None

            for index, character in enumerate(cleaned):
                if character != "{":
                    continue

                try:
                    candidate, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue

                if isinstance(candidate, dict):
                    parsed_object = candidate
                    break

            if parsed_object is None:
                raise LLMResponseError(
                    "The model response is not valid JSON. "
                    "Please retry or check the model provider response."
                ) from original_error

            try:
                return ClinicalNote.model_validate(parsed_object)
            except ValidationError as exc:
                # Report field names and error types, not patient text.
                issues = [
                    f"{'.'.join(str(part) for part in error['loc'])}: "
                    f"{error['type']}"
                    for error in exc.errors()
                ]
                details = "; ".join(issues[:8])

                raise LLMResponseError(
                    "The model returned JSON, but it does not match the "
                    f"required clinical note schema. Validation issues: {details}"
                ) from exc

