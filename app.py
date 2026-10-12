"""Streamlit interface for the Smart Medical Note Assistant."""

import json

import streamlit as st

from src.config import get_settings
from src.core.exceptions import (
    ApplicationError,
    LLMClientError,
)
from src.services.note_generation_service import NoteGenerationService


st.set_page_config(
    page_title="Smart Medical Note Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("Smart Medical Note Assistant")
st.caption("AI-assisted clinical documentation")

st.warning(
    "Clinical documentation draft only. A qualified clinician must "
    "review and approve every generated note before clinical use."
)

st.markdown(
    """
    Organize practitioner-provided information into a structured clinical
    note. The assistant is designed to preserve supplied facts and identify
    missing information rather than invent clinical details.
    """
)

with st.expander("Privacy and safe use", expanded=True):
    st.markdown(
        """
        - Enter only information you are authorized to process.
        - Use fictional information when testing or demonstrating the app.
        - Do not enter unnecessary patient identifiers.
        - Generated content is a draft, not a diagnosis or treatment decision.
        - Review every field for accuracy before using or sharing the note.
        - Downloaded JSON files may contain sensitive information; store them
          securely.
        """
    )

with st.form("clinical_note_form"):
    st.subheader("Source information")

    source_notes = st.text_area(
        "Practitioner notes",
        placeholder=(
            "Example: Patient reports a headache since yesterday. "
            "No examination findings or treatment plan were provided."
        ),
        height=220,
        help=(
            "Enter the information supplied by the practitioner. "
            "Use fictional details for demonstrations."
        ),
    )

    col1, col2 = st.columns(2)

    with col1:
        note_type = st.selectbox(
            "Document type",
            options=[
                "Clinical Note",
                "Progress Note",
                "Consultation Note",
                "SOAP Note",
                "Discharge Summary",
            ],
        )

    with col2:
        clinician_role = st.selectbox(
            "Author role",
            options=[
                "Practitioner",
                "Physician",
                "Nurse",
                "Medical Student",
                "Other Clinical Professional",
            ],
        )

    submitted = st.form_submit_button(
        "Generate Clinical Note",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not source_notes.strip():
        st.error("Enter practitioner notes before generating a draft.")
    else:
        try:
            with st.spinner("Generating and validating the draft..."):
                service = NoteGenerationService()
                note = service.generate(
                    source_notes=source_notes,
                    note_type=note_type,
                    clinician_role=clinician_role,
                )

            st.session_state["generated_note"] = note.model_dump()
            st.session_state["generated_note_type"] = note_type
            st.session_state["generated_author_role"] = clinician_role

        except ValueError as exc:
            st.error(str(exc))

        except ApplicationError as exc:
            st.error(str(exc))

        except LLMClientError:
            st.error(
                "The model request could not be completed. Check the "
                "provider configuration and availability, then try again."
            )

        except Exception:
            st.error(
                "An unexpected error occurred. No draft was generated. "
                "Check the application terminal for diagnostic information."
            )

if "generated_note" in st.session_state:
    note_data = st.session_state["generated_note"]
    saved_note_type = st.session_state.get(
        "generated_note_type",
        "Clinical Note",
    )
    saved_author_role = st.session_state.get(
        "generated_author_role",
        "Practitioner",
    )

    st.divider()
    st.subheader("Generated draft")

    st.info(
        "Review all fields. Empty lists mean no entries were returned for "
        "that section; they do not confirm that the information is absent."
    )

    narrative_fields = [
        ("chief_complaint", "Chief complaint"),
        ("history_of_present_illness", "History of present illness"),
        ("relevant_medical_history", "Relevant medical history"),
        ("assessment", "Documented assessment"),
    ]

    for field_name, label in narrative_fields:
        st.markdown(f"**{label}**")
        st.write(note_data[field_name])

    list_fields = [
        ("symptoms", "Symptoms"),
        ("objective_findings", "Objective findings"),
        ("plan", "Documented plan"),
        ("missing_information", "Missing information to review"),
    ]

    empty_messages = {
        "symptoms": "No symptoms were documented in the generated draft.",
        "objective_findings": (
            "No objective findings were returned. "
            "This does not confirm that an examination was normal."
        ),
        "plan": (
            "No plan entries were returned. "
            "A qualified clinician must review and document the plan."
        ),
        "missing_information": (
            "No missing information was flagged by the model. "
            "Review the source notes to confirm completeness."
        ),
    }

    for field_name, label in list_fields:
        st.markdown(f"**{label}**")
        values = note_data[field_name]

        if values:
            st.markdown(
                "\n".join(f"- {value}" for value in values)
            )
        else:
            st.caption(empty_messages[field_name])

    st.divider()
    st.subheader("Export")

    export_data = {
        "document_type": saved_note_type,
        "author_role": saved_author_role,
        "status": "Draft - requires qualified clinician review",
        "clinical_note": note_data,
    }

    json_content = json.dumps(
        export_data,
        ensure_ascii=False,
        indent=2,
    )

    st.download_button(
        label="Download note as JSON",
        data=json_content,
        file_name="clinical_note_draft.json",
        mime="application/json",
        use_container_width=True,
    )

    st.warning(
        "Do not use this draft for clinical decisions until a qualified "
        "clinician has reviewed and approved it."
    )

st.divider()
st.caption("Smart Medical Note Assistant | Clinical review required")



