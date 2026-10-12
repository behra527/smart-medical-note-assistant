# Smart Medical Note Assistant

An AI-powered Streamlit application that transforms practitioner-provided notes into structured clinical note drafts using an LLM API, dynamic prompting, and schema validation.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white" alt="Pydantic">
  <img src="https://img.shields.io/badge/Pytest-69%20Tests%20Passed-0A9EDC?logo=pytest&logoColor=white" alt="Pytest">
  <img src="https://img.shields.io/badge/LLM-OpenRouter-7C3AED" alt="OpenRouter">
  <img src="https://img.shields.io/badge/Status-Under%20Development-orange" alt="Under Development">
</p>

## Overview

Smart Medical Note Assistant helps organize practitioner notes into a structured clinical format. It combines dynamic prompt construction, model integration, input safeguards, and output validation to support AI-assisted documentation.

## Features

- Structured clinical note generation
- Dynamic prompts based on note type and clinician role
- OpenRouter integration through an OpenAI-compatible API
- Environment-based API configuration
- Context-length validation
- JSON parsing and Pydantic schema validation
- Retry for invalid model output
- JSON export
- Automated test suite

## Tech Stack

- Python 3.13
- Streamlit
- OpenAI Python SDK
- OpenRouter API
- Pydantic
- Pytest

## Getting Started

### Prerequisites

- Python 3.13
- An API key for a supported model provider

### Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/behra527/smart-medical-note-assistant.git
cd smart-medical-note-assistant
```

Create and activate a virtual environment on Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root using `.env.example` as a reference.

Configure the environment variables expected by the application, including:

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openrouter/free
```

Use the exact variable names defined in the project's settings module. Never commit API keys or the `.env` file to version control.

### Run the Application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Testing

Run the test suite:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest -v
```

The latest recorded test run completed with **69 passed**. Run the tests again after code changes to verify the current state.

## Clinical Note Structure

Generated drafts use the following fields:

- Chief complaint
- History of present illness
- Relevant medical history
- Symptoms
- Objective findings
- Assessment
- Plan
- Missing information

The output is validated against a Pydantic schema before it is returned by the generation service.

## Limitations and Safety

- This project is a development prototype, not a diagnostic or treatment system.
- Generated notes may contain errors or omissions and require review by a qualified clinician.
- Missing information must not be interpreted as a normal finding.
- Use fictional or appropriately de-identified information during development and testing.
- Do not submit identifiable patient information to external providers without appropriate authorization and privacy safeguards.
- Provider availability and JSON response behavior may vary by model.
- Passing automated tests does not establish clinical safety or production readiness.

## Disclaimer

For educational and software development purposes only. This application does not replace professional medical judgment. Every generated note must be reviewed by a qualified healthcare professional before clinical use.

