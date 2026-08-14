"""
Loader for real EMH interview transcripts.

The E2E interview run stores the captured conversation as
JSON at artifacts/transcripts/actual_transcript.json
(override with the EMH_TRANSCRIPT_PATH environment variable).

Expected JSON structure:

    [
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."}
    ]

"assistant" turns are the AI interviewer and "user" turns
are the candidate. The loader converts these turns into the
plain-text transcript format consumed by the rubric
evaluator in evaluator.py.
"""

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_TRANSCRIPT_PATH = Path(
    "artifacts/transcripts/actual_transcript.json"
)


ROLE_LABELS = {
    "assistant": "AI Interviewer",
    "user": "Candidate",
}


def get_transcript_path() -> Path:
    """
    Return the transcript location, honouring the
    EMH_TRANSCRIPT_PATH environment variable.
    """

    override = os.getenv("EMH_TRANSCRIPT_PATH")

    if override:
        return Path(override)

    return DEFAULT_TRANSCRIPT_PATH


def load_transcript_turns(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load and validate the raw transcript JSON.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {path}\n"
            "Run the complete interview E2E test first, "
            "or set EMH_TRANSCRIPT_PATH."
        )

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{path}: transcript must be a JSON list of turns."
        )

    turns: list[dict[str, Any]] = []

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            raise ValueError(
                f"{path}: turn {index} must be a JSON object."
            )

        role = item.get("role")
        content = item.get("content")

        if role not in ROLE_LABELS:
            # Ignore system / tool / metadata entries.
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        turns.append(
            {
                "role": role,
                "content": content.strip(),
            }
        )

    if not turns:
        raise ValueError(
            f"{path}: transcript contains no valid "
            "conversation turns."
        )

    return turns


def format_transcript(
    turns: list[dict[str, Any]],
) -> str:
    """
    Convert transcript turns into the plain-text format
    expected by the rubric evaluator.
    """

    blocks = [
        f"{ROLE_LABELS[turn['role']]}:\n{turn['content']}"
        for turn in turns
    ]

    return "\n\n".join(blocks)


def load_real_transcript(
    path: Path | None = None,
    *,
    validate: bool = True,
) -> str:
    """
    Load the real EMH interview transcript as evaluator-ready
    plain text. Raises if the transcript is missing or invalid.

    By default the transcript is also validated for freshness,
    completeness and minimum turn counts (transcript_validation),
    so an empty, stale or truncated capture is rejected loudly
    instead of being silently scored as a whole interview.
    Callers that deliberately inspect partial captures can pass
    validate=False.
    """

    resolved_path = path or get_transcript_path()

    turns = load_transcript_turns(resolved_path)

    if validate:
        # Imported here (not at module top) so unit tests of the
        # loader itself can exercise parsing without the status
        # sidecar machinery.
        from evaluation.transcript_validation import (
            transcript_status_line,
            validate_transcript_turns,
        )

        validate_transcript_turns(turns)
        print(transcript_status_line(turns))

    return format_transcript(turns)
