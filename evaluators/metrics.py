import json
from pathlib import Path

from deepeval.test_case import (
    Turn,
    ConversationalTestCase,
)


TRANSCRIPT_PATH = Path(
    "artifacts/transcripts/actual_transcript.json"
)


def load_transcript():

    if not TRANSCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Transcript not found: {TRANSCRIPT_PATH}\n"
            "Run the complete interview test first."
        )

    with TRANSCRIPT_PATH.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def build_conversation():

    transcript = load_transcript()

    turns = []

    for item in transcript:

        role = item.get("role")
        content = item.get("content")

        if role not in ("assistant", "user"):
            continue

        if not content:
            continue

        turns.append(
            Turn(
                role=role,
                content=content,
            )
        )

    if not turns:
        raise ValueError(
            "Transcript contains no valid conversation turns."
        )

    return ConversationalTestCase(
        scenario=(
            "A technical interview for a "
            "Frontend Developer position."
        ),
        chatbot_role=(
            "A professional AI technical interviewer "
            "conducting a structured software engineering "
            "interview."
        ),
        expected_outcome=(
            "The AI should conduct a relevant, coherent, "
            "professional and technically accurate "
            "multi-turn interview."
        ),
        turns=turns,
    )