"""
Shared loader that turns the REAL captured EMH transcript into a
DeepEval test case for the whole-interview quality tests.

Reads through the TranscriptCapture layer (select_capture), NOT
the DOM-scrape file artifacts/transcripts/actual_transcript.json:
the app renders no bot text in the DOM, so that file never holds
interviewer turns and every whole-interview judgment built on it
skipped as "TRUNCATED - 0 interviewer turns". The LiveKit data
channel (preferred backend) carries both the bot's captions and
the agent's own STT of the candidate, identity-tagged.

A real capture is mandatory: callers get a hard failure with the
capture command instead of a silent synthetic fallback. Status,
provenance, coverage and minimum-turn gates are the ones shared
with the per-dimension tests (evaluators.real_capture_case).
"""

import pytest

from deepeval.test_case import LLMTestCase

from collectors.transcript_capture import select_capture
from evaluators.real_capture_case import (
    INTERVIEW_CONTEXT,
    format_capture_turns,
    real_capture_turns,
)

__all__ = [
    "INTERVIEW_CONTEXT",
    "require_real_transcript",
    "real_interview_test_case",
]


def require_real_transcript(
    *,
    test_label: str = "real_interview",
    min_confidence: str = "medium",
    exclude_sources: tuple[str, ...] = (),
) -> str:
    """
    Load the captured transcript (provenance-tagged, evaluator-
    ready text) or fail the calling test with the exact command
    needed to produce it.
    """

    try:
        capture = select_capture()
    except RuntimeError as error:
        pytest.fail(
            "\n"
            "REAL TRANSCRIPT MISSING\n"
            f"No usable EMH capture backend: {error}\n"
            "Run the E2E capture first:\n"
            "    pytest tests/e2e/test_bot_responsiveness.py -s\n"
            "AI-quality tests do not score synthetic transcripts."
        )

    print(f"\nUsing real EMH transcript backend: {capture.name}")

    # Status (fresh/complete), provenance floor, capture-coverage
    # and minimum-turn gates - identical to the per-dimension
    # tests so both families judge the same turns.
    turns = real_capture_turns(
        test_label=test_label,
        min_confidence=min_confidence,
        exclude_sources=exclude_sources,
    )
    return format_capture_turns(turns)


def real_interview_test_case(**kwargs) -> LLMTestCase:
    return LLMTestCase(
        input=INTERVIEW_CONTEXT,
        actual_output=require_real_transcript(**kwargs),
    )
