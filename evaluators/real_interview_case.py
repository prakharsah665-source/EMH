"""
Shared loader that turns the REAL captured EMH transcript
(artifacts/transcripts/actual_transcript.json, written by the
bot-responsiveness E2E run) into a DeepEval test case.

A real transcript is mandatory: callers get a hard failure with
the capture command instead of a silent synthetic fallback.
"""

import pytest

from deepeval.test_case import LLMTestCase

from evaluation.transcript import (
    get_transcript_path,
    load_real_transcript,
)
from evaluation.transcript_validation import (
    TranscriptValidationError,
    classify_status_for_scoring,
)


INTERVIEW_CONTEXT = (
    "A real technical interview conducted by the EMH AI "
    "interviewer (Jamie). The interviewer is expected to run a "
    "professional, relevant, technically accurate, context-aware "
    "and adaptive multi-turn interview, starting from the "
    "candidate's introduction."
)


def require_real_transcript() -> str:
    """
    Load the captured transcript or fail the calling test with
    the exact command needed to produce it.
    """

    path = get_transcript_path()

    if not path.exists():
        pytest.fail(
            "\n"
            "REAL TRANSCRIPT MISSING\n"
            f"No captured EMH transcript at {path}.\n"
            "Run the E2E capture first:\n"
            "    pytest tests/e2e/test_bot_responsiveness.py -s\n"
            "(or set EMH_TRANSCRIPT_PATH). AI-quality tests do "
            "not score synthetic transcripts."
        )

    print(f"\nUsing real EMH transcript: {path}")

    # Same status rule as the CI gate: incomplete capture ->
    # SKIP (the upstream bot/capture failure is already
    # reported by the capture test - do not duplicate it once
    # per quality dimension); missing/stale status -> FAIL at
    # the capture/session layer.
    verdict, reason = classify_status_for_scoring()
    if verdict == "skip-upstream":
        pytest.skip(f"NOT JUDGED - {reason}")
    if verdict != "ok":
        pytest.fail(
            "\nTRANSCRIPT NOT SCORABLE\n"
            f"{reason}\n"
        )

    try:
        return load_real_transcript(path)
    except TranscriptValidationError as error:
        pytest.skip(
            "NOT JUDGED - the capture is fresh and complete "
            "but its transcript cannot be judged as a whole "
            "interview (CAPTURE LIMITATION, not a bot "
            f"failure): {error} "
            "See docs/bot_text_capture.md."
        )


def real_interview_test_case() -> LLMTestCase:
    return LLMTestCase(
        input=INTERVIEW_CONTEXT,
        actual_output=require_real_transcript(),
    )
