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

    try:
        return load_real_transcript(path)
    except TranscriptValidationError as error:
        pytest.fail(
            "\n"
            "TRANSCRIPT NOT SCORABLE\n"
            "The captured transcript is not a complete, current "
            "interview - refusing to judge it as one.\n"
            f"{error}\n"
            "Root cause lives upstream: check the bot-"
            "responsiveness E2E result/artifacts (it classifies "
            "whether the interview died from a bot failure, a "
            "LiveKit/audio issue, or a capture problem), then "
            "re-run the capture:\n"
            "    pytest tests/e2e/test_bot_responsiveness.py -s\n"
        )


def real_interview_test_case() -> LLMTestCase:
    return LLMTestCase(
        input=INTERVIEW_CONTEXT,
        actual_output=require_real_transcript(),
    )
