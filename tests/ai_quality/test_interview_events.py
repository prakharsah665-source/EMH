"""
Interview event handling by the REAL interviewer.

Judges how the interviewer handles interview events across the
real captured conversation (TranscriptCapture): start/greeting,
follow-ups after answers, clarification and repeat requests,
off-topic answers, and completion/closing. Skips with a
capture-layer reason when capture provenance is insufficient -
synthetic data is never scored silently.

min_confidence note: event handling is judged on the shape of
each exchange, which survives imperfect transcription, so
"medium" provenance suffices and no sources are excluded.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_event_handling():
    judge_real_dimension(
        name="interview_events",
        criteria=(
            "Evaluate how the AI interviewer in the actual "
            "output handles the interview events that actually "
            "occur in the transcript: opening with a welcoming, "
            "professional greeting and an appropriate "
            "introductory question; acknowledging answers and "
            "asking on-topic follow-ups; clarifying, giving "
            "examples, or repeating a question when the "
            "candidate asks, without answering on the "
            "candidate's behalf; steering off-topic answers "
            "back to the interview; responding gracefully when "
            "the candidate does not know; and closing with "
            "thanks and a clear, natural conclusion. Only judge "
            "events that occur; penalize mishandled events. "
            "Judge ONLY the turns labelled 'AI Interviewer'."
        ),
        min_confidence="medium",
    )
