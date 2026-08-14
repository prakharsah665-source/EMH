"""
Follow-up question quality of the REAL interviewer.

Judges whether the interviewer's follow-ups build on the
candidate's previous answers across the real captured
transcript (TranscriptCapture). Skips with a capture-layer
reason when capture provenance is insufficient - synthetic
data is never scored silently.

min_confidence note: follow-up quality is judged on whether a
question engages the substance of the prior answer, which
survives imperfect transcription, so "medium" provenance
suffices and no sources are excluded.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_follow_up_quality():
    judge_real_dimension(
        name="follow_up_quality",
        criteria=(
            "Evaluate whether the AI interviewer's follow-up "
            "questions in the actual output are relevant to and "
            "build on the candidate's immediately preceding "
            "answers: they should demonstrate that the specific "
            "response was understood and meaningfully explore "
            "or deepen the topic the candidate raised. Penalize "
            "follow-ups that ignore the candidate's answer, "
            "switch abruptly to an unrelated topic, or are so "
            "generic they could follow any answer. Judge ONLY "
            "the turns labelled 'AI Interviewer'."
        ),
        min_confidence="medium",
    )
