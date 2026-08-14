"""
Question relevance of the REAL interviewer.

Judges whether the interviewer's questions fit the role and
the candidate's stated background across the real captured
transcript (TranscriptCapture). Skips with a capture-layer
reason when capture provenance is insufficient - synthetic
data is never scored silently.

min_confidence note: relevance is a topical judgment that
survives imperfect transcription, so "medium" provenance
suffices and no sources are excluded.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_question_relevance():
    judge_real_dimension(
        name="question_relevance",
        criteria=(
            "Evaluate whether the AI interviewer's questions in "
            "the actual output are relevant to the interview "
            "role and to the background, experience and "
            "projects the candidate states in the transcript. "
            "Questions should meaningfully assess skills tied "
            "to that role and stated experience. Penalize "
            "questions that are unrelated, random, overly "
            "generic, or outside the interview context. Judge "
            "ONLY the turns labelled 'AI Interviewer'."
        ),
        min_confidence="medium",
    )
