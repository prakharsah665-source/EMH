"""
Whole-interview quality of the REAL interview.

Judges the real captured transcript (TranscriptCapture) as a
complete interview: coverage of introduction, experience and
technical depth, coherent progression, and a proper closing.
Skips with a capture-layer reason when capture provenance is
insufficient - synthetic data is never scored silently.

min_confidence note: whole-interview structure and flow
survive imperfect transcription, so "medium" provenance
suffices and no sources are excluded.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_full_interview_quality():
    judge_real_dimension(
        name="full_interview_quality",
        criteria=(
            "Evaluate the quality of the actual output as a "
            "complete interview: the interviewer should open "
            "with an introduction, explore the candidate's "
            "experience and projects, and progress coherently "
            "from introductory questions to deeper technical "
            "questions that go beyond simple definitions, "
            "using earlier answers to guide later questions, "
            "and end with a proper closing after gathering "
            "enough information. Score the interview as a "
            "whole, not individual questions; penalize missing "
            "phases, incoherent jumps, and an abrupt or absent "
            "closing. Judge ONLY the turns labelled "
            "'AI Interviewer' for interviewer behavior."
        ),
        min_confidence="medium",
    )
