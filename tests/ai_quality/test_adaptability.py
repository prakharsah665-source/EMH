"""
Adaptability of the REAL interviewer.

Judges whether the interviewer adapts to the candidate's level
across the real captured transcript (TranscriptCapture): probing
deeper on strong answers, simplifying on confusion, correcting
wrong answers professionally. Skips with a capture-layer reason
when capture provenance is insufficient - synthetic data is
never scored silently.

min_confidence note: adaptation judgments concern the gist of
each exchange and survive imperfect transcription, so "medium"
provenance suffices and no sources are excluded.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_adaptability():
    judge_real_dimension(
        name="adaptability",
        criteria=(
            "Evaluate whether the AI interviewer in the actual "
            "output adapts to the candidate's demonstrated "
            "level: asking deeper, more challenging follow-ups "
            "after strong answers; rephrasing in simpler, more "
            "concrete language when the candidate is confused "
            "or gives little detail; giving concrete examples "
            "when clarification is requested; and briefly, "
            "professionally addressing incorrect answers before "
            "re-checking understanding with a simpler question. "
            "Penalize responses that ignore the candidate's "
            "situation, invent candidate information, or follow "
            "a fixed script regardless of the answers. Judge "
            "ONLY the turns labelled 'AI Interviewer'."
        ),
        min_confidence="medium",
    )
