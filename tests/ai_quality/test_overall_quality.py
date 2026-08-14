"""
Overall holistic quality of the REAL interview.

Judges the real captured transcript (TranscriptCapture)
holistically across relevance, context retention, follow-up
quality, professionalism and pacing. Skips with a
capture-layer reason when capture provenance is insufficient -
synthetic data is never scored silently.

min_confidence note: a holistic judgment tolerates imperfect
transcription of individual turns, so "medium" provenance
suffices and no sources are excluded.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_overall_interview_quality():
    judge_real_dimension(
        name="overall_quality",
        criteria=(
            "Evaluate the overall quality of the AI interview "
            "in the actual output as a whole: questions "
            "relevant to the role and the candidate's stated "
            "experience; correct use of information from "
            "earlier answers without contradiction or "
            "unnecessary repetition; follow-ups that "
            "meaningfully deepen the candidate's previous "
            "responses; a professional, respectful tone "
            "throughout; and natural pacing that progresses "
            "from general experience to deeper technical and "
            "practical questions before concluding. The score "
            "should reflect the interview as a whole, not any "
            "single question. Judge ONLY the turns labelled "
            "'AI Interviewer'."
        ),
        min_confidence="medium",
    )
