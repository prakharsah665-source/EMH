"""
Technical accuracy of the REAL interviewer.

Judges whether the interviewer's technical statements across
the real captured transcript (TranscriptCapture) are correct.
Skips with a capture-layer reason when capture provenance is
insufficient - synthetic data is never scored silently.

min_confidence note: "high" is required - judging technical
claims needs VERBATIM interviewer text; paraphrased or
heuristically-attributed text can fabricate technical errors
(or hide real ones) that the bot never made.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_technical_accuracy():
    judge_real_dimension(
        name="technical_accuracy",
        criteria=(
            "Evaluate whether the AI interviewer's technical "
            "statements, questions and assumptions in the "
            "actual output are factually and conceptually "
            "correct, with technical terminology used "
            "appropriately and no false or misleading premises. "
            "Also penalize the interviewer for accepting or "
            "endorsing a candidate answer that is clearly "
            "technically wrong instead of addressing the "
            "misconception. Judge ONLY the turns labelled "
            "'AI Interviewer'."
        ),
        min_confidence="high",
    )
