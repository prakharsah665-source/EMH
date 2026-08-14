"""
Completeness of the REAL interview.

Judges breadth and depth of the real captured interview
(TranscriptCapture): enough distinct topics for the role,
sufficient probing per topic, and no premature ending. Skips
with a capture-layer reason when capture provenance is
insufficient - synthetic data is never scored silently.

min_confidence note: breadth/depth judgments concern which
topics were covered and how far each was probed, which
survives imperfect transcription, so "medium" provenance
suffices and no sources are excluded.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interview_completeness():
    judge_real_dimension(
        name="interview_completeness",
        criteria=(
            "Evaluate whether the AI interviewer in the actual "
            "output conducts a sufficiently complete interview: "
            "covering a reasonable breadth of distinct topics "
            "relevant to the role and the candidate's stated "
            "experience (experience, projects, core technical "
            "fundamentals, practical problem solving), and "
            "probing each important topic with follow-ups that "
            "explore reasoning and tradeoffs rather than only "
            "definitions. Penalize skimming one question per "
            "topic, covering too few topics, and ending the "
            "interview prematurely before enough information "
            "was gathered to evaluate the candidate. Judge ONLY "
            "the turns labelled 'AI Interviewer'."
        ),
        min_confidence="medium",
    )
