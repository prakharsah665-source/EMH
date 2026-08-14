"""
Context retention of the REAL interviewer.

Judges whether the interviewer remembers what the candidate
said earlier across the real captured transcript
(TranscriptCapture). Skips with a capture-layer reason when
capture provenance is insufficient - synthetic data is never
scored silently.

Provenance note: retention judgments depend on the EXACT
earlier candidate content, so stt-derived text is rejected
(exclude_sources=("stt-local",)) - STT normalisation can make
the interviewer appear to forget or contradict details it was
actually given. "medium" confidence or better is required.
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_context_retention():
    judge_real_dimension(
        name="context_retention",
        criteria=(
            "Evaluate whether the AI interviewer in the actual "
            "output retains and correctly uses information the "
            "candidate provided earlier in the interview "
            "(experience, projects, technologies). Later "
            "interviewer turns should build on those details "
            "without contradicting them, misattributing them, "
            "or re-asking for information the candidate already "
            "gave. Penalize forgetting, contradiction, and "
            "re-asking answered questions proportionally to how "
            "often they occur. Judge ONLY the turns labelled "
            "'AI Interviewer'."
        ),
        min_confidence="medium",
        exclude_sources=("stt-local",),
    )
