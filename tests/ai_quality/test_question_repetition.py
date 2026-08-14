"""
Question repetition by the REAL interviewer.

Judges whether the interviewer repeats substantially the same
question across the real captured transcript
(TranscriptCapture).

min_source note: stt-derived text is REJECTED for this metric
(exclude_sources=("stt-local",)): STT normalisation can both
fabricate repetition (two different questions transcribed to
similar text) and hide it (the same question transcribed
differently). Repetition judgments need verbatim channels
(LiveKit transcription/chat, app API, DOM).
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_avoids_question_repetition():
    judge_real_dimension(
        name="question_repetition",
        criteria=(
            "Evaluate whether the AI interviewer in the actual "
            "output avoids re-asking substantially the same "
            "question. Asking again for information the "
            "candidate already provided must be penalized "
            "proportionally to how often it happens. Judge ONLY "
            "the turns labelled 'AI Interviewer'; rephrased "
            "follow-ups that genuinely deepen a topic are not "
            "repetition."
        ),
        min_confidence="medium",
        exclude_sources=("stt-local",),
    )
