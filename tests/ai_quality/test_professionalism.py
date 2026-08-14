"""
Professionalism of the REAL interviewer.

Judges the interviewer's tone across the real captured
transcript (TranscriptCapture). Skips with a capture-layer
reason when no sufficiently-provenanced interviewer text was
captured - a hand-written ideal transcript is never scored in
its place (the bot cannot fail a test whose input the test
author wrote).
"""

import pytest

from evaluators.real_capture_case import judge_real_dimension


@pytest.mark.ai_evaluation
def test_real_interviewer_professionalism():
    judge_real_dimension(
        name="professionalism",
        criteria=(
            "Evaluate whether the AI interviewer in the actual "
            "output remains professional, respectful, clear and "
            "appropriately conversational throughout the real "
            "interview transcript. Penalize rudeness, mockery, "
            "dismissiveness, unprofessional slang, or an "
            "interrogation-like tone. Judge ONLY the turns "
            "labelled 'AI Interviewer'."
        ),
        # Tone judgments survive imperfect transcription;
        # medium provenance (DOM-attributed or better) is
        # required so UI chrome is not judged as speech.
        min_confidence="medium",
    )
