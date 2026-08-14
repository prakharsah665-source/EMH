"""
AI-quality evaluation of the REAL captured EMH interview.

Unlike the scenario tests in this package (which score
controlled synthetic transcripts and act as judge behavior
checks), every test here evaluates the actual transcript
captured from the live interview by the bot-responsiveness E2E
run. A missing capture fails loudly - there is no synthetic
fallback.

Each dimension keeps the package's existing 0.70 threshold and
the existing Nemotron judge, and is scored with a 3-run median
plus run-disagreement reporting (evaluators/consistent_geval).
"""

import pytest

from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

from evaluators.consistent_geval import assert_metric_median
from evaluators.nvidia_judge import get_nvidia_judge
from evaluators.real_interview_case import real_interview_test_case


QUALITY_THRESHOLD = 0.70

EVALUATION_PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
]


def geval_factory(name: str, criteria: str):
    """
    Build a fresh GEval metric per consistency run (fresh
    instance per measurement, same judge and threshold).
    """

    def factory() -> GEval:
        return GEval(
            name=name,
            criteria=criteria,
            evaluation_params=EVALUATION_PARAMS,
            model=get_nvidia_judge(),
            threshold=QUALITY_THRESHOLD,
        )

    return factory


DIMENSIONS = {
    "question_relevance": (
        "Evaluate whether the interviewer's questions in the "
        "actual output are relevant to a professional technical "
        "interview and to the candidate's stated background and "
        "previous answers. Penalize random, generic or "
        "off-context questions."
    ),
    "follow_up_quality": (
        "Evaluate whether the interviewer's follow-up questions "
        "build meaningfully on the candidate's previous answers "
        "rather than ignoring them or switching topics abruptly."
    ),
    "context_retention": (
        "Evaluate whether the interviewer correctly remembers "
        "and uses information the candidate provided earlier in "
        "the interview, without contradictions or unnecessary "
        "repetition of already-answered questions."
    ),
    "professionalism": (
        "Evaluate whether the interviewer remains professional, "
        "respectful, clear and appropriately conversational "
        "throughout the interview."
    ),
    "technical_accuracy": (
        "Evaluate whether the interviewer's technical statements "
        "and reactions to the candidate's answers are correct. "
        "Penalize technically wrong claims or acceptance of "
        "clearly wrong answers as correct."
    ),
    "question_repetition": (
        "Evaluate whether the interviewer avoids repeating "
        "substantially the same question. Asking for the same "
        "information again that the candidate already provided "
        "should be penalized."
    ),
    "adaptability": (
        "Evaluate whether the interviewer adapts its questions "
        "to the candidate's demonstrated level and answers - "
        "probing deeper on strong answers and adjusting when "
        "the candidate struggles or gives short answers."
    ),
    "interview_flow": (
        "Evaluate the overall conversational flow: the "
        "interviewer greets the candidate, elicits an "
        "introduction, progresses through coherent topics, and "
        "does not stall, go silent, or produce broken or "
        "truncated turns."
    ),
}


@pytest.mark.ai_evaluation
@pytest.mark.parametrize("dimension", sorted(DIMENSIONS))
def test_real_interview_dimension(dimension):
    test_case = real_interview_test_case()

    assert_metric_median(
        geval_factory(dimension, DIMENSIONS[dimension]),
        test_case,
    )


@pytest.mark.ai_evaluation
def test_real_interview_overall_quality():
    test_case = real_interview_test_case()

    assert_metric_median(
        geval_factory(
            "overall_quality",
            (
                "Evaluate the overall quality of this real AI-led "
                "technical interview considering relevance, "
                "context retention, follow-up quality, "
                "adaptability, technical accuracy, "
                "professionalism and completeness."
            ),
        ),
        test_case,
    )
