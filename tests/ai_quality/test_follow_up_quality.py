"""
AI Quality Test: Follow-Up Question Quality

Purpose:
    Verify that the AI interviewer asks a relevant,
    context-aware follow-up question based on the
    candidate's previous answer.

Evaluation model:
    NVIDIA Nemotron (OpenAI-compatible API)
"""

from deepeval import assert_test
from config.settings import NVIDIA_BASE_URL, NVIDIA_MODEL
from evaluators.nvidia_judge import get_nvidia_judge
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)

from deepeval.metrics import GEval


# ============================================================
# Configuration
# ============================================================


FOLLOW_UP_THRESHOLD = 0.70


# ============================================================
# Evaluation model
# ============================================================

def create_evaluation_model():
    """
    Create the NVIDIA Nemotron judge used as the
    DeepEval judge.
    """

    return get_nvidia_judge()


# ============================================================
# Test data
# ============================================================

def create_follow_up_test_case():
    """
    Create a realistic frontend interview scenario.

    INPUT:
        Candidate's previous answer.

    ACTUAL_OUTPUT:
        AI interviewer's follow-up question.
    """

    candidate_answer = (
        "In my student management application, I used React "
        "to build reusable components for students, courses, "
        "and attendance. I also integrated REST APIs to fetch "
        "and update student data."
    )

    interviewer_follow_up = (
        "You mentioned that you integrated REST APIs. "
        "How did you handle API calls and loading or error "
        "states in your application?"
    )

    return LLMTestCase(
        input=candidate_answer,
        actual_output=interviewer_follow_up,
    )


# ============================================================
# Evaluation metric
# ============================================================

def create_follow_up_metric(model):
    """
    Create a GEval metric for evaluating the interviewer's
    follow-up question.
    """

    return GEval(
        name="Follow-Up Question Quality",

        criteria=(
            "Evaluate whether the interviewer's follow-up "
            "question is relevant to the candidate's previous "
            "answer. The interviewer should demonstrate that "
            "they understood the candidate's specific response. "
            "The question should meaningfully explore or deepen "
            "the topic discussed instead of changing to an "
            "unrelated topic."
        ),

        evaluation_params=[
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
],

        evaluation_steps=[
            (
                "Identify the main technical information "
                "provided in the candidate's answer."
            ),
            (
                "Identify the subject of the interviewer's "
                "follow-up question."
            ),
            (
                "Check whether the follow-up question is "
                "directly related to the candidate's answer."
            ),
            (
                "Check whether the question demonstrates "
                "understanding of the candidate's specific "
                "experience."
            ),
            (
                "Check whether the question meaningfully "
                "explores the candidate's technical knowledge."
            ),
            (
                "Give a score based on relevance, context "
                "awareness, specificity, and technical depth."
            ),
        ],

        model=model,
        threshold=FOLLOW_UP_THRESHOLD,
    )


# ============================================================
# Test
# ============================================================

def test_follow_up_quality():
    """
    Verify that the AI interviewer asks a relevant
    and context-aware follow-up question.
    """

    print()
    print("=" * 70)
    print("AI FOLLOW-UP QUALITY TEST")
    print("=" * 70)

    print()
    print(
        f"Evaluation model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA base URL: {NVIDIA_BASE_URL}"
    )

    print(
        f"Quality threshold: {FOLLOW_UP_THRESHOLD}"
    )

    # --------------------------------------------------------
    # 1. Create evaluator
    # --------------------------------------------------------

    print()
    print("Creating evaluation model...")

    model = create_evaluation_model()

    print(
        "[PASS] NVIDIA Nemotron evaluation model configured."
    )

    # --------------------------------------------------------
    # 2. Create test case
    # --------------------------------------------------------

    print()
    print("Creating interview test case...")

    test_case = create_follow_up_test_case()

    print(
        "[PASS] Interview test case created."
    )

    # --------------------------------------------------------
    # 3. Create metric
    # --------------------------------------------------------

    print()
    print("Creating follow-up quality metric...")

    metric = create_follow_up_metric(
        model
    )

    print(
        "[PASS] Follow-up quality metric configured."
    )

    # --------------------------------------------------------
    # 4. Run evaluation
    # --------------------------------------------------------

    print()
    print(
        "Evaluating follow-up question quality..."
    )

    assert_test(
        test_case,
        [metric],
    )

    # --------------------------------------------------------
    # 5. Completed
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FOLLOW-UP QUALITY TEST PASSED")
    print("=" * 70)