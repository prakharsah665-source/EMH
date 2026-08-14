"""
AI Quality Test: Technical Accuracy

Purpose:
    Verify that the AI interviewer asks technically accurate
    questions and does not contain incorrect technical
    assumptions.

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


TECHNICAL_ACCURACY_THRESHOLD = 0.70


# ============================================================
# Evaluation model
# ============================================================

def create_evaluation_model():
    """
    Create the NVIDIA Nemotron judge used as the
    DeepEval evaluation judge.
    """

    return get_nvidia_judge()


# ============================================================
# Test data
# ============================================================

def create_technical_accuracy_test_case():
    """
    Create a realistic technical interview scenario.

    The candidate has React and JavaScript experience.
    The interviewer asks a technically valid question
    about React state management.
    """

    interview_context = """
    Interview role:
    Frontend Developer

    Candidate experience:
    The candidate has experience with JavaScript and React.
    The candidate has built a student management application
    using React components and REST APIs.

    Current technical topic:
    React state management.
    """

    interviewer_question = (
        "In your React application, how did you manage "
        "component state, and when would you choose local "
        "component state over a shared state management "
        "solution?"
    )

    return LLMTestCase(
        input=interview_context,
        actual_output=interviewer_question,
    )


# ============================================================
# Evaluation metric
# ============================================================

def create_technical_accuracy_metric(model):
    """
    Create a GEval metric that evaluates the technical
    correctness of the interviewer's question.
    """

    return GEval(
        name="Technical Accuracy",

        criteria=(
            "Evaluate whether the interviewer's technical "
            "question is factually and conceptually accurate. "
            "The question should use correct technical "
            "terminology and should not contain false "
            "assumptions about JavaScript, React, APIs, or "
            "frontend development."
        ),

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        evaluation_steps=[
            (
                "Identify the technical concepts involved "
                "in the interview context."
            ),
            (
                "Identify the technical claims or assumptions "
                "contained in the interviewer's question."
            ),
            (
                "Determine whether those claims are technically "
                "correct."
            ),
            (
                "Check whether technical terminology is used "
                "appropriately."
            ),
            (
                "Check whether the question contains any "
                "misleading or factually incorrect assumptions."
            ),
            (
                "Check whether the question represents a "
                "reasonable technical interview question for "
                "a Frontend Developer."
            ),
            (
                "Give a final score based on factual correctness, "
                "technical terminology, conceptual correctness, "
                "and absence of misleading assumptions."
            ),
        ],

        model=model,
        threshold=TECHNICAL_ACCURACY_THRESHOLD,
    )


# ============================================================
# Test
# ============================================================

def test_technical_accuracy():
    """
    Verify that the AI interviewer asks a technically
    accurate frontend development question.
    """

    print()
    print("=" * 70)
    print("AI TECHNICAL ACCURACY TEST")
    print("=" * 70)

    print()
    print(
        f"Evaluation model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA base URL: {NVIDIA_BASE_URL}"
    )

    print(
        f"Quality threshold: "
        f"{TECHNICAL_ACCURACY_THRESHOLD}"
    )

    # --------------------------------------------------------
    # 1. Create evaluation model
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
    print("Creating technical interview test case...")

    test_case = (
        create_technical_accuracy_test_case()
    )

    print(
        "[PASS] Technical interview test case created."
    )

    # --------------------------------------------------------
    # 3. Create metric
    # --------------------------------------------------------

    print()
    print("Creating technical accuracy metric...")

    metric = create_technical_accuracy_metric(
        model
    )

    print(
        "[PASS] Technical accuracy metric configured."
    )

    # --------------------------------------------------------
    # 4. Run evaluation
    # --------------------------------------------------------

    print()
    print(
        "Evaluating technical accuracy..."
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
    print("TECHNICAL ACCURACY TEST PASSED")
    print("=" * 70)