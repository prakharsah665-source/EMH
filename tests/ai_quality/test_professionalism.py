"""
AI Quality Test: Interviewer Professionalism

Purpose:
    Verify that the AI interviewer communicates in a
    professional, respectful, neutral, and appropriate
    manner throughout an interview.

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


PROFESSIONALISM_THRESHOLD = 0.70


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

def create_professionalism_test_case():
    """
    Create a realistic interview scenario.

    The candidate has answered a technical question and
    the AI interviewer responds with a professional
    follow-up.
    """

    interview_context = """
    Interview role:
    Frontend Developer

    Candidate:
    The candidate is discussing their experience with
    React and a student management application.

    Candidate answer:
    "I built reusable React components for managing
    students and courses. I also integrated REST APIs
    to retrieve and update application data."

    The interviewer should respond professionally,
    respectfully, and neutrally.
    """

    interviewer_response = (
        "That's a useful example. Could you explain how "
        "you handled API errors and loading states in your "
        "React application? Please describe the approach "
        "you used and why you chose it."
    )

    return LLMTestCase(
        input=interview_context,
        actual_output=interviewer_response,
    )


# ============================================================
# Evaluation metric
# ============================================================

def create_professionalism_metric(model):
    """
    Create a GEval metric for interviewer professionalism.
    """

    return GEval(
        name="Interviewer Professionalism",

        criteria=(
            "Evaluate whether the AI interviewer's response "
            "is professional, respectful, neutral, clear, and "
            "appropriate for a technical job interview. The "
            "response should maintain a professional tone "
            "without being rude, dismissive, insulting, "
            "discriminatory, overly casual, or inappropriate."
        ),

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        evaluation_steps=[
            (
                "Identify the context and purpose of the "
                "interviewer's response."
            ),
            (
                "Evaluate whether the interviewer uses a "
                "professional and respectful tone."
            ),
            (
                "Check whether the response is neutral and "
                "free from discriminatory, insulting, or "
                "biased language."
            ),
            (
                "Check whether the interviewer avoids "
                "unnecessary sarcasm, mockery, hostility, "
                "or inappropriate humor."
            ),
            (
                "Check whether the wording is clear and "
                "appropriate for a technical interview."
            ),
            (
                "Check whether the interviewer maintains "
                "appropriate professional boundaries."
            ),
            (
                "Check whether the response encourages the "
                "candidate to explain their experience without "
                "being dismissive or judgmental."
            ),
            (
                "Give a final score based on professionalism, "
                "respect, neutrality, clarity, and appropriate "
                "interview behavior."
            ),
        ],

        model=model,
        threshold=PROFESSIONALISM_THRESHOLD,
    )


# ============================================================
# Test
# ============================================================

def test_professionalism():
    """
    Verify that the AI interviewer communicates
    professionally.
    """

    print()
    print("=" * 70)
    print("AI INTERVIEWER PROFESSIONALISM TEST")
    print("=" * 70)

    print()
    print(
        f"Evaluation model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA base URL: {NVIDIA_BASE_URL}"
    )

    print(
        f"Quality threshold: {PROFESSIONALISM_THRESHOLD}"
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
    print("Creating interview test case...")

    test_case = (
        create_professionalism_test_case()
    )

    print(
        "[PASS] Interview test case created."
    )

    # --------------------------------------------------------
    # 3. Create metric
    # --------------------------------------------------------

    print()
    print("Creating professionalism metric...")

    metric = create_professionalism_metric(
        model
    )

    print(
        "[PASS] Professionalism metric configured."
    )

    # --------------------------------------------------------
    # 4. Run evaluation
    # --------------------------------------------------------

    print()
    print(
        "Evaluating interviewer professionalism..."
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
    print("INTERVIEWER PROFESSIONALISM TEST PASSED")
    print("=" * 70)