"""
AI Quality Test: Interview Question Repetition

Purpose:
    Verify that the AI interviewer does not repeatedly ask
    the same question or ask semantically redundant questions.

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


QUESTION_REPETITION_THRESHOLD = 0.70


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

def create_question_repetition_test_case():
    """
    Create a conversation containing several interviewer
    questions.

    The final question must introduce a genuinely new
    dimension. The metric fails if it semantically repeats
    an earlier question.
    """

    conversation = """
    Interview role:
    Frontend Developer

    Previous interview questions:

    Question 1:
    How did you use React in your student management
    application?

    Question 2:
    What challenges did you face while building the
    frontend of your student management application?

    Question 3:
    How did you manage state in your React application?

    The interviewer must now ask the next question without
    repeating the information already requested above.
    """

    final_question = (
        "How did you test your React components in the "
        "student management application, and how did you "
        "decide what was worth testing?"
    )

    return LLMTestCase(
        input=conversation,
        actual_output=final_question,
    )


# ============================================================
# Evaluation metric
# ============================================================

def create_question_repetition_metric(model):
    """
    Create a GEval metric that evaluates whether the latest
    interviewer question unnecessarily repeats an earlier
    question.
    """

    return GEval(
        name="Interview Question Repetition",

        criteria=(
            "Evaluate whether the interviewer's latest "
            "question unnecessarily repeats a previous "
            "interview question. Repetition should be judged "
            "semantically rather than by exact wording. "
            "Questions that use different words but ask for "
            "substantially the same information should be "
            "considered repetitive. Questions that explore "
            "a genuinely different aspect of the candidate's "
            "knowledge should not be considered repetitive."
        ),

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        evaluation_steps=[
            (
                "Identify the topics and intent of the previous "
                "interview questions."
            ),
            (
                "Identify the topic and intent of the latest "
                "interviewer question."
            ),
            (
                "Compare the latest question with each previous "
                "question based on meaning and information "
                "being requested, not just wording."
            ),
            (
                "Determine whether the latest question asks "
                "for substantially the same information as an "
                "earlier question."
            ),
            (
                "Do not mark a question as repetitive merely "
                "because it belongs to the same broad technical "
                "topic. It should be substantially redundant."
            ),
            (
                "Penalize unnecessary repetition and reward "
                "questions that introduce a genuinely new "
                "technical or behavioral dimension."
            ),
            (
                "Give a final score based on semantic "
                "non-repetition, topic progression, and "
                "interview diversity."
            ),
        ],

        model=model,
        threshold=QUESTION_REPETITION_THRESHOLD,
    )


# ============================================================
# Test
# ============================================================

def test_question_repetition():
    """
    Verify that the AI interviewer avoids asking a
    semantically repetitive question.
    """

    print()
    print("=" * 70)
    print("AI QUESTION REPETITION TEST")
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
        f"{QUESTION_REPETITION_THRESHOLD}"
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
        create_question_repetition_test_case()
    )

    print(
        "[PASS] Interview test case created."
    )

    # --------------------------------------------------------
    # 3. Create metric
    # --------------------------------------------------------

    print()
    print("Creating question repetition metric...")

    metric = create_question_repetition_metric(
        model
    )

    print(
        "[PASS] Question repetition metric configured."
    )

    # --------------------------------------------------------
    # 4. Run evaluation
    # --------------------------------------------------------

    print()
    print(
        "Evaluating question repetition..."
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
    print("QUESTION REPETITION TEST PASSED")
    print("=" * 70)
    