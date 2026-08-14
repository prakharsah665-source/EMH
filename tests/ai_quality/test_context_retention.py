from deepeval import assert_test
from config.settings import NVIDIA_BASE_URL, NVIDIA_MODEL
from evaluators.nvidia_judge import get_nvidia_judge
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric


# ============================================================
# NVIDIA NEMOTRON CONFIGURATION
# ============================================================



# Create the NVIDIA Nemotron judge explicitly.
#
# This avoids relying on DeepEval's global provider selection,
# which was resolving to LocalModel(model=None) in your setup.
#
nvidia_model = get_nvidia_judge()


# ============================================================
# CONTEXT RETENTION TEST
# ============================================================


def test_context_retention():

    print()
    print("=" * 70)
    print("AI CONTEXT RETENTION TEST")
    print("=" * 70)

    print()
    print("Evaluation model:", NVIDIA_MODEL)
    print("NVIDIA base URL:", NVIDIA_BASE_URL)

    # ========================================================
    # Conversation
    # ========================================================

    test_case = ConversationalTestCase(

        turns=[

            # ------------------------------------------------
            # TURN 1 - Candidate provides important information
            # ------------------------------------------------

            Turn(
                role="user",
                content=(
                    "I have around two years of experience with React "
                    "and I recently worked on a student management "
                    "application."
                ),
            ),

            # ------------------------------------------------
            # TURN 2 - AI acknowledges information
            # ------------------------------------------------

            Turn(
                role="assistant",
                content=(
                    "That's good experience. Your work with React "
                    "and the student management application gives "
                    "you practical frontend experience."
                ),
            ),

            # ------------------------------------------------
            # TURN 3 - Candidate asks a follow-up
            # ------------------------------------------------

            Turn(
                role="user",
                content=(
                    "What kind of project experience would you "
                    "highlight during a frontend interview?"
                ),
            ),

            # ------------------------------------------------
            # TURN 4 - AI should retain previous information
            # ------------------------------------------------

            Turn(
                role="assistant",
                content=(
                    "I would highlight your React experience and "
                    "your student management application. You can "
                    "explain the components you built, state "
                    "management, API integration, and the challenges "
                    "you solved."
                ),
            ),
        ]
    )

    # ========================================================
    # Knowledge Retention Metric
    # ========================================================

    print()
    print("Creating Knowledge Retention metric...")

    metric = KnowledgeRetentionMetric(
        threshold=0.70,
        model=nvidia_model,
        include_reason=True,
        verbose_mode=True,
    )

    print("[PASS] NVIDIA Nemotron model configured.")
    print("[PASS] Knowledge Retention metric configured.")

    # ========================================================
    # Run DeepEval
    # ========================================================

    print()
    print("Running AI context retention evaluation...")
    print("Model:", NVIDIA_MODEL)

    assert_test(
        test_case,
        [metric],
    )

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 70)
    print("CONTEXT RETENTION TEST PASSED")
    print("=" * 70)