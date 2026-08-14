"""
CI test for EMH AI interviewer quality.

The test:
1. Evaluates the interview three times.
2. Uses the median score for each criterion.
3. Calculates the final overall score.
4. Fails CI if the required quality thresholds are not met.
"""

import os

import pytest

from evaluation.consistency import evaluate_with_consistency
from evaluation.models import NvidiaInferenceError
from evaluation.transcript import (
    get_transcript_path,
    load_real_transcript,
)
from evaluation.transcript_validation import (
    TranscriptValidationError,
)


# ============================================================
# CI thresholds
# ============================================================

# Overall interviewer quality must be at least 0.75.
OVERALL_THRESHOLD = float(
    os.getenv(
        "INTERVIEW_QUALITY_THRESHOLD",
        "0.75",
    )
)


# Individual criteria that are important enough
# to have their own CI requirements.
CRITERION_THRESHOLDS = {
    "technical_correctness": 0.75,
    "relevance": 0.75,
    "follow_up_quality": 0.75,
    "question_quality": 0.75,
    "difficulty_calibration": 0.50,
    "communication_quality": 0.75,
    "context_awareness": 0.75,
}


# ============================================================
# Main CI test
# ============================================================

def get_interview_transcript() -> str:
    """
    Load the REAL captured EMH transcript
    (artifacts/transcripts/actual_transcript.json, written by
    the bot-responsiveness E2E run). A real transcript is
    MANDATORY: evaluating a canned benchmark transcript can
    only ever grade the judge, not the product, so a missing
    capture fails the gate loudly instead of silently passing
    on fiction.
    """

    path = get_transcript_path()

    if not path.exists():
        pytest.fail(
            "\n"
            "REAL TRANSCRIPT MISSING\n"
            f"No captured EMH transcript at {path}.\n"
            "Run the E2E capture first:\n"
            "    pytest tests/e2e/test_bot_responsiveness.py -s\n"
            "(or set EMH_TRANSCRIPT_PATH to a captured "
            "transcript). The quality gate refuses to score a "
            "synthetic benchmark transcript."
        )

    print(f"\nUsing real EMH transcript: {path}")

    # TranscriptValidationError subclasses ValueError, so it
    # must be classified HERE: letting it reach the evaluator's
    # ValueError handler would mislabel a capture problem as an
    # evaluator-output problem.
    try:
        return load_real_transcript(path)
    except TranscriptValidationError as error:
        pytest.fail(
            "\n"
            "TRANSCRIPT NOT SCORABLE\n"
            "The captured transcript is not a complete, current "
            "interview, so the quality gate refuses to rubric-"
            "score it as one.\n"
            f"{error}\n"
            "Root cause lives upstream: check the bot-"
            "responsiveness E2E result/artifacts (it classifies "
            "whether the interview died from a bot failure, a "
            "LiveKit/audio issue, or a capture problem), then "
            "re-run the capture:\n"
            "    pytest tests/e2e/test_bot_responsiveness.py -s\n"
        )


@pytest.mark.ai_evaluation
def test_interview_overall_quality():

    # ----------------------------------------------------
    # Keep API/inference failures separate from genuine
    # quality failures: an unreachable or misbehaving judge
    # must never read as "the interviewer failed CI".
    # ----------------------------------------------------

    try:

        result = evaluate_with_consistency(
            get_interview_transcript()
        )

    except NvidiaInferenceError as error:

        pytest.fail(
            "\n"
            "NVIDIA API/INFERENCE FAILURE\n"
            "This is an infrastructure failure, NOT an "
            "interview quality failure.\n"
            f"{error}\n"
        )

    except ValueError as error:

        pytest.fail(
            "\n"
            "EVALUATOR OUTPUT FAILURE\n"
            "The judge returned an invalid or incomplete "
            "evaluation. This is NOT an interview quality "
            "failure.\n"
            f"{error}\n"
        )

    final_result = result["final"]

    overall_score = final_result[
        "overall_score"
    ]

    criteria = final_result[
        "criteria"
    ]

    # --------------------------------------------------------
    # Print evaluation report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EMH AI INTERVIEW QUALITY")
    print("=" * 70)

    print()

    print(
        f"Overall median score: "
        f"{overall_score:.4f}"
    )

    print(
        f"Required overall score: "
        f"{OVERALL_THRESHOLD:.2f}"
    )

    print()

    print("Criterion scores:")

    for criterion, data in criteria.items():

        score = data["score"]

        print(
            f"  {criterion:<30} "
            f"{score:.2f} "
            f"runs={data['run_scores']}"
        )

        reasoning = (data.get("reasoning") or "").strip()
        if reasoning:
            print(f"      judge: {reasoning}")

    print()
    print("=" * 70)

    # --------------------------------------------------------
    # Overall quality gate
    # --------------------------------------------------------

    weakest = sorted(
        criteria.items(),
        key=lambda item: item[1]["score"],
    )[:3]

    weakest_lines = "".join(
        f"  {name}: {data['score']:.2f} "
        f"(runs={data['run_scores']})\n"
        f"    judge: {(data.get('reasoning') or '').strip()}\n"
        for name, data in weakest
    )

    assert overall_score >= OVERALL_THRESHOLD, (
        "\n"
        "AI INTERVIEW QUALITY FAILED\n"
        f"Overall median score: {overall_score:.4f}\n"
        f"Required score: {OVERALL_THRESHOLD:.2f}\n"
        "Weakest criteria (judge reasoning included so the "
        "root cause is identifiable):\n"
        f"{weakest_lines}"
    )

    # --------------------------------------------------------
    # Individual criterion gates
    # --------------------------------------------------------

    failures = []

    for criterion, required_score in (
        CRITERION_THRESHOLDS.items()
    ):

        actual_score = criteria[
            criterion
        ]["score"]

        if actual_score < required_score:

            failures.append(
                {
                    "criterion": criterion,
                    "actual": actual_score,
                    "required": required_score,
                }
            )

    if failures:

        failure_message = [
            "",
            "AI INTERVIEW CRITERION GATES FAILED",
            "",
        ]

        for failure in failures:

            data = criteria[failure["criterion"]]

            failure_message.append(
                f"{failure['criterion']}: "
                f"{failure['actual']:.2f} "
                f"< required "
                f"{failure['required']:.2f} "
                f"(runs={data['run_scores']}, "
                f"spread={data['spread']:.2f})"
            )

            reasoning = (data.get("reasoning") or "").strip()
            if reasoning:
                failure_message.append(
                    f"    judge: {reasoning}"
                )

        failure_message.append("")

        pytest.fail(
            "\n".join(failure_message)
        )

    print()
    print("AI INTERVIEW QUALITY: PASSED")
    print("=" * 70)
