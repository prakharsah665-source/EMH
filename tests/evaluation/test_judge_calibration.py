"""
Judge calibration: prove the Nemotron rubric judge can tell a
good interview from a bad one BEFORE trusting it to gate CI.

Golden set (data/calibration/):
  known_good_transcript.txt - competent, context-aware,
      technically correct interview. Must clear the same
      thresholds the CI gate uses.
  known_bad_transcript.txt  - irrelevant questions, hostile
      tone, technically false claims, lost context, repeated
      questions. Must score clearly BELOW the gate.

A judge that passes the good case but also passes the bad case
is worthless as a gate - that is exactly the false-pass mode
this test exists to catch. Both cases use the same 3-run-median
consistency path as the real gate, so run disagreement is
reported here too.

Requires NVIDIA credits (marked ai_evaluation, like the gate).
"""

from pathlib import Path

import pytest

from evaluation.consistency import evaluate_with_consistency
from evaluation.models import NvidiaInferenceError


CALIBRATION_DIR = Path("data/calibration")

# The known-good case must clear the CI gate's own bar; the
# known-bad case must land clearly below it. The 0.50 ceiling
# is intentionally far below the 0.75 gate so a merely-lenient
# judge fails calibration before it can false-pass real CI.
GOOD_MINIMUM_OVERALL = 0.75
BAD_MAXIMUM_OVERALL = 0.50


def evaluate_calibration_case(filename: str) -> dict:
    transcript = (CALIBRATION_DIR / filename).read_text(
        encoding="utf-8"
    )

    try:
        result = evaluate_with_consistency(transcript)
    except NvidiaInferenceError as error:
        pytest.fail(
            "\nNVIDIA API/INFERENCE FAILURE during judge "
            "calibration - infrastructure problem, not a "
            f"calibration verdict.\n{error}\n"
        )
    except ValueError as error:
        # Same classification the CI gate uses: malformed or
        # incomplete judge output is an evaluator problem and
        # must never read as a calibration verdict.
        pytest.fail(
            "\nEVALUATOR OUTPUT FAILURE during judge "
            "calibration - the judge returned an invalid or "
            "incomplete evaluation. Infrastructure/evaluator "
            f"problem, not a calibration verdict.\n{error}\n"
        )

    final = result["final"]

    print(f"\nCalibration case: {filename}")
    print(f"  overall median: {final['overall_score']:.4f}")
    for criterion, data in final["criteria"].items():
        print(
            f"  {criterion:<30} {data['score']:.2f} "
            f"runs={data['run_scores']} "
            f"spread={data['spread']:.2f}"
            + ("  << DISAGREEMENT" if data["disagreement"] else "")
        )
    if final["disagreement_criteria"]:
        print(
            "  RUN DISAGREEMENT on: "
            + ", ".join(final["disagreement_criteria"])
        )

    return final


@pytest.mark.ai_evaluation
def test_judge_passes_known_good_interview():
    final = evaluate_calibration_case("known_good_transcript.txt")

    assert final["overall_score"] >= GOOD_MINIMUM_OVERALL, (
        "\nJUDGE CALIBRATION FAILED (known-good case)\n"
        f"The judge scored a competent reference interview "
        f"{final['overall_score']:.4f}, below "
        f"{GOOD_MINIMUM_OVERALL:.2f}. A judge this harsh would "
        "false-fail real interviews - fix the judge/rubric "
        "before trusting CI results.\n"
    )


@pytest.mark.ai_evaluation
def test_judge_fails_known_bad_interview():
    final = evaluate_calibration_case("known_bad_transcript.txt")

    assert final["overall_score"] <= BAD_MAXIMUM_OVERALL, (
        "\nJUDGE CALIBRATION FAILED (known-bad case)\n"
        f"The judge scored an egregiously bad interview "
        f"{final['overall_score']:.4f}, above "
        f"{BAD_MAXIMUM_OVERALL:.2f}. A judge this lenient "
        "would false-pass broken interviews - the CI quality "
        "gate cannot be trusted until this is fixed.\n"
    )

    # The bad interview is bad in specific, designed ways; the
    # judge must notice at least the designed defects.
    for criterion in (
        "relevance",
        "technical_correctness",
        "context_awareness",
    ):
        score = final["criteria"][criterion]["score"]
        assert score <= 0.50, (
            f"\nJUDGE CALIBRATION FAILED: '{criterion}' scored "
            f"{score:.2f} on a transcript specifically "
            "constructed to violate it (irrelevant questions, "
            "false technical claims, forgotten context).\n"
        )
