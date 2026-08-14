"""
Consistency evaluation for the EMH AI interviewer.

Runs the same interview evaluation exactly three times and
uses the median score for every criterion.

This protects CI from a single inconsistent LLM judgment.
"""

from __future__ import annotations

import json
from statistics import median
from typing import Any

from evaluation.evaluator import evaluate_interview
from evaluation.rubric import ALLOWED_SCORES, RUBRIC


# ============================================================
# Configuration
# ============================================================

NUMBER_OF_RUNS = 3


# ============================================================
# Validation helpers
# ============================================================

def validate_run_result(
    result: dict[str, Any],
) -> None:
    """
    Validate that an individual evaluation contains every
    required criterion and a valid discrete score.
    """

    if "criteria" not in result:
        raise ValueError(
            "Evaluation result is missing 'criteria'."
        )

    criteria = result["criteria"]

    for criterion in RUBRIC:

        # Overall score is calculated separately.
        if criterion == "overall_interview_quality":
            continue

        if criterion not in criteria:
            raise ValueError(
                f"Missing criterion: {criterion}"
            )

        score = criteria[criterion].get("score")

        if score not in ALLOWED_SCORES:
            raise ValueError(
                f"Invalid score for {criterion}: {score}. "
                f"Allowed: {sorted(ALLOWED_SCORES)}"
            )


# ============================================================
# Median calculation
# ============================================================

def calculate_median_scores(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate the median score for every evaluation criterion.

    The LLM does NOT calculate the median.
    Python does it deterministically.
    """

    if len(results) != NUMBER_OF_RUNS:
        raise ValueError(
            f"Expected exactly {NUMBER_OF_RUNS} evaluation runs, "
            f"received {len(results)}."
        )

    for result in results:
        validate_run_result(result)

    final_criteria: dict[str, dict[str, Any]] = {}

    for criterion in RUBRIC:

        if criterion == "overall_interview_quality":
            continue

        scores = [
            result["criteria"][criterion]["score"]
            for result in results
        ]

        median_score = float(
            median(scores)
        )

        # Run-disagreement reporting: the median alone hides
        # judge instability (0.00/0.75/0.75 and 0.75/0.75/0.75
        # both yield 0.75). Spread is max-min across the runs;
        # a spread of two or more rubric buckets (>= 0.50) is
        # flagged so unstable judgments are visible in reports
        # and can be investigated instead of silently trusted.
        spread = float(max(scores) - min(scores))

        # Preserve the judge's own reasoning: without it a
        # failing gate can only report a number, never WHY the
        # criterion failed. "reasoning" is the reasoning of the
        # run whose score equals the median (the judgment the
        # gate actually asserts on); run_reasonings keeps all
        # three for disagreement analysis.
        run_reasonings = [
            result["criteria"][criterion].get("reasoning", "")
            for result in results
        ]

        median_reasoning = next(
            (
                reasoning
                for score, reasoning in zip(scores, run_reasonings)
                if float(score) == median_score
            ),
            "",
        )

        final_criteria[criterion] = {
            "score": median_score,
            "run_scores": scores,
            "spread": spread,
            "disagreement": spread >= 0.50,
            "reasoning": median_reasoning,
            "run_reasonings": run_reasonings,
        }

    # --------------------------------------------------------
    # Calculate final overall score from the median criterion
    # scores.
    # --------------------------------------------------------

    criterion_scores = [
        criterion["score"]
        for criterion in final_criteria.values()
    ]

    overall_score = (
        sum(criterion_scores)
        / len(criterion_scores)
    )

    disagreements = [
        criterion
        for criterion, data in final_criteria.items()
        if data["disagreement"]
    ]

    return {
        "number_of_runs": NUMBER_OF_RUNS,
        "criteria": final_criteria,
        "overall_score": overall_score,
        "disagreement_criteria": disagreements,
        "max_criterion_spread": max(
            data["spread"] for data in final_criteria.values()
        ),
    }


# ============================================================
# Three-run evaluation
# ============================================================

def evaluate_with_consistency(
    interview_transcript: str,
) -> dict[str, Any]:
    """
    Evaluate an interview exactly three times.

    Each evaluation uses the existing evaluator, which calls
    NVIDIA Nemotron with temperature=0.

    The median is then calculated independently for every
    criterion.
    """

    if not interview_transcript.strip():
        raise ValueError(
            "Interview transcript cannot be empty."
        )

    results: list[dict[str, Any]] = []

    for run_number in range(1, NUMBER_OF_RUNS + 1):

        print(
            f"\nRunning evaluation "
            f"{run_number}/{NUMBER_OF_RUNS}..."
        )

        result = evaluate_interview(
            interview_transcript
        )

        validate_run_result(result)

        results.append(result)

        print(
            f"Run {run_number} overall score: "
            f"{result['overall_score']:.4f}"
        )

    final_result = calculate_median_scores(
        results
    )

    return {
        "runs": results,
        "final": final_result,
    }


# ============================================================
# Human-readable summary
# ============================================================

def print_consistency_report(
    result: dict[str, Any],
) -> None:
    """
    Print a readable summary of the three-run evaluation.
    """

    runs = result["runs"]
    final = result["final"]

    print()
    print("=" * 70)
    print("INTERVIEW QUALITY CONSISTENCY REPORT")
    print("=" * 70)

    print()

    print("Individual run scores:")

    for index, run in enumerate(runs, start=1):

        print(
            f"  Run {index}: "
            f"{run['overall_score']:.4f}"
        )

    print()

    print("Criterion median scores:")

    for criterion, data in final["criteria"].items():

        print(
            f"  {criterion:<30} "
            f"{data['score']:.2f} "
            f"runs={data['run_scores']} "
            f"spread={data['spread']:.2f}"
            + ("  << DISAGREEMENT" if data["disagreement"] else "")
        )

    print()

    disagreements = final.get("disagreement_criteria", [])
    if disagreements:
        print(
            "RUN DISAGREEMENT (spread >= 0.50) on: "
            + ", ".join(disagreements)
        )
        print(
            "The median was used, but these judgments are "
            "unstable across identical runs - inspect the "
            "per-run reasoning before trusting them."
        )
        print()

    print(
        f"FINAL MEDIAN OVERALL SCORE: "
        f"{final['overall_score']:.4f}"
    )

    print("=" * 70)


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":

    sample_transcript = """
AI Interviewer:
Can you explain what React is?

Candidate:
React is a JavaScript library used to build user interfaces.

AI Interviewer:
Good. Can you explain why components are useful in React?

Candidate:
Components let us break the UI into reusable pieces.

AI Interviewer:
Can you give an example of when you would use state?

Candidate:
I would use state when data needs to change over time,
such as a counter or form input.
"""

    result = evaluate_with_consistency(
        sample_transcript
    )

    print_consistency_report(
        result
    )

    print()
    print("Raw JSON result:")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )
