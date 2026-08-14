"""
Three-run-median execution for DeepEval GEval metrics.

Mirrors evaluation/consistency.py (which covers the rubric
evaluator) for the GEval-based AI-quality tests: every metric
is measured three times against the same test case, Python
computes the median deterministically, and run disagreement
(spread >= 0.50) is reported instead of silently discarded.

The judge, thresholds and metric definitions are unchanged -
this module only controls HOW MANY times a metric runs and
which score is asserted.
"""

from statistics import median
from typing import Any, Callable

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase


NUMBER_OF_RUNS = 3

# Two or more rubric-scale buckets apart across identical runs.
DISAGREEMENT_SPREAD = 0.50


def measure_with_consistency(
    metric_factory: Callable[[], GEval],
    test_case: LLMTestCase,
    runs: int = NUMBER_OF_RUNS,
) -> dict[str, Any]:
    """
    Measure a GEval metric `runs` times (fresh metric instance
    per run so no state leaks) and return the median score with
    the full per-run detail.
    """

    scores: list[float] = []
    reasons: list[str] = []
    name = None
    threshold = None

    for run_number in range(1, runs + 1):
        metric = metric_factory()
        name = metric.name
        threshold = metric.threshold

        print(f"\n[{name}] run {run_number}/{runs}...")
        metric.measure(test_case)

        if metric.score is None:
            raise ValueError(
                f"GEval metric '{name}' returned no score on "
                f"run {run_number}."
            )

        scores.append(float(metric.score))
        reasons.append(metric.reason or "")
        print(f"[{name}] run {run_number} score: {metric.score:.4f}")

    spread = max(scores) - min(scores)

    return {
        "metric": name,
        "threshold": threshold,
        "median_score": float(median(scores)),
        "run_scores": scores,
        "spread": spread,
        "disagreement": spread >= DISAGREEMENT_SPREAD,
        "run_reasons": reasons,
    }


def assert_metric_median(
    metric_factory: Callable[[], GEval],
    test_case: LLMTestCase,
    runs: int = NUMBER_OF_RUNS,
) -> dict[str, Any]:
    """
    Three-run-median replacement for deepeval.assert_test:
    asserts the MEDIAN score meets the metric's own threshold
    and prints the per-run scores and any disagreement.
    """

    result = measure_with_consistency(
        metric_factory, test_case, runs=runs
    )

    print()
    print(
        f"[{result['metric']}] median {result['median_score']:.4f} "
        f"(runs: {['%.2f' % s for s in result['run_scores']]}, "
        f"spread {result['spread']:.2f}, "
        f"threshold {result['threshold']:.2f})"
    )
    if result["disagreement"]:
        print(
            f"[{result['metric']}] RUN DISAGREEMENT: identical "
            "runs differ by >= "
            f"{DISAGREEMENT_SPREAD:.2f}. The median is used, "
            "but inspect the per-run reasoning:"
        )
        for index, reason in enumerate(result["run_reasons"], 1):
            print(f"  run {index}: {reason[:300]}")

    assert result["median_score"] >= result["threshold"], (
        f"\n{result['metric']} FAILED\n"
        f"Median score: {result['median_score']:.4f} "
        f"(runs: {result['run_scores']})\n"
        f"Required: {result['threshold']:.2f}\n"
        f"Reasons: {result['run_reasons']}\n"
    )

    # A median above threshold is not trustworthy when the
    # three identical runs disagree by two or more rubric
    # buckets - the judgment itself is unstable. This is an
    # EVALUATOR/JUDGE failure classification, not a bot
    # failure: the score cannot be relied on in either
    # direction.
    assert not result["disagreement"], (
        f"\n{result['metric']} JUDGE UNSTABLE "
        "(evaluator failure, NOT a bot failure)\n"
        f"Identical runs scored {result['run_scores']} "
        f"(spread {result['spread']:.2f} >= "
        f"{DISAGREEMENT_SPREAD:.2f}), so neither pass nor "
        "fail is trustworthy. Inspect the per-run reasoning "
        "and the judge configuration before rerunning:\n"
        + "\n".join(
            f"  run {index}: {reason[:300]}"
            for index, reason in enumerate(
                result["run_reasons"], 1
            )
        )
    )

    return result
