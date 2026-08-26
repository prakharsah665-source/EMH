"""
Shared harness for judging REAL captured interview transcripts
via the TranscriptCapture layer (collectors/transcript_capture).

Every ai_quality test declares its provenance requirements:

    min_confidence   - "low" | "medium" | "high": turns below
                       this confidence are excluded before
                       judging (e.g. "injected-audio" candidate
                       text is low - it is what the harness
                       SAID, not what the app heard).
    exclude_sources  - source prefixes that are unacceptable
                       for this metric regardless of confidence
                       (e.g. repetition metrics must reject
                       stt-derived text: STT normalisation can
                       fabricate or hide repetition).

When the surviving turns cannot support the metric, the test
SKIPS with a capture-layer reason - synthetic data is never
scored silently, and a missing capture is never reported as a
bot failure.
"""

import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from collectors.transcript_capture import select_capture
from evaluation.transcript_validation import (
    classify_status_for_scoring,
)
from evaluators.consistent_geval import assert_metric_median
from evaluators.nvidia_judge import get_nvidia_judge


QUALITY_THRESHOLD = 0.70

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

INTERVIEW_CONTEXT = (
    "A real technical interview conducted by the EMH AI "
    "interviewer (Jamie). The interviewer is expected to run a "
    "professional, relevant, technically accurate, context-aware "
    "and adaptive multi-turn interview, starting from the "
    "candidate's introduction."
)


def real_capture_turns(
    *,
    test_label: str,
    min_confidence: str = "medium",
    exclude_sources: tuple[str, ...] = (),
    min_assistant_turns: int = 2,
    min_user_turns: int = 1,
) -> list[dict]:
    """
    Return the real captured turns that satisfy this test's
    provenance requirements, or skip with a capture-layer
    reason.
    """

    floor = _CONFIDENCE_ORDER[min_confidence]

    # Freshness/completeness gate BEFORE provenance filtering:
    # scoring is allowed only for a fresh, complete capture.
    # An incomplete capture SKIPS (its bot/capture failure is
    # already reported upstream by the capture test); missing
    # or stale status FAILS at the capture/session layer.
    # load_status() reads the sidecar the capture run just
    # wrote, so the classifier always sees the REAL current
    # status object (printed for the report).
    from evaluation.transcript_validation import load_status

    status = load_status()
    print(
        f"[{test_label}] capture status: "
        f"complete={status.complete}, "
        f"turns={status.turn_count}, "
        f"age={status.age_seconds and round(status.age_seconds)}s"
    )

    verdict, reason = classify_status_for_scoring(status)
    if verdict == "skip-upstream":
        pytest.skip(f"{test_label}: {reason}")
    if verdict != "ok":
        pytest.fail(f"{test_label}: {reason}")

    try:
        capture = select_capture()
    except RuntimeError as error:
        pytest.skip(f"{test_label}: {error}")

    all_turns = capture.get_turns()

    turns = [
        turn
        for turn in all_turns
        if _CONFIDENCE_ORDER.get(turn["confidence"], 0) >= floor
        and not turn["source"].startswith(exclude_sources)
    ]

    assistant = sum(1 for t in turns if t["role"] == "assistant")
    user = sum(1 for t in turns if t["role"] == "user")

    # Cross-check capture vs drive: a COMPLETE interview whose
    # capture channel caught only a fragment must not be judged
    # as the whole interview - missing turns mean "not captured",
    # never "never asked". Capture limitation -> skip.
    from evaluation.transcript_validation import (
        capture_coverage_problem,
    )

    coverage_problem = capture_coverage_problem(user, status)
    if coverage_problem:
        pytest.skip(f"{test_label}: {coverage_problem}")

    if assistant < min_assistant_turns or user < min_user_turns:
        pytest.skip(
            f"{test_label}: capture backend "
            f"'{capture.name}' produced {len(all_turns)} turns "
            f"but only {assistant} interviewer / {user} candidate "
            f"turn(s) meet this metric's provenance requirements "
            f"(min_confidence={min_confidence}, "
            f"exclude_sources={list(exclude_sources)}; need >= "
            f"{min_assistant_turns}/{min_user_turns}). "
            "CAPTURE LIMITATION - the real bot text was not "
            "captured with sufficient provenance; this is not a "
            "bot failure and synthetic data is never scored "
            "instead. See docs/bot_text_capture.md."
        )

    return turns


def format_capture_turns(turns: list[dict]) -> str:
    """Evaluator-ready transcript text with provenance tags."""

    labels = {"assistant": "AI Interviewer", "user": "Candidate"}
    blocks = [
        f"{labels[turn['role']]} "
        f"[source={turn['source']}]:\n{turn['text']}"
        for turn in turns
    ]
    return "\n\n".join(blocks)


def judge_real_dimension(
    *,
    name: str,
    criteria: str,
    min_confidence: str = "medium",
    exclude_sources: tuple[str, ...] = (),
    threshold: float = QUALITY_THRESHOLD,
) -> dict:
    """
    Judge one quality dimension over the real captured
    transcript with the shared Nemotron judge: 3-run median,
    disagreement asserted (see consistent_geval).
    """

    turns = real_capture_turns(
        test_label=name,
        min_confidence=min_confidence,
        exclude_sources=exclude_sources,
    )

    transcript = format_capture_turns(turns)
    print(
        f"[{name}] judging {len(turns)} captured turns "
        f"({sum(1 for t in turns if t['role'] == 'assistant')} "
        "interviewer)."
    )

    test_case = LLMTestCase(
        input=INTERVIEW_CONTEXT,
        actual_output=transcript,
    )

    def factory() -> GEval:
        return GEval(
            name=name,
            criteria=criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=get_nvidia_judge(),
            threshold=threshold,
        )

    return assert_metric_median(factory, test_case)
