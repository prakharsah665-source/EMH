"""
Per-turn judgment of the REAL AI interviewer.

For every interviewer turn captured from the live interview the
judge (NVIDIA Nemotron, temperature 0) reads the preceding
conversation, decides independently whether this was the right
question/response at this point, scores relevance, clarity,
question quality, context awareness / logical follow-up,
repetition, professionalism and flow alignment, and lists any
incorrect / irrelevant / repetitive / nonsensical turn.

Candidate text is shown ONLY as captured STT context
(evaluation.interviewer_turn_evaluation rejects the harness's
injected-audio reference text before anything reaches the
judge). A candidate reply that was not captured is shown as
"(candidate response not captured)" and the judge is told not to
guess it.

The full report (exact prompt, exact transcript data, per-turn
results, Python aggregation) is written to
artifacts/reports/interviewer_turn_evaluation.json.
"""

import json

import pytest

from evaluation.interviewer_turn_evaluation import (
    InterviewerEvidenceError,
    TurnValidationError,
    build_interviewer_turns,
    dump_report,
    evaluate_interviewer_turns,
    filter_turns,
    select_turn_capture,
)
from evaluation.transcript_validation import (
    capture_coverage_problem,
    classify_status_for_scoring,
    load_status,
)


REPORT_PATH = "artifacts/reports/interviewer_turn_evaluation.json"
QUALITY_THRESHOLD = 0.70
# Issues that make a single interviewer turn a hard failure on
# their own, regardless of the mean score.
HARD_ISSUES = ("nonsensical", "unprofessional", "incorrect")


@pytest.mark.ai_evaluation
def test_real_interviewer_turns_are_appropriate_in_context():
    # ---- capture freshness/completeness gate (shared policy) ----
    status = load_status()
    verdict, reason = classify_status_for_scoring(status)
    if verdict == "skip-upstream":
        pytest.skip(f"interviewer_turns: {reason}")
    if verdict != "ok":
        pytest.fail(f"interviewer_turns: {reason}")

    try:
        backend, raw_turns = select_turn_capture()
    except InterviewerEvidenceError as error:
        pytest.skip(f"interviewer_turns: {error}")

    kept, provenance = filter_turns(raw_turns)
    units = build_interviewer_turns(kept)
    print(
        f"\n[interviewer_turns] backend={backend.name} "
        f"interviewer turns={len(units)} candidate context "
        f"accepted={provenance.accepted_sources} "
        f"rejected={provenance.rejected_sources}"
    )

    # Coverage: a complete drive of N candidate turns should yield
    # about N+1 interviewer utterances; a fragment is not judged
    # as the whole interview.
    coverage_problem = capture_coverage_problem(
        max(len(units) - 1, 0), status
    )
    if coverage_problem:
        pytest.skip(f"interviewer_turns: {coverage_problem}")

    # ---- judge ----
    try:
        report = evaluate_interviewer_turns(units)
    except TurnValidationError as error:
        pytest.fail(f"JUDGE OUTPUT INVALID (evaluator issue): {error}")
    report["backend"] = backend.name
    report["provenance"] = provenance.as_dict()
    dump_report(report, REPORT_PATH)

    # ---- show the evidence in the test log ----
    print("\n===== EXACT PROMPT SENT TO THE JUDGE =====")
    print(report["prompt"])
    print("===== PER-TURN RESULTS =====")
    for result in report["results"]:
        print(
            f"\nTurn {result['number']} | score={result['score']:.2f} "
            f"| issues={result['issues'] or 'none'}\n"
            f"  Interviewer: {result['interviewer_turn']}\n"
            f"  Evaluation: "
            + ", ".join(
                f"{k}={v['score']:.2f}"
                for k, v in result["evaluation"].items()
            )
            + f"\n  Reasoning: {result['reasoning']}"
            + (
                f"\n  Issue details: {result['issue_details']}"
                if result["issue_details"]
                else ""
            )
        )
    print("\n===== AGGREGATE (Python, from model scores) =====")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {REPORT_PATH}")

    # ---- assertions ----
    aggregate = report["aggregate"]
    hard = [
        r for r in report["results"]
        if any(issue in HARD_ISSUES for issue in r["issues"])
    ]
    assert not hard, (
        "Interviewer turns with hard issues: "
        + "; ".join(
            f"T{r['number']} {r['issues']}: {r['interviewer_turn'][:80]}"
            for r in hard
        )
    )
    assert aggregate["overall_score"] >= QUALITY_THRESHOLD, (
        f"Mean interviewer-turn score {aggregate['overall_score']:.2f} "
        f"< {QUALITY_THRESHOLD}; flagged turns: "
        f"{aggregate['flagged_turns']}"
    )
