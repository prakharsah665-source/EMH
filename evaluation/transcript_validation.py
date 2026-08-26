"""
Whole-interview transcript validation.

The AI evaluation must score ONE complete, current interview
transcript - never an empty, stale, or truncated one. This
module centralises those rejections so every evaluation entry
point (rubric gate, GEval suite) fails loudly with a specific
reason instead of silently grading partial or old data.

Completion/freshness come from the status sidecar written by the
capture run (collectors.transcript_collector.save_transcript_
status). Absent status is treated as UNKNOWN and, by default,
rejected under require_complete so a transcript of unverified
provenance cannot pass the gate.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from collectors.transcript_collector import TRANSCRIPT_STATUS_PATH


# A transcript older than this is considered stale (not from the
# current run). Overridable for slow CI pipelines.
MAX_AGE_SECONDS = int(os.getenv("EMH_TRANSCRIPT_MAX_AGE_SECONDS", "7200"))

# Minimum exchanges for a "whole interview" (greeting + a few
# real turns). A 1-2 line capture is truncated, not an interview.
MIN_ASSISTANT_TURNS = 2
MIN_USER_TURNS = 1

# Capture-coverage floor: even when the drive COMPLETED the
# interview (status.complete=True with turn_count driven
# candidate turns), the capture channel may have caught only a
# fraction of it. Judging that fragment as "the whole interview"
# penalizes the interviewer for turns that happened but were not
# captured. Captured candidate turns must cover at least this
# fraction of the driven turns. Kept below 1.0 deliberately:
# the collector de-duplicates repeated content (cycled candidate
# answers on long interviews legitimately collapse), so exact
# equality is not achievable.
MIN_CAPTURE_COVERAGE = float(
    os.getenv("EMH_MIN_CAPTURE_COVERAGE", "0.5")
)


class TranscriptValidationError(ValueError):
    """The transcript is empty, stale or truncated - do not score."""


@dataclass
class TranscriptStatus:
    complete: bool | None
    turn_count: int | None
    reached_cap: bool | None
    captured_at: float | None
    age_seconds: float | None
    # Why an incomplete interview ended, when known: the room/
    # server tearing the connection down is an environment
    # condition and must be reported as such, not as an agent
    # that silently died.
    room_disconnected: bool | None = None
    conclusion_reason: str | None = None


def load_status(
    path: Path = TRANSCRIPT_STATUS_PATH,
) -> TranscriptStatus:
    path = Path(path)
    if not path.exists():
        return TranscriptStatus(None, None, None, None, None)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A corrupt/unreadable sidecar means provenance is
        # UNKNOWN - same as no sidecar. Crashing here would
        # surface as an unclassified evaluator error instead of
        # a capture-layer verdict.
        return TranscriptStatus(None, None, None, None, None)
    if not isinstance(data, dict):
        return TranscriptStatus(None, None, None, None, None)
    captured_at = data.get("captured_at")
    age = (
        time.time() - captured_at
        if isinstance(captured_at, (int, float))
        else None
    )
    return TranscriptStatus(
        complete=data.get("complete"),
        turn_count=data.get("turn_count"),
        reached_cap=data.get("reached_cap"),
        captured_at=captured_at,
        age_seconds=age,
        room_disconnected=data.get("room_disconnected"),
        conclusion_reason=data.get("conclusion_reason"),
    )


def capture_coverage_problem(
    user_turns: int,
    status: TranscriptStatus,
    *,
    min_coverage: float = MIN_CAPTURE_COVERAGE,
) -> str | None:
    """
    Cross-check the captured transcript against what the drive
    actually did. status.turn_count is the number of candidate
    turns the E2E run DROVE; user_turns is how many the capture
    channel actually caught. When coverage is too low, the
    transcript is a fragment of a longer interview - a captured
    turn proves that exchange happened, but a MISSING turn never
    proves it did not - so judging the fragment as the whole
    interview is invalid. Returns a human-readable problem, or
    None when coverage is acceptable/unknowable.
    """

    driven = status.turn_count
    if not driven or driven <= 0:
        return None
    coverage = user_turns / driven
    if coverage >= min_coverage:
        return None
    return (
        f"Capture coverage too low: the drive completed {driven} "
        f"candidate turn(s) but only {user_turns} were captured "
        f"({coverage:.0%} < required {min_coverage:.0%}). The "
        "transcript is a FRAGMENT of the interview - judging it "
        "as the whole interview would penalize turns that "
        "happened but were not captured (CAPTURE LIMITATION, "
        "not a bot failure). Override the floor with "
        "EMH_MIN_CAPTURE_COVERAGE if deliberate."
    )


def validate_transcript_turns(
    turns: list[dict],
    *,
    status: TranscriptStatus | None = None,
    require_complete: bool = True,
    max_age_seconds: int = MAX_AGE_SECONDS,
) -> None:
    """
    Raise TranscriptValidationError unless `turns` is a complete,
    current, non-empty interview. Called before any AI scoring.
    """

    # --- empty / truncated by structure ----------------------
    if not turns:
        raise TranscriptValidationError(
            "Transcript is EMPTY - nothing to evaluate."
        )
    assistant_turns = sum(1 for t in turns if t.get("role") == "assistant")
    user_turns = sum(1 for t in turns if t.get("role") == "user")
    if assistant_turns < MIN_ASSISTANT_TURNS or user_turns < MIN_USER_TURNS:
        raise TranscriptValidationError(
            "Transcript is TRUNCATED - only "
            f"{assistant_turns} interviewer / {user_turns} candidate "
            f"turn(s) (need >= {MIN_ASSISTANT_TURNS}/"
            f"{MIN_USER_TURNS}). A partial interview must not be "
            "scored as a whole interview."
        )

    status = status or load_status()

    # --- completeness (truncated interview) -------------------
    if require_complete:
        if status.complete is False:
            raise TranscriptValidationError(
                "Transcript is marked INCOMPLETE by the capture run "
                f"(turns={status.turn_count}, "
                f"reached_cap={status.reached_cap}). The interview "
                "did not reach its closing statement - refusing to "
                "evaluate a truncated interview."
            )
        if status.complete is None:
            raise TranscriptValidationError(
                "No capture status sidecar found next to the "
                "transcript, so completion cannot be confirmed. Run "
                "the E2E capture (tests/e2e/test_bot_responsiveness"
                ".py) or set require_complete=False deliberately."
            )

        # A complete DRIVE with a fragmentary CAPTURE must not be
        # judged as the whole interview either.
        coverage_problem = capture_coverage_problem(
            user_turns, status
        )
        if coverage_problem:
            raise TranscriptValidationError(coverage_problem)

    # --- staleness --------------------------------------------
    if status.age_seconds is not None and status.age_seconds > max_age_seconds:
        raise TranscriptValidationError(
            "Transcript is STALE - captured "
            f"{status.age_seconds / 3600:.1f}h ago (limit "
            f"{max_age_seconds / 3600:.1f}h). Re-run the E2E "
            "capture so the evaluation scores the CURRENT "
            "interview, not an old one."
        )


def classify_status_for_scoring(
    status: TranscriptStatus | None = None,
    max_age_seconds: int | None = None,
) -> tuple[str, str]:
    """
    Decide how an AI-quality/rubric consumer should treat the
    current capture status BEFORE scoring. Returns
    (verdict, reason) with verdict one of:

      "ok"            - fresh, complete capture: scoring allowed.
      "skip-upstream" - the capture run recorded an INCOMPLETE
                        interview: that upstream bot/capture
                        failure is already reported by the
                        capture test, so downstream scoring
                        must SKIP instead of duplicating it.
      "fail-missing"  - no status sidecar: provenance unknown,
                        scoring must FAIL loudly (capture layer).
      "fail-stale"    - the capture is older than the freshness
                        window: scoring must FAIL loudly
                        (session/capture layer), never silently
                        reuse an old interview.
    """

    status = status or load_status()
    if max_age_seconds is None:
        max_age_seconds = MAX_AGE_SECONDS

    if status.complete is None:
        return (
            "fail-missing",
            "No capture status sidecar exists, so completeness "
            "and freshness cannot be confirmed (CAPTURE layer). "
            "Run the E2E capture: pytest tests/e2e/"
            "test_bot_responsiveness.py -s",
        )

    if (
        status.age_seconds is not None
        and status.age_seconds > max_age_seconds
    ):
        return (
            "fail-stale",
            f"The capture is {status.age_seconds / 3600:.1f}h "
            f"old (limit {max_age_seconds / 3600:.1f}h) - "
            "refusing to score a PREVIOUS interview as the "
            "current one (SESSION/CAPTURE layer). Re-run the "
            "E2E capture with a fresh INTERVIEW_URL.",
        )

    if status.complete is False:
        if status.room_disconnected:
            cause = (
                "the ROOM/SERVER disconnected mid-interview "
                "(environment condition, not an agent or "
                "candidate failure)"
            )
        elif status.reached_cap:
            cause = (
                "the safety cap was hit without a conclusion "
                "signal (agent dead, over-long or looping)"
            )
        else:
            cause = "it never reached a conclusion signal"
        return (
            "skip-upstream",
            "The capture run recorded an INCOMPLETE interview "
            f"(turns={status.turn_count}, "
            f"reached_cap={status.reached_cap}): {cause}. That "
            "upstream failure is already reported by "
            "tests/e2e/test_bot_responsiveness.py - skipping "
            "here instead of duplicating it as a quality "
            "failure. A truncated interview is never scored.",
        )

    return ("ok", "capture is fresh and complete")


def transcript_status_line(
    turns: list[dict],
    status: TranscriptStatus | None = None,
) -> str:
    """One-line transcript status for evaluation reports."""

    status = status or load_status()
    assistant_turns = sum(1 for t in turns if t.get("role") == "assistant")
    user_turns = sum(1 for t in turns if t.get("role") == "user")
    age = (
        f"{status.age_seconds / 60:.1f}min"
        if status.age_seconds is not None
        else "unknown"
    )
    return (
        f"transcript: {len(turns)} turns "
        f"({assistant_turns} interviewer / {user_turns} candidate), "
        f"complete={status.complete}, captured {age} ago"
    )