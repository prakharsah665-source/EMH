"""
Session policy for the E2E suite.

Two separate FRESH sessions per run:

  INTERVIEW_URL -> reserved EXCLUSIVELY for the full-interview
                   evaluation (test_bot_responsiveness), which
                   joins the room and consumes the session.
  EMH_TESTS_URL -> every other E2E test. Setup-screen tests
                   re-enter it as intentional continuation
                   (they never join the room); room-joining
                   tests (continue_to_interview / interview_room
                   / livekit) share ONE join of it (tests/e2e/
                   shared_room.py): the first to run needs it
                   UNCONSUMED and records the join in the
                   used-session ledger, the others attach to
                   the same live room.

Room-joining tests must NEVER touch the primary session: if the
full-interview capture failed before joining, that session is
the only way to retry the capture, and the app's one-tab lock
means a concurrent joiner would eject the evaluator anyway.
"""

import pytest

from config.interview_session import (
    InterviewClaims,
    InterviewSessionError,
    consumed_session_entry,
    mark_session_consumed,
    require_fresh_tests_url,
    tests_url_configured,
)


def resolve_tests_session(test_name: str) -> tuple[str, InterviewClaims]:
    """
    (url, claims) of the EMH_TESTS_URL session for a test that
    only walks the setup screens (never joins the room). Skips
    with a session-policy reason when EMH_TESTS_URL is missing,
    fails when it is stale/expired/identical to INTERVIEW_URL.
    """

    if not tests_url_configured():
        pytest.skip(
            f"EMH_TESTS_URL not set - {test_name} runs only on the "
            "isolated tests session (INTERVIEW_URL is reserved for "
            "the full-interview evaluation). Session-policy skip, "
            "not a bot failure."
        )
    try:
        return require_fresh_tests_url()
    except InterviewSessionError as error:
        pytest.fail(str(error))


def resolve_room_session(
    test_name: str,
) -> tuple[str, InterviewClaims]:
    """
    (url, claims) for a test that will JOIN the interview room:
    the EMH_TESTS_URL session, which must be UNCONSUMED (the
    agent greeting fires on room join). Skips with SESSION
    ALREADY CONSUMED otherwise - never a red "bot" failure.
    """

    url, claims = resolve_tests_session(test_name)
    if consumed_session_entry(claims) is not None:
        pytest.skip(
            "SESSION ALREADY CONSUMED - the EMH_TESTS_URL session's "
            f"room was already joined, so {test_name} cannot join a "
            "fresh room. This is a session/environment condition, "
            "not a bot failure. Provide a fresh EMH_TESTS_URL."
        )
    print(f"{test_name}: using the isolated EMH_TESTS_URL session.")
    return url, claims


def mark_room_joined(
    claims: InterviewClaims, test_name: str
) -> None:
    """Record the room join in the used-session ledger."""

    mark_session_consumed(claims, f"{test_name} room join")
