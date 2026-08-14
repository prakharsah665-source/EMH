"""
Session policy for ROOM-JOINING e2e tests.

Joining the interview room fires the agent greeting and
consumes the session's freshness. The full-interview
evaluation (test_bot_responsiveness) owns the primary session
in a complete run, so room-joining tests that come after it
must either use an isolated second session
(EMH_ROOM_TESTS_URL) or skip with SESSION ALREADY CONSUMED -
never re-enter a consumed room and report the resulting
silence as a bot failure.

Setup-screen tests (system config, audio config, recording
consent, permissions, socket, launch) do NOT join the room and
therefore do not use this policy: for them, re-entering via
"Continue interview" is intentional continuation.
"""

import pytest

from config.interview_session import (
    InterviewClaims,
    InterviewSessionError,
    consumed_session_entry,
    decode_interview_claims,
    mark_session_consumed,
    require_fresh_interview_url,
)
from config.session_provisioner import isolated_room_test_url


def resolve_room_session(
    test_name: str,
) -> tuple[str, InterviewClaims]:
    """
    Return (url, claims) for a test that will JOIN the
    interview room, or skip/fail with the correct
    session-layer classification.
    """

    # Room-joining tests must NEVER be the first consumer of
    # the primary session: if the full-interview capture failed
    # BEFORE joining the room, the primary session is the only
    # way to retry that capture, and a room-joiner grabbing it
    # would destroy that option (this happened live 2026-08-14).
    # They therefore run ONLY on the isolated
    # EMH_ROOM_TESTS_URL session.
    isolated = isolated_room_test_url()
    if isolated:
        isolated_claims = decode_interview_claims(isolated)
        if consumed_session_entry(isolated_claims) is None:
            print(
                f"{test_name}: using the isolated "
                "EMH_ROOM_TESTS_URL session."
            )
            return isolated, isolated_claims
        pytest.skip(
            "SESSION ALREADY CONSUMED - the isolated "
            "EMH_ROOM_TESTS_URL session's room was already "
            f"joined, so {test_name} cannot join a fresh room. "
            "This is a session/environment condition, not a "
            "bot failure. Provide a fresh EMH_ROOM_TESTS_URL."
        )

    # Fail loudly on a broken primary config either way, so a
    # missing/expired URL is still reported at the right layer.
    try:
        require_fresh_interview_url()
    except InterviewSessionError as error:
        pytest.fail(str(error))

    pytest.skip(
        f"ISOLATED SESSION REQUIRED - {test_name} joins the "
        "interview room, which fires the agent greeting and "
        "consumes the session. The primary INTERVIEW_URL is "
        "reserved for the full-interview evaluation "
        "(test_bot_responsiveness), so this test only runs "
        "with a second fresh URL in EMH_ROOM_TESTS_URL. "
        "Session-policy skip, not a bot failure."
    )


def mark_room_joined(
    claims: InterviewClaims, test_name: str
) -> None:
    """Record the room join in the used-session ledger."""

    mark_session_consumed(claims, f"{test_name} room join")
