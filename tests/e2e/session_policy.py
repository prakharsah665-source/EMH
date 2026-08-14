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

    try:
        url, claims = require_fresh_interview_url()
    except InterviewSessionError as error:
        pytest.fail(str(error))

    if consumed_session_entry(claims) is None:
        return url, claims

    isolated = isolated_room_test_url()
    if isolated:
        isolated_claims = decode_interview_claims(isolated)
        if consumed_session_entry(isolated_claims) is None:
            print(
                f"{test_name}: primary session already "
                "consumed - using the isolated "
                "EMH_ROOM_TESTS_URL session."
            )
            return isolated, isolated_claims

    pytest.skip(
        "SESSION ALREADY CONSUMED - the primary interview "
        "session's room was already joined (normally by the "
        f"full-interview evaluation), so {test_name} cannot "
        "join a fresh room. This is a session/environment "
        "condition, not a bot failure. Provide a second fresh "
        "URL via EMH_ROOM_TESTS_URL (or run this test alone "
        "against a fresh INTERVIEW_URL) to execute it."
    )


def mark_room_joined(
    claims: InterviewClaims, test_name: str
) -> None:
    """Record the room join in the used-session ledger."""

    mark_session_consumed(claims, f"{test_name} room join")
