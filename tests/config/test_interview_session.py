"""
Offline unit tests for the used-session ledger and guard.
No browser, no network.
"""

import pytest

from config import interview_session
from config.interview_session import (
    InterviewClaims,
    InterviewSessionError,
    consumed_session_entry,
    mark_session_consumed,
    require_unconsumed_session,
)


def make_claims(candidate=1, job=2, iat=1000):
    return InterviewClaims(
        candidate_id=candidate,
        job_id=job,
        company_id=3,
        issued_at=iat,
        expires_at=None,
        raw={},
    )


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "session_ledger.json"
    monkeypatch.setattr(
        interview_session, "SESSION_LEDGER_PATH", path
    )
    return path


def test_unconsumed_session_passes(ledger):
    claims = make_claims()
    assert consumed_session_entry(claims) is None
    require_unconsumed_session(claims)  # must not raise


def test_consumed_session_is_refused_with_classification(ledger):
    claims = make_claims()
    mark_session_consumed(claims, "bot_responsiveness full interview")

    entry = consumed_session_entry(claims)
    assert entry is not None
    assert entry["reason"] == "bot_responsiveness full interview"

    with pytest.raises(InterviewSessionError) as excinfo:
        require_unconsumed_session(claims)

    message = str(excinfo.value)
    # The classification the whole suite relies on: session
    # layer, never a bot failure.
    assert "SESSION ALREADY CONSUMED" in message
    assert "not a bot failure" in message


def test_different_session_is_not_affected(ledger):
    mark_session_consumed(make_claims(candidate=1), "x")
    # Same candidate, new token mint (fresh iat) = new session.
    require_unconsumed_session(make_claims(candidate=1, iat=2000))
    # Different candidate entirely.
    require_unconsumed_session(make_claims(candidate=99))


def test_corrupt_ledger_is_treated_as_empty(ledger):
    ledger.write_text("{not json", encoding="utf-8")
    require_unconsumed_session(make_claims())  # must not raise
    # And marking still works (rewrites the file).
    mark_session_consumed(make_claims(), "y")
    assert consumed_session_entry(make_claims()) is not None
