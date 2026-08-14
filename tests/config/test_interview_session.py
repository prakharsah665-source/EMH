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


# ============================================================
# classify_status_for_scoring (evaluation/transcript_validation)
# ============================================================

import time as _time

from evaluation.transcript_validation import (
    TranscriptStatus,
    classify_status_for_scoring,
)


def _status(complete, age_seconds):
    captured = (
        _time.time() - age_seconds
        if age_seconds is not None
        else None
    )
    return TranscriptStatus(
        complete=complete,
        turn_count=5,
        reached_cap=False,
        captured_at=captured,
        age_seconds=age_seconds,
    )


def test_fresh_complete_capture_is_scorable():
    verdict, _ = classify_status_for_scoring(_status(True, 60))
    assert verdict == "ok"


def test_incomplete_capture_skips_as_upstream_failure():
    verdict, reason = classify_status_for_scoring(
        _status(False, 60)
    )
    assert verdict == "skip-upstream"
    assert "already reported" in reason
    assert "test_bot_responsiveness" in reason


def test_missing_status_fails_at_capture_layer():
    verdict, reason = classify_status_for_scoring(
        _status(None, None)
    )
    assert verdict == "fail-missing"
    assert "CAPTURE" in reason


def test_stale_capture_fails_even_if_complete():
    verdict, reason = classify_status_for_scoring(
        _status(True, 3 * 3600)
    )
    assert verdict == "fail-stale"
    assert "refusing to score" in reason.lower()
