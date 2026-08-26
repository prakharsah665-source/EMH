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


# ============================================================
# URL resolution: ONE primary session, split-config warning
# ============================================================

import base64
import json


def _url(candidate, job, iat):
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"candidate_id": candidate, "job_id": job,
             "company_id": 3, "iat": iat,
             "exp": int(__import__("time").time()) + 100_000}
        ).encode()
    ).rstrip(b"=").decode()
    return f"https://x.test/interview/h.{payload}.s?version=v1"


@pytest.fixture
def clean_url_env(monkeypatch):
    monkeypatch.delenv("EMH_INTERVIEW_URL", raising=False)
    monkeypatch.setattr(interview_session, "_split_config_warned", False)
    yield monkeypatch


def test_override_wins_over_primary(clean_url_env, capsys):
    primary, override = _url(1, 2, 100), _url(9, 2, 200)
    clean_url_env.setattr(interview_session, "INTERVIEW_URL", primary)
    clean_url_env.setenv("EMH_INTERVIEW_URL", override)
    assert interview_session.get_interview_url() == override
    assert "DIFFERENT interview sessions" in capsys.readouterr().out


def test_no_warning_when_only_primary(clean_url_env, capsys):
    primary = _url(1, 2, 100)
    clean_url_env.setattr(interview_session, "INTERVIEW_URL", primary)
    assert interview_session.get_interview_url() == primary
    assert "DIFFERENT" not in capsys.readouterr().out
    assert interview_session.interview_url_configured()


def test_no_warning_when_both_same_session(clean_url_env, capsys):
    url = _url(1, 2, 100)
    clean_url_env.setattr(interview_session, "INTERVIEW_URL", url)
    clean_url_env.setenv("EMH_INTERVIEW_URL", url)
    interview_session.get_interview_url()
    assert "DIFFERENT" not in capsys.readouterr().out


def test_missing_url_raises(clean_url_env):
    clean_url_env.setattr(interview_session, "INTERVIEW_URL", None)
    assert not interview_session.interview_url_configured()
    with pytest.raises(InterviewSessionError):
        interview_session.get_interview_url()


# ============================================================
# Two-session setup: EMH_TESTS_URL + cross-process lock
# ============================================================

def test_tests_url_must_differ_from_primary(clean_url_env):
    url = _url(1, 2, 100)
    clean_url_env.setattr(interview_session, "INTERVIEW_URL", url)
    clean_url_env.setenv("EMH_TESTS_URL", url)
    with pytest.raises(InterviewSessionError, match="SAME session"):
        interview_session.require_fresh_tests_url()


def test_tests_url_resolves_separately(clean_url_env):
    clean_url_env.setattr(interview_session, "INTERVIEW_URL", _url(1, 2, 100))
    clean_url_env.setenv("EMH_TESTS_URL", _url(9, 2, 200))
    url, claims = interview_session.require_fresh_tests_url()
    assert claims.candidate_id == 9
    assert interview_session.require_fresh_interview_url()[1].candidate_id == 1


def test_missing_tests_url_raises(clean_url_env):
    clean_url_env.delenv("EMH_TESTS_URL", raising=False)
    clean_url_env.delenv("EMH_ROOM_TESTS_URL", raising=False)
    assert not interview_session.tests_url_configured()
    with pytest.raises(InterviewSessionError, match="EMH_TESTS_URL"):
        interview_session.get_tests_url()


def test_legacy_room_tests_url_still_honoured(clean_url_env):
    clean_url_env.delenv("EMH_TESTS_URL", raising=False)
    clean_url_env.setenv("EMH_ROOM_TESTS_URL", _url(9, 2, 200))
    assert interview_session.get_tests_url().endswith("?version=v1")


def test_session_lock_blocks_live_holder_and_takes_over_stale(tmp_path, monkeypatch):
    import json, os
    from config import session_lock
    monkeypatch.setattr(session_lock, "LOCK_DIR", tmp_path)
    claims = make_claims()

    with session_lock.acquire_session_lock(claims, "run A") as path:
        assert path.exists()
        # Same pid re-enters fine.
        with session_lock.acquire_session_lock(claims, "run A again"):
            pass
        assert path.exists()
        # A different LIVE pid is refused.
        record = json.loads(path.read_text())
        record["pid"] = os.getppid()  # a live pid that is not us
        path.write_text(json.dumps(record))
        with pytest.raises(InterviewSessionError, match="SESSION IN USE"):
            with session_lock.acquire_session_lock(claims, "run B"):
                pass
        # A dead pid is stale and taken over.
        record["pid"] = 999_999_999
        path.write_text(json.dumps(record))
        with session_lock.acquire_session_lock(claims, "run C"):
            assert json.loads(path.read_text())["holder"] == "run C"
    assert not path.exists()
