"""
Cross-process interview-session lock.

The EMH app enforces a ONE-TAB/one-session lock: a second browser
joining the same interview session ejects the first ("Interview
Already Open"). Two harness processes driving the same URL at
once therefore destroy each other's run (observed live
2026-08-17: a second test_bot_responsiveness instance ejected the
first mid-interview and the first saw a "transport close" that
looked like an agent freeze).

acquire_session_lock() takes a per-session file lock under
artifacts/session_locks/ recording the holder pid + test name;
a live holder makes the caller fail with SESSION IN USE
(session/environment classification, never a bot failure). A
lock whose pid is dead is stale and is taken over.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from config.interview_session import InterviewClaims, InterviewSessionError


LOCK_DIR = Path("artifacts/session_locks")


def _lock_path(claims: InterviewClaims) -> Path:
    return LOCK_DIR / (
        f"candidate-{claims.candidate_id}_job-{claims.job_id}"
        f"_iat-{claims.issued_at}.lock"
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def current_holder(claims: InterviewClaims) -> dict | None:
    """Return the live holder record for this session, else None."""

    path = _lock_path(claims)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if _pid_alive(int(record.get("pid", 0))):
        return record
    return None  # stale lock (holder died)


@contextmanager
def acquire_session_lock(claims: InterviewClaims, holder: str):
    """
    Hold the session lock for the duration of the block. Raises
    InterviewSessionError("SESSION IN USE ...") if another LIVE
    process holds it. Re-entrant for the same pid.
    """

    path = _lock_path(claims)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    existing = current_holder(claims)
    if existing and int(existing.get("pid", 0)) != os.getpid():
        raise InterviewSessionError(
            "SESSION IN USE\n"
            f"Interview session candidate {claims.candidate_id} / "
            f"job {claims.job_id} is currently being driven by "
            f"'{existing.get('holder')}' (pid {existing.get('pid')}, "
            "since "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(existing.get('since', 0)))}). "
            "The EMH app allows ONE tab per session - a second "
            "joiner ejects the first - so this test refuses to "
            "run concurrently against it. Wait for that run to "
            "finish or give this test its own fresh URL. "
            "(Session/environment condition, not a bot failure.)"
        )

    reentrant = existing is not None  # same pid already holds it
    if not reentrant:
        path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "holder": holder,
                    "since": time.time(),
                    "candidate_id": claims.candidate_id,
                    "job_id": claims.job_id,
                    "issued_at": claims.issued_at,
                }
            ),
            encoding="utf-8",
        )
    try:
        yield path
    finally:
        if not reentrant:
            try:
                path.unlink()
            except OSError:
                pass
