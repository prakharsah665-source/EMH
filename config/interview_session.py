"""
Interview session / URL resolution with a staleness guard.

The EMH interview URL embeds a session JWT
(.../interview/<JWT>?version=v1) whose payload identifies the
candidate/job/company and carries iat/exp. There is no session-
provisioning API in this repo, so a fresh interview is supplied
by pasting a new INTERVIEW_URL into .env per run.

This module refuses to run a test against the known-stale
demo session (candidate 8581 / job 1391) or an expired token,
so downstream LiveKit/audio/recording/config failures caused by
reusing a dead room fail LOUDLY here with a clear reason instead
of surfacing as confusing downstream errors.

The JWT signature is NOT verified (we only read public claims
for routing/diagnostics) - the token itself is never logged.
"""

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from config.settings import INTERVIEW_URL


# Run-scoped ledger of interview sessions this harness has
# already driven into the interview room. Joining the room
# triggers the agent greeting and advances server-side state,
# so a session recorded here can never again serve as a FRESH
# full interview. The launch page cannot tell us this ("Continue
# interview" only means "opened before", even for a finished
# interview), hence the local ledger. External consumption
# (someone else joining the room) is not detectable.
SESSION_LEDGER_PATH = Path("artifacts/session_ledger.json")


# The demo/stale session the suite must stop reusing.
STALE_CANDIDATE_ID = 8581
STALE_JOB_ID = 1391

# Escape hatch for deliberate debugging against the stale URL.
ALLOW_STALE = os.getenv("EMH_ALLOW_STALE_INTERVIEW") == "1"


class InterviewSessionError(RuntimeError):
    """The configured interview URL is missing, stale or expired."""


@dataclass(frozen=True)
class InterviewClaims:
    candidate_id: int | None
    job_id: int | None
    company_id: int | None
    issued_at: int | None
    expires_at: int | None
    raw: dict

    @property
    def is_stale_demo(self) -> bool:
        return (
            self.candidate_id == STALE_CANDIDATE_ID
            and self.job_id == STALE_JOB_ID
        )

    @property
    def is_expired(self) -> bool:
        return (
            self.expires_at is not None
            and self.expires_at < int(time.time())
        )


def _decode_jwt_payload(token: str) -> dict:
    if token.count(".") != 2:
        raise InterviewSessionError(
            "Interview URL path is not a JWT (expected three "
            "dot-separated segments)."
        )
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception as error:  # malformed base64/JSON
        raise InterviewSessionError(
            f"Could not decode the interview JWT payload: {error}"
        ) from error


def decode_interview_claims(url: str) -> InterviewClaims:
    """Decode the public claims from an interview URL's path JWT."""

    path = urlsplit(url).path
    token = path.rstrip("/").rsplit("/", 1)[-1]
    claims = _decode_jwt_payload(token)
    return InterviewClaims(
        candidate_id=claims.get("candidate_id"),
        job_id=claims.get("job_id"),
        company_id=claims.get("company_id"),
        issued_at=claims.get("iat"),
        expires_at=claims.get("exp"),
        raw=claims,
    )


def get_interview_url() -> str:
    """
    Resolve the interview URL, honouring a per-run override
    (EMH_INTERVIEW_URL takes precedence over INTERVIEW_URL so a
    fresh session can be injected without editing .env).
    """

    url = os.getenv("EMH_INTERVIEW_URL") or INTERVIEW_URL
    if not url:
        raise InterviewSessionError(
            "No interview URL configured. Set INTERVIEW_URL (or "
            "EMH_INTERVIEW_URL) to a fresh interview session URL."
        )
    return url


def require_fresh_interview_url() -> tuple[str, InterviewClaims]:
    """
    Return a validated (url, claims). Raises InterviewSessionError
    for a missing, stale-demo or expired session so every test
    starts from a known-fresh room.
    """

    url = get_interview_url()
    claims = decode_interview_claims(url)

    if claims.is_stale_demo and not ALLOW_STALE:
        raise InterviewSessionError(
            "Refusing to run against the STALE demo interview "
            f"(candidate {STALE_CANDIDATE_ID} / job {STALE_JOB_ID}). "
            "Paste a FRESH interview URL into INTERVIEW_URL (or set "
            "EMH_INTERVIEW_URL). Reusing this dead session is the "
            "root cause of the downstream LiveKit/audio/recording/"
            "config failures. Set EMH_ALLOW_STALE_INTERVIEW=1 only "
            "to deliberately debug against it."
        )

    if claims.is_expired and not ALLOW_STALE:
        raise InterviewSessionError(
            "The configured interview JWT is EXPIRED "
            f"(exp={claims.expires_at}). Paste a fresh interview "
            "URL into INTERVIEW_URL / EMH_INTERVIEW_URL."
        )

    return url, claims


# ============================================================
# Used-session ledger
#
# "Consumed" means THIS harness joined the interview room for
# the session (the agent greeting fires on room join, so the
# interview is no longer fresh). Setup-screen re-entries do
# NOT consume a session - clicking Start/Continue only walks
# the System Configuration flow.
# ============================================================

def _session_key(claims: InterviewClaims) -> str:
    return (
        f"candidate={claims.candidate_id}"
        f"/job={claims.job_id}"
        f"/iat={claims.issued_at}"
    )


def _load_ledger() -> dict:
    if not SESSION_LEDGER_PATH.exists():
        return {}
    try:
        return json.loads(
            SESSION_LEDGER_PATH.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}


def mark_session_consumed(
    claims: InterviewClaims,
    reason: str,
) -> None:
    """
    Record that this session's interview room was joined (the
    interview is no longer fresh). Called by any test that
    enters the interview room.
    """

    ledger = _load_ledger()
    ledger[_session_key(claims)] = {
        "candidate_id": claims.candidate_id,
        "job_id": claims.job_id,
        "issued_at": claims.issued_at,
        "consumed_at": time.time(),
        "reason": reason,
    }
    SESSION_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_LEDGER_PATH.write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )


def consumed_session_entry(
    claims: InterviewClaims,
) -> dict | None:
    """Return the ledger entry if this session was consumed."""

    return _load_ledger().get(_session_key(claims))


def require_unconsumed_session(
    claims: InterviewClaims,
) -> None:
    """
    Raise SESSION ALREADY CONSUMED for a session whose
    interview room was already joined. Session/environment
    classification - NEVER a bot failure.
    """

    entry = consumed_session_entry(claims)
    if entry is None or ALLOW_STALE:
        return

    raise InterviewSessionError(
        "SESSION ALREADY CONSUMED\n"
        f"This interview session (candidate "
        f"{claims.candidate_id} / job {claims.job_id}) already "
        "had its interview room joined by "
        f"'{entry.get('reason')}' at "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.get('consumed_at', 0)))}. "
        "The agent greeting fires on room join, so it can no "
        "longer serve as a FRESH full interview.\n"
        "This is a SESSION/ENVIRONMENT condition, not a bot "
        "failure. Paste a fresh interview URL into "
        "INTERVIEW_URL / EMH_INTERVIEW_URL (or set "
        "EMH_ALLOW_STALE_INTERVIEW=1 to deliberately reuse it)."
    )
