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
from urllib.parse import urlsplit

from config.settings import INTERVIEW_URL


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
