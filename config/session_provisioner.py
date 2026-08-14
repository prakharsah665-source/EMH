"""
Interview session provisioning.

Goal: one FRESH interview session per complete run for the
full-interview evaluation, without creating a session per test.

Today no session-provisioning API is exposed to this repo, so
the default source remains the manually pasted INTERVIEW_URL /
EMH_INTERVIEW_URL (see config.interview_session). This module
adds:

1. provision_interview_url() - mints a fresh session when a
   provisioning endpoint is configured, otherwise returns None
   so callers fall back to the manual URL. The exact endpoint
   the EMH backend needs to expose is documented below.
2. isolated_room_test_url() - optional second session for the
   few tests that genuinely require joining the interview room
   in isolation AFTER the full-interview evaluation consumed
   the primary session (EMH_ROOM_TESTS_URL).

REQUIRED PROVISIONING ENDPOINT (not yet available)
--------------------------------------------------
    POST {EMH_PROVISION_API_URL}
    Authorization: Bearer <ACCESS_TOKEN>          (dashboard JWT)
    Content-Type: application/json
    Body:     {"job_id": <int>, "candidate": {"name": ..., "email": ...}}
    Response: {"interview_url": "https://hiring.easemyhiring.ai/interview/<JWT>?version=v1"}

The returned URL must embed the standard interview JWT
(candidate_id / job_id / company_id / iat / exp) so the
existing freshness and used-session guards keep working
unchanged. Configure via:

    EMH_PROVISION_API_URL  - the endpoint above
    ACCESS_TOKEN           - dashboard bearer token (.env)
"""

import json
import os
import urllib.error
import urllib.request

from config.interview_session import (
    InterviewSessionError,
    decode_interview_claims,
)


PROVISION_API_URL = os.getenv("EMH_PROVISION_API_URL")
PROVISION_JOB_ID = os.getenv("EMH_PROVISION_JOB_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


def provision_interview_url() -> str | None:
    """
    Mint a fresh interview session via the provisioning API.

    Returns None when no provisioning endpoint is configured
    (the normal case today) so the caller falls back to the
    manual INTERVIEW_URL / EMH_INTERVIEW_URL. Raises
    InterviewSessionError - an ENVIRONMENT classification,
    never a bot failure - when the endpoint is configured but
    unusable.
    """

    if not PROVISION_API_URL:
        return None

    if not ACCESS_TOKEN:
        raise InterviewSessionError(
            "EMH_PROVISION_API_URL is configured but "
            "ACCESS_TOKEN is missing - cannot authenticate the "
            "provisioning call. (Environment/configuration "
            "issue, not a bot failure.)"
        )

    body = json.dumps(
        {"job_id": int(PROVISION_JOB_ID)}
        if PROVISION_JOB_ID
        else {}
    ).encode("utf-8")

    request = urllib.request.Request(
        PROVISION_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError) as error:
        raise InterviewSessionError(
            "Session provisioning call FAILED "
            f"({PROVISION_API_URL}): {error}. "
            "(Environment/provisioning issue, not a bot "
            "failure.) Fall back by unsetting "
            "EMH_PROVISION_API_URL and pasting a manual "
            "INTERVIEW_URL."
        ) from error

    url = payload.get("interview_url")
    if not url:
        raise InterviewSessionError(
            "Provisioning endpoint responded without an "
            "'interview_url' field - response keys: "
            f"{sorted(payload)}. (Provisioning contract "
            "issue, not a bot failure.)"
        )

    # Validate the minted URL decodes like a normal session.
    decode_interview_claims(url)
    return url


def isolated_room_test_url() -> str | None:
    """
    Optional SECOND fresh session for room-joining tests that
    run after the full-interview evaluation consumed the
    primary session. Returns None when not configured - those
    tests then skip with SESSION ALREADY CONSUMED instead of
    producing false bot failures.
    """

    url = os.getenv("EMH_ROOM_TESTS_URL")
    if not url:
        return None
    decode_interview_claims(url)
    return url