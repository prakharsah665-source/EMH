"""
Authentication for the Dashboard API.

Token resolution order:

  1. ACCESS_TOKEN from .env - used only while it is a well-formed,
     unexpired JWT AND the Dashboard API accepts it (the API rejects
     tokens minted for another environment with "Invalid token").
  2. Otherwise POST <LOGIN_API_URL> {email, password} - the same call
     the web app's login form makes - and use the returned idToken.

Whichever path is taken is recorded in the report. JWT claims are
decoded WITHOUT signature verification purely for reporting (expiry,
audience, subject); the API is the authority on validity.
"""

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import requests

from app_e2e.dashboard.config import DashboardConfig, mask_email


class DashboardAuthError(RuntimeError):
    """Sign-in failed or produced no usable token."""


def decode_jwt_claims(token: str) -> dict:
    """Decode the payload segment of a JWT (no verification)."""

    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("not a JWT (expected header.payload.signature)")
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception as error:  # noqa: BLE001 - reported to the caller
        raise ValueError(f"JWT payload is not base64url JSON: {error}")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%SZ"
    )


def token_validity(
    token: str | None, now: float | None = None
) -> tuple[bool, str]:
    """(usable, reason) for a stored token, judged locally."""

    if not token:
        return False, "no ACCESS_TOKEN configured"
    try:
        claims = decode_jwt_claims(token)
    except ValueError as error:
        return False, f"ACCESS_TOKEN is not a decodable JWT ({error})"
    now = time.time() if now is None else now
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        if exp <= now:
            return False, f"ACCESS_TOKEN expired at {_iso(exp)}"
        return True, f"ACCESS_TOKEN valid until {_iso(exp)}"
    return True, "ACCESS_TOKEN carries no exp claim"


@dataclass
class SignInResult:
    token: str
    status_code: int
    elapsed_s: float
    response: dict


def sign_in(
    config: DashboardConfig, session: requests.Session | None = None
) -> SignInResult:
    """POST the credentials to the sign-in endpoint and return idToken."""

    http = session or requests.Session()
    started = time.perf_counter()
    try:
        response = http.post(
            config.login_api_url,
            json={
                "email": config.login_email,
                "password": config.login_password,
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
    except requests.RequestException as error:
        raise DashboardAuthError(
            f"Sign-in request to {config.login_api_url} failed: {error}"
        )
    elapsed = time.perf_counter() - started

    if response.status_code not in (200, 201):
        raise DashboardAuthError(
            f"Sign-in as {mask_email(config.login_email)} at "
            f"{config.login_api_url} returned HTTP "
            f"{response.status_code}: {response.text[:300]}"
        )

    try:
        body = response.json()
    except ValueError:
        raise DashboardAuthError(
            "Sign-in returned a non-JSON body: " + response.text[:300]
        )

    token = body.get("idToken") if isinstance(body, dict) else None
    if not token:
        next_step = (
            (body.get("nextStep") or {}).get("signInStep")
            if isinstance(body, dict)
            else None
        )
        raise DashboardAuthError(
            "Sign-in succeeded without an idToken "
            f"(signInStep={next_step!r}, isSignedIn="
            f"{body.get('isSignedIn') if isinstance(body, dict) else None!r})"
            " - the account may need a new password or MFA."
        )

    return SignInResult(
        token=token,
        status_code=response.status_code,
        elapsed_s=elapsed,
        response=body,
    )


@dataclass
class TokenResolution:
    token: str
    source: str
    claims: dict
    notes: list[str] = field(default_factory=list)
    sign_in: SignInResult | None = None


def resolve_token(
    config: DashboardConfig,
    accepted_by_api: Callable[[str], tuple[bool, str]],
    session: requests.Session | None = None,
) -> TokenResolution:
    """
    Prefer the stored ACCESS_TOKEN when it is valid and the API
    accepts it; otherwise sign in with LOGIN_EMAIL / LOGIN_PASSWORD.
    """

    notes: list[str] = []

    usable, reason = token_validity(config.access_token)
    if usable:
        accepted, api_reason = accepted_by_api(config.access_token)
        if accepted:
            notes.append(
                f"Stored ACCESS_TOKEN accepted by the Dashboard API "
                f"({reason})."
            )
            return TokenResolution(
                token=config.access_token,
                source="ACCESS_TOKEN (.env)",
                claims=decode_jwt_claims(config.access_token),
                notes=notes,
            )
        notes.append(
            "Stored ACCESS_TOKEN rejected by the Dashboard API "
            f"({api_reason}); falling back to sign-in."
        )
    else:
        notes.append(f"Stored ACCESS_TOKEN not used: {reason}.")

    result = sign_in(config, session)
    claims = decode_jwt_claims(result.token)
    notes.append(
        f"Signed in via {config.login_api_url} as "
        f"{mask_email(config.login_email)}: HTTP {result.status_code} "
        f"in {result.elapsed_s:.2f}s; token expires "
        f"{_iso(claims['exp']) if isinstance(claims.get('exp'), (int, float)) else 'n/a'}."
    )
    return TokenResolution(
        token=result.token,
        source="sign-in (LOGIN_EMAIL / LOGIN_PASSWORD)",
        claims=claims,
        notes=notes,
        sign_in=result,
    )
