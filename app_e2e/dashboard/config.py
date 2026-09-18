"""
Configuration for the Dashboard data validation.

Every URL, credential and token is read from the environment
(.env via python-dotenv). Nothing is hardcoded.

Required variables (exact names as stored in .env):

  LOGIN_URL          dashboard web-app login page; its origin also
                     serves /dashboard
  LOGIN_EMAIL        dashboard user
  LOGIN_PASSWORD     dashboard password
  Dashboard_API_URL  dashboard-data API base,
                     e.g. https://<api-host>/dashboard-data
  ACCESS_TOKEN       (optional) pre-issued idToken. Used while it is
                     still valid AND accepted by the API; otherwise a
                     fresh sign-in is performed with the credentials
                     above and the reason is recorded in the report.

Optional overrides:

  LOGIN_API_URL        sign-in endpoint
                       (default: <Dashboard_API_URL origin>/auth/sign-in)
  DASHBOARD_URL        dashboard page
                       (default: <LOGIN_URL origin>/dashboard)
  DASHBOARD_HEADLESS   "0" / "false" to watch the browser (default on)
  DASHBOARD_TIMEOUT_MS per-step browser timeout (default 60000)
  DASHBOARD_DRIFT_TOLERANCE
                       allowed absolute difference between the payload
                       the page received and the harness's own API
                       read, for live environments (default 0)
"""

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv


load_dotenv()


class DashboardConfigError(RuntimeError):
    """A required environment variable is missing or malformed."""


def _env(*names: str) -> str | None:
    """First non-empty value among the given variable names, stripped."""

    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise DashboardConfigError(
            f"Not an absolute URL: {url!r} (expected https://host/...)"
        )
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


@dataclass(frozen=True)
class DashboardConfig:
    login_url: str
    login_email: str
    login_password: str
    dashboard_api_url: str
    access_token: str | None
    login_api_url: str
    dashboard_page_url: str
    headless: bool
    timeout_ms: int
    drift_tolerance: int

    @property
    def api_origin(self) -> str:
        return origin_of(self.dashboard_api_url)

    @property
    def app_origin(self) -> str:
        return origin_of(self.login_url)


def load_config() -> DashboardConfig:
    """
    Build the config from the environment, failing with ONE message
    that lists every missing variable.
    """

    values = {
        "LOGIN_URL": _env("LOGIN_URL"),
        "LOGIN_EMAIL": _env("LOGIN_EMAIL"),
        "LOGIN_PASSWORD": _env("LOGIN_PASSWORD"),
        "Dashboard_API_URL": _env("Dashboard_API_URL", "DASHBOARD_API_URL"),
    }

    missing = [name for name, value in values.items() if not value]
    if missing:
        raise DashboardConfigError(
            "Dashboard validation is not configured - missing in .env: "
            + ", ".join(missing)
            + ". Nothing is hardcoded; add the variables and re-run."
        )

    dashboard_api_url = values["Dashboard_API_URL"].rstrip("/")
    login_url = values["LOGIN_URL"]

    headless_flag = (_env("DASHBOARD_HEADLESS") or "1").lower()

    return DashboardConfig(
        login_url=login_url,
        login_email=values["LOGIN_EMAIL"],
        login_password=values["LOGIN_PASSWORD"],
        dashboard_api_url=dashboard_api_url,
        access_token=_env("ACCESS_TOKEN"),
        login_api_url=(
            _env("LOGIN_API_URL")
            or f"{origin_of(dashboard_api_url)}/auth/sign-in"
        ),
        dashboard_page_url=(
            _env("DASHBOARD_URL") or f"{origin_of(login_url)}/dashboard"
        ),
        headless=headless_flag not in ("0", "false", "no", "off"),
        timeout_ms=int(_env("DASHBOARD_TIMEOUT_MS") or 60_000),
        drift_tolerance=int(_env("DASHBOARD_DRIFT_TOLERANCE") or 0),
    )


def mask_email(email: str | None) -> str:
    if not email:
        return "(none)"
    local, _, domain = email.partition("@")
    if not domain:
        return "***"
    return f"{local[:1]}***@{domain}"


def mask_token(token: str | None) -> str:
    if not token:
        return "(none)"
    if len(token) <= 16:
        return "***"
    return f"{token[:8]}…{token[-4:]} ({len(token)} chars)"
