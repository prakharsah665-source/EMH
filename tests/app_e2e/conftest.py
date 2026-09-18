"""
Session wiring for the application E2E suite (Dashboard data
validation).

ONE token resolution, ONE independent API read and ONE browser
capture per pytest session, shared by every test. Every check row a
test produces is collected in a registry and written to

    reports/dashboard_validation_report.html
    reports/dashboard_validation_results.json

when the session ends, so the pytest verdict and the report always
agree.

Completely independent of tests/e2e: no interview session, no
session lock, no transcript capture. Setup problems are recorded per
stage (config / auth / api / ui) and surfaced by the tests that need
that stage, instead of erroring the whole session.
"""

import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from app_e2e.dashboard.api_client import (
    ApiResponse,
    DashboardApiClient,
    token_accepted,
)
from app_e2e.dashboard.auth import (
    DashboardAuthError,
    TokenResolution,
    resolve_token,
)
from app_e2e.dashboard.checks import (
    CheckRegistry,
    CheckResult,
    assert_all_pass,
)
from app_e2e.dashboard.config import (
    DashboardConfig,
    DashboardConfigError,
    load_config,
    mask_email,
    mask_token,
)
from app_e2e.dashboard.expected import (
    SUPER_ADMIN_ROLE,
    ExpectedDashboard,
    derive_expected,
)
from app_e2e.dashboard.report import build_report, write_reports
from app_e2e.dashboard.ui import (
    DashboardUiError,
    UiSnapshot,
    capture_dashboard,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_HTML = ROOT / "reports" / "dashboard_validation_report.html"
REPORT_JSON = ROOT / "reports" / "dashboard_validation_results.json"
ARTIFACT_DIR = ROOT / "artifacts" / "dashboard"

SUITE_PATH_MARKER = "tests/app_e2e/"


@dataclass
class DashboardRun:
    config: DashboardConfig | None = None
    token: TokenResolution | None = None
    api_response: ApiResponse | None = None
    lookup_response: ApiResponse | None = None
    snapshot: UiSnapshot | None = None
    expected: ExpectedDashboard | None = None
    expected_source: str = ""
    registry: CheckRegistry = field(default_factory=CheckRegistry)
    notes: list[str] = field(default_factory=list)
    setup_errors: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    # -------- accessors --------

    @property
    def api_payload(self) -> dict | None:
        if self.api_response and isinstance(self.api_response.json, dict):
            return self.api_response.json
        return None

    @property
    def ui_payload(self) -> dict | None:
        return self.snapshot.intercepted_payload if self.snapshot else None

    # -------- test helpers --------

    def require(self, *stages: str) -> None:
        """Fail the calling test with the recorded setup error(s)."""

        problems = [
            f"[{stage}] {self.setup_errors[stage]}"
            for stage in stages
            if stage in self.setup_errors
        ]
        if problems:
            pytest.fail(
                "Dashboard validation setup failed:\n" + "\n".join(problems)
            )

    def record(self, results: list[CheckResult], heading: str) -> None:
        self.registry.add(results)
        assert_all_pass(results, heading)


RUN_KEY = pytest.StashKey[DashboardRun]()
_TEST_OUTCOMES: list[dict] = []


def _roles_from_claims(claims: dict) -> list[str]:
    raw = claims.get("https://hasura.io/jwt/claims")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    if isinstance(raw, dict):
        roles = raw.get("x-hasura-allowed-roles") or []
        default = raw.get("x-hasura-default-role")
        if default and default not in roles:
            roles = [default, *roles]
        return [str(r) for r in roles]
    return []


def _iso(epoch) -> str:
    if not isinstance(epoch, (int, float)):
        return "n/a"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%SZ"
    )


def _build_run() -> DashboardRun:
    run = DashboardRun()

    def log(message: str) -> None:
        print(f"[dashboard] {message}")

    # ---- 1. configuration (.env) ----
    try:
        run.config = load_config()
    except DashboardConfigError as error:
        run.setup_errors["config"] = str(error)
        log(str(error))
        return run
    config = run.config
    log(
        f"login {config.login_url} | api {config.dashboard_api_url} | "
        f"user {mask_email(config.login_email)} | headless={config.headless}"
    )

    # ---- 2. token + independent API read ----
    http = requests.Session()
    started = time.perf_counter()
    try:
        run.token = resolve_token(
            config, lambda token: token_accepted(config, token), http
        )
        run.notes.extend(run.token.notes)
        for note in run.token.notes:
            log(note)
    except (DashboardAuthError, requests.RequestException) as error:
        run.setup_errors["auth"] = str(error)
        log(f"auth failed: {error}")
    run.timings["auth_s"] = time.perf_counter() - started

    if run.token is not None:
        started = time.perf_counter()
        try:
            client = DashboardApiClient(config, run.token.token, http)
            run.api_response = client.dashboard_data()
            run.lookup_response = client.lookup()
            run.notes.append(
                "Independent API read: getDashboardData HTTP "
                f"{run.api_response.status_code} in "
                f"{run.api_response.elapsed_s:.2f}s; GetLookup HTTP "
                f"{run.lookup_response.status_code} in "
                f"{run.lookup_response.elapsed_s:.2f}s."
            )
            log(run.notes[-1])
            if not run.api_response.ok:
                run.setup_errors["api"] = (
                    f"getDashboardData returned HTTP "
                    f"{run.api_response.status_code}: "
                    f"{run.api_response.snippet()}"
                )
        except requests.RequestException as error:
            run.setup_errors["api"] = f"API request failed: {error}"
            log(run.setup_errors["api"])
        run.timings["api_s"] = time.perf_counter() - started
    else:
        run.setup_errors["api"] = "no token - authentication failed"

    # ---- 3. browser capture of the live dashboard ----
    started = time.perf_counter()
    try:
        run.snapshot = asyncio.run(
            capture_dashboard(config, ARTIFACT_DIR, log=log)
        )
        run.notes.append(
            f"Browser: login {run.snapshot.login_elapsed_s:.1f}s, dashboard "
            f"data + render {run.snapshot.render_elapsed_s:.1f}s, roles "
            f"{run.snapshot.roles or '(unknown)'}, screenshot "
            f"{run.snapshot.screenshot_path}."
        )
    except DashboardUiError as error:
        run.setup_errors["ui"] = str(error)
        log(f"ui failed: {error}")
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        run.setup_errors["ui"] = (
            f"{type(error).__name__}: {error}\n"
            + "".join(traceback.format_exception(error)[-4:])
        )
        log(f"ui failed: {run.setup_errors['ui']}")
    run.timings["ui_s"] = time.perf_counter() - started

    # ---- 4. expected UI values from the payload the page received ----
    payload = run.ui_payload
    if payload is not None:
        run.expected_source = "payload intercepted from the page's own request"
    else:
        payload = run.api_payload
        run.expected_source = "harness API read (page payload unavailable)"

    if payload is not None:
        roles: list[str] = []
        if run.snapshot and run.snapshot.roles:
            roles = run.snapshot.roles
        elif run.token:
            roles = _roles_from_claims(run.token.claims)

        lookup = None
        if run.snapshot and isinstance(run.snapshot.lookup_payload, dict):
            lookup = run.snapshot.lookup_payload
        elif run.lookup_response and isinstance(run.lookup_response.json, dict):
            lookup = run.lookup_response.json
        companies = (lookup or {}).get("companies") or []

        run.expected = derive_expected(
            payload,
            is_super_admin=SUPER_ADMIN_ROLE in roles,
            companies_count=len(companies) if isinstance(companies, list) else 0,
        )
        run.notes.append(
            f"Expected values derived from the {run.expected_source}; "
            f"super admin={run.expected.is_super_admin} (roles {roles}), "
            f"{run.expected.companies_count} companies in lookup."
        )
        log(run.notes[-1])

    return run


@pytest.fixture(scope="session")
def dashboard_run(request) -> DashboardRun:
    run = _build_run()
    request.config.stash[RUN_KEY] = run
    return run


# ------------------------------------------------------------
# Report generation
# ------------------------------------------------------------

def pytest_runtest_logreport(report):
    if SUITE_PATH_MARKER not in report.nodeid.replace("\\", "/"):
        return
    if report.when == "call" or (
        report.when == "setup" and report.outcome != "passed"
    ):
        _TEST_OUTCOMES.append(
            {
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "duration": float(report.duration or 0.0),
            }
        )


def pytest_sessionfinish(session, exitstatus):
    run = session.config.stash.get(RUN_KEY, None)
    if run is None:
        return

    config = run.config
    claims = run.token.claims if run.token else {}
    environment = {
        "LOGIN_URL": config.login_url if config else "(not configured)",
        "Dashboard page": run.snapshot.page_url if run.snapshot else (
            config.dashboard_page_url if config else ""
        ),
        "Dashboard_API_URL": config.dashboard_api_url if config else "",
        "Sign-in endpoint": config.login_api_url if config else "",
        "User": mask_email(config.login_email) if config else "",
        "Roles": ", ".join(run.snapshot.roles) if run.snapshot and run.snapshot.roles
        else ", ".join(_roles_from_claims(claims)),
        "Token source": run.token.source if run.token else "(none)",
        "Token expires": _iso(claims.get("exp")) if claims else "n/a",
        "Stored ACCESS_TOKEN": mask_token(config.access_token) if config else "",
        "Expected values source": run.expected_source or "(none)",
        "Browser": (
            f"Chromium headless={config.headless}" if config else ""
        ),
        "Timings": ", ".join(
            f"{k}={v:.1f}s" for k, v in run.timings.items()
        ),
    }

    report = build_report(
        registry_results=[r.to_dict() for r in run.registry.results],
        environment=environment,
        notes=run.notes,
        test_outcomes=list(_TEST_OUTCOMES),
        screenshot_path=run.snapshot.screenshot_path if run.snapshot else None,
        api_payload=run.ui_payload or run.api_payload,
        setup_errors=run.setup_errors,
    )
    write_reports(report, REPORT_HTML, REPORT_JSON)

    counts = report["counts"]
    line = (
        f"Dashboard validation {report['overall']}: {counts['PASS']} passed, "
        f"{counts['FAIL']} failed, {counts['N/A']} n/a "
        f"({counts['total']} checks) -> {REPORT_HTML}"
    )
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line("")
        terminal.write_line(line)
    else:
        print(line)
