"""
Dashboard data validation - live checks against the QA application.

API side (independent read with the resolved token):
  authentication -> required fields -> data types -> value ranges ->
  cross-field consistency.

UI side (Playwright login + intercepted page payload):
  data source -> drift vs independent read -> KPI cards -> stage
  breakdown -> pipeline chart -> donut charts -> credits by company.

Every test prints its PASS/FAIL rows, records them for the report and
fails on any FAIL row. Setup failures (missing .env, login rejected,
page did not render) fail the dependent tests with the stage message.
"""

import time

from app_e2e.dashboard.checks import check
from app_e2e.dashboard.config import mask_email, mask_token
from app_e2e.dashboard.schema import (
    validate_consistency,
    validate_data_types,
    validate_required_fields,
    validate_values,
)
from app_e2e.dashboard.validator import (
    credits_checks,
    donut_checks,
    drift_checks,
    kpi_checks,
    pipeline_checks,
    source_checks,
    stage_checks,
)


CAT_AUTH = "API authentication"
CAT_LOGIN = "Dashboard login"


def _ui_ready(run):
    run.require("config", "ui")
    assert run.snapshot is not None
    assert run.expected is not None, (
        "no payload available to derive expected dashboard values"
    )
    return run.snapshot, run.expected


# ============================================================
# API
# ============================================================

def test_api_authentication_and_response(dashboard_run):
    run = dashboard_run
    run.require("config", "auth", "api")

    config, token, response, lookup = (
        run.config, run.token, run.api_response, run.lookup_response
    )
    claims = token.claims
    now = time.time()
    exp = claims.get("exp")

    results = [
        check(
            CAT_AUTH, f"bearer token obtained via {token.source}",
            bool(token.token), expected="token", actual=mask_token(token.token),
        ),
        check(
            CAT_AUTH, "token subject matches LOGIN_EMAIL",
            str(claims.get("email", "")).lower() == config.login_email.lower(),
            expected=mask_email(config.login_email),
            actual=mask_email(claims.get("email")),
        ),
        check(
            CAT_AUTH, "token is not expired",
            isinstance(exp, (int, float)) and exp > now,
            expected=f"exp > {int(now)}", actual=exp,
        ),
        check(
            CAT_AUTH, "getDashboardData accepts the token",
            response.ok, expected="HTTP 2xx",
            actual=f"HTTP {response.status_code} {response.snippet(80)}",
        ),
        check(
            CAT_AUTH, "getDashboardData returns a JSON object",
            "json" in response.content_type.lower()
            and isinstance(response.json, dict),
            expected="application/json object",
            actual=f"{response.content_type or '(no content-type)'} / "
            f"{type(response.json).__name__}",
        ),
        check(
            CAT_AUTH, "getDashboardData responds within 10s",
            response.elapsed_s < 10, expected="< 10s",
            actual=f"{response.elapsed_s:.2f}s",
        ),
        check(
            CAT_AUTH, "GetLookup accepts the token",
            lookup is not None and lookup.ok and isinstance(lookup.json, dict),
            expected="HTTP 2xx object",
            actual=f"HTTP {lookup.status_code}" if lookup else "not called",
        ),
    ]
    run.record(results, "API authentication and response")


def test_api_required_fields(dashboard_run):
    run = dashboard_run
    run.require("config", "auth", "api")
    run.record(
        validate_required_fields(run.api_payload),
        "API required fields",
    )


def test_api_data_types(dashboard_run):
    run = dashboard_run
    run.require("config", "auth", "api")
    run.record(validate_data_types(run.api_payload), "API data types")


def test_api_values(dashboard_run):
    run = dashboard_run
    run.require("config", "auth", "api")
    run.record(validate_values(run.api_payload), "API value ranges")


def test_api_cross_field_consistency(dashboard_run):
    run = dashboard_run
    run.require("config", "auth", "api")
    run.record(
        validate_consistency(run.api_payload),
        "API cross-field consistency",
    )


# ============================================================
# UI
# ============================================================

def test_ui_login_and_data_source(dashboard_run):
    run = dashboard_run
    run.require("config", "ui")
    snapshot, config = run.snapshot, run.config

    results = [
        check(
            CAT_LOGIN, "login form accepted LOGIN_EMAIL / LOGIN_PASSWORD "
            "and reached the dashboard",
            "/dashboard" in snapshot.page_url,
            expected=config.dashboard_page_url, actual=snapshot.page_url,
        ),
        check(
            CAT_LOGIN, "page title names the dashboard",
            "dashboard" in snapshot.page_title.lower(),
            expected="… Dashboard", actual=snapshot.page_title,
        ),
        check(
            CAT_LOGIN, "signed-in user matches LOGIN_EMAIL",
            str(snapshot.user_data.get("email", "")).lower()
            == config.login_email.lower(),
            expected=mask_email(config.login_email),
            actual=mask_email(snapshot.user_data.get("email")),
        ),
    ]
    results += source_checks(snapshot, config)
    run.record(results, "Dashboard login and data source")


def test_ui_payload_matches_independent_api_read(dashboard_run):
    run = dashboard_run
    run.require("config", "ui", "auth", "api")
    run.record(
        drift_checks(run.ui_payload, run.api_payload, run.config.drift_tolerance),
        "Page payload vs independent API read",
    )


def test_ui_kpi_cards_match_api(dashboard_run):
    snapshot, expected = _ui_ready(dashboard_run)
    dashboard_run.record(
        kpi_checks(snapshot, expected), "KPI cards vs API"
    )


def test_ui_stage_breakdown_matches_api(dashboard_run):
    snapshot, expected = _ui_ready(dashboard_run)
    dashboard_run.record(
        stage_checks(snapshot, expected), "Stage breakdown vs API"
    )


def test_ui_pipeline_chart_matches_api(dashboard_run):
    snapshot, expected = _ui_ready(dashboard_run)
    dashboard_run.record(
        pipeline_checks(snapshot, expected), "Interview pipeline chart vs API"
    )


def test_ui_donut_charts_match_api(dashboard_run):
    snapshot, expected = _ui_ready(dashboard_run)
    dashboard_run.record(
        donut_checks(snapshot, expected), "Donut charts vs API"
    )


def test_ui_credits_by_company_match_api(dashboard_run):
    snapshot, expected = _ui_ready(dashboard_run)
    dashboard_run.record(
        credits_checks(snapshot, expected), "Credits by company dialog vs API"
    )
