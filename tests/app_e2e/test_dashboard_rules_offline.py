"""
Offline unit tests for the dashboard validation rules.
No browser, no network - guards the PASS/FAIL logic itself.
"""

import pytest

from app_e2e.dashboard import config as config_module
from app_e2e.dashboard.auth import token_validity
from app_e2e.dashboard.checks import FAIL, NA, PASS
from app_e2e.dashboard.expected import derive_expected, fmt_int, js_round
from app_e2e.dashboard.schema import (
    validate_consistency,
    validate_data_types,
    validate_required_fields,
    validate_values,
)
from app_e2e.dashboard.ui import (
    DonutCard,
    KpiCard,
    UiSnapshot,
    parse_donut_text,
    parse_number,
)
from app_e2e.dashboard.validator import (
    credits_checks,
    donut_checks,
    kpi_checks,
    pipeline_checks,
    stage_checks,
)


def make_payload(**overrides) -> dict:
    payload = {
        "candidates": {"aggregate": {"count": 1200}},
        "jobs": {"aggregate": {"count": 40}},
        "totalInterviews": {"aggregate": {"count": 800}},
        "credits": [
            {"remaining_credits": 10, "company": {"company_id": 1, "company_name": "Acme"}},
            {"remaining_credits": 40, "company": {"company_id": 2, "company_name": "Beta"}},
        ],
        "totalCreditsDisbursed": {"aggregate": {"sum": {"remaining_credits": 50}}},
        "pipeline": {
            "notOpened": 100,
            "opened": 0,
            "started": 800,
            "startedOnly": 5,
            "inProgress": 3,
            "paused": 42,
            "expired": 60,
            "completed": 705,
            "candidateExit": 50,
        },
    }
    payload.update(overrides)
    return payload


def failures(results):
    return [r for r in results if r.status == FAIL]


# ------------------------------------------------------------
# Derivation
# ------------------------------------------------------------

def test_js_round_rounds_halves_up():
    assert js_round(2.5) == 3
    assert js_round(0.5) == 1
    assert js_round(70.4) == 70
    assert fmt_int(3635) == "3,635"


def test_expected_values_follow_dashboard_formulas():
    expected = derive_expected(
        make_payload(), is_super_admin=True, companies_count=55
    )
    stats = expected.stats
    assert stats.total == 800
    assert stats.opened_total == 0 + 5 + 3 + 42 + 705 + 50 == 805
    assert stats.invites == 805 + 100 + 60 == 965
    assert stats.completion_rate == js_round(705 / 800 * 100) == 88

    kpis = expected.kpis
    assert kpis["active candidates"].value == 1200
    assert kpis["active candidates"].subtitle == "across 55 companies"
    assert kpis["active jobs"].subtitle is None
    assert kpis["total interviews"].value == 800
    assert kpis["total interviews"].badge == "88%"
    assert kpis["total interviews"].subtitle == "completion rate"
    assert kpis["credits allocated"].value == 50
    assert kpis["credits allocated"].subtitle == "1 companies below 25"

    denominator = 100 + 0 + 5 + 3 + 42 + 705 + 50 + 60
    assert expected.stage_denominator == denominator == 965
    assert expected.stage_rows == [
        ("Link not opened", 100, js_round(100 / 965 * 100)),
        ("Link opened", 0, 0),
        ("Started", 5, js_round(5 / 965 * 100)),
        ("In progress", 3, 0),
        ("Paused", 42, 4),
        ("Completed", 705, 73),
        ("Candidate exit", 50, 5),
        ("Invite expired", 60, 6),
    ]
    assert list(expected.pipeline_counts) == [
        "Not opened", "Opened", "Started", "In progress", "Paused",
        "Completed", "Candidate exit", "Expired",
    ]
    assert expected.donuts["Communications"] == (965, [("Email", 965)])
    assert expected.donuts["Open rate"] == (
        965, [("Opened", 805), ("Not opened", 100), ("Expired", 60)]
    )
    assert expected.donuts["Response mix"] == (
        805,
        [("Completed", 705), ("Candidate exit", 50), ("Paused", 42),
         ("In progress", 3), ("Started", 5)],
    )
    assert expected.credits_by_company == [("Acme", 10), ("Beta", 40)]


def test_zero_entries_are_dropped_from_donuts_and_completion_is_clamped():
    payload = make_payload()
    payload["pipeline"].update(
        {"expired": 0, "startedOnly": 0, "completed": 900}
    )
    expected = derive_expected(payload, is_super_admin=True, companies_count=1)
    assert expected.stats.completion_rate == 100
    # Opened = opened + startedOnly + inProgress + paused + completed + candidateExit
    assert expected.donuts["Open rate"][1] == [("Opened", 995), ("Not opened", 100)]
    assert ("Started", 0) not in expected.donuts["Response mix"][1]


def test_non_super_admin_gets_no_credits_card_or_company_subtitle():
    expected = derive_expected(
        make_payload(), is_super_admin=False, companies_count=55
    )
    assert "credits allocated" not in expected.kpis
    assert expected.kpis["active candidates"].subtitle is None
    assert expected.credits_by_company is None


def test_empty_pipeline_yields_zero_completion_and_unit_denominator():
    payload = make_payload(pipeline={})
    expected = derive_expected(payload, is_super_admin=True, companies_count=1)
    assert expected.stats.completion_rate == 0
    assert expected.stage_denominator == 1
    assert all(pct == 0 for _, _, pct in expected.stage_rows)


# ------------------------------------------------------------
# Schema
# ------------------------------------------------------------

def test_schema_passes_on_well_formed_payload():
    payload = make_payload()
    for validate in (
        validate_required_fields, validate_data_types,
        validate_values, validate_consistency,
    ):
        assert not failures(validate(payload)), validate.__name__


def test_required_fields_reports_missing_top_level_and_nested():
    payload = make_payload()
    del payload["jobs"]
    del payload["pipeline"]["expired"]
    failed = failures(validate_required_fields(payload))
    names = " | ".join(r.name for r in failed)
    assert "all top-level fields present" in names
    assert "jobs" in " ".join(r.actual for r in failed)
    assert "pipeline.expired present" in names


def test_data_types_reject_strings_bools_and_floats():
    payload = make_payload()
    payload["candidates"]["aggregate"]["count"] = "1200"
    payload["pipeline"]["paused"] = True
    payload["credits"][0]["remaining_credits"] = 10.0
    failed = failures(validate_data_types(payload))
    names = " | ".join(r.name for r in failed)
    assert "candidates.aggregate.count is integer" in names
    assert "pipeline.paused is integer" in names
    assert "credits[*] remaining_credits:int" in names


def test_values_flag_negatives_duplicates_and_unknown_stages():
    payload = make_payload()
    payload["credits"][0]["remaining_credits"] = -5
    payload["credits"][1]["company"]["company_id"] = 1
    payload["pipeline"]["completed"] = -1
    payload["pipeline"]["ghosted"] = 4
    failed = failures(validate_values(payload))
    names = " | ".join(r.name for r in failed)
    assert "remaining_credits >= 0" in names
    assert "company_id unique" in names
    assert "pipeline.completed >= 0" in names
    assert "no unknown stage fields" in names


def test_consistency_flags_total_mismatch_and_credit_sum_mismatch():
    payload = make_payload(
        totalInterviews={"aggregate": {"count": 900}},
        totalCreditsDisbursed={"aggregate": {"sum": {"remaining_credits": 99}}},
    )
    payload["pipeline"]["completed"] = 801
    failed = failures(validate_consistency(payload))
    names = " | ".join(r.name for r in failed)
    assert "pipeline.started equals totalInterviews" in names
    assert "pipeline.completed <= pipeline.started" in names
    assert "totalCreditsDisbursed equals the sum" in names


# ------------------------------------------------------------
# UI parsing and comparison
# ------------------------------------------------------------

def test_parse_number_and_donut_text():
    assert parse_number("3,635") == 3635
    assert parse_number("24") == 24
    assert parse_number("70%") is None
    assert parse_number("") is None

    donut = parse_donut_text(
        "Response mix",
        "Response mix\nActive interview statuses\n2,554\nTOTAL\n"
        "Completed\n2,430\nPaused\n100\nIn progress\n24\n",
    )
    assert donut.total_text == "2,554"
    assert donut.legend == [
        ("Completed", "2,430"), ("Paused", "100"), ("In progress", "24")
    ]


def make_snapshot(expected, *, credits_value_override=None) -> UiSnapshot:
    cards = {}
    for key, kpi in expected.kpis.items():
        cards[key] = KpiCard(
            title=kpi.title.upper(),
            value_text=fmt_int(kpi.value),
            subtitle=kpi.subtitle,
            badge=kpi.badge,
        )
    if credits_value_override is not None:
        cards["credits allocated"].value_text = credits_value_override
    donuts = {
        title: DonutCard(
            title=title,
            total_text=fmt_int(total),
            legend=[(n, fmt_int(v)) for n, v in legend],
        )
        for title, (total, legend) in expected.donuts.items()
    }
    return UiSnapshot(
        page_url="https://app.example/dashboard",
        page_title="EMH | Dashboard",
        kpi_cards=cards,
        stage_rows=[
            (label, fmt_int(value), f"{pct}%")
            for label, value, pct in expected.stage_rows
        ],
        donuts=donuts,
        pipeline_labels=[
            fmt_int(v) for v in expected.pipeline_counts.values() if v > 0
        ] + ["0", "550", "1100"],
        credits_button_present=True,
        credits_by_company=[
            (name, str(value)) for name, value in expected.credits_by_company
        ],
        dashboard_requests=[],
        intercepted_payload=None,
        lookup_payload=None,
        user_data={},
        roles=["admin"],
        screenshot_path=None,
        login_elapsed_s=0.0,
        render_elapsed_s=0.0,
        main_text="",
    )


def test_ui_comparisons_pass_when_page_matches_derived_values():
    expected = derive_expected(
        make_payload(), is_super_admin=True, companies_count=55
    )
    snapshot = make_snapshot(expected)
    for checker in (
        kpi_checks, stage_checks, pipeline_checks, donut_checks, credits_checks
    ):
        results = checker(snapshot, expected)
        assert not failures(results), checker.__name__
        assert all(r.status in (PASS, NA) for r in results)
    # "Opened" is 0 -> its bar label is N/A, not a failure.
    assert any(
        r.status == NA and "'Opened' bar label" in r.name
        for r in pipeline_checks(snapshot, expected)
    )


def test_ui_comparison_names_the_mismatching_card():
    expected = derive_expected(
        make_payload(), is_super_admin=True, companies_count=55
    )
    snapshot = make_snapshot(expected, credits_value_override="49")
    failed = failures(kpi_checks(snapshot, expected))
    assert [r.name for r in failed] == ["'Credits allocated' value"]
    assert failed[0].expected == "50" and failed[0].actual == "49"


def test_stage_row_mismatch_names_the_row():
    expected = derive_expected(
        make_payload(), is_super_admin=True, companies_count=55
    )
    snapshot = make_snapshot(expected)
    snapshot.stage_rows[5] = ("Completed", "704", "73%")
    failed = failures(stage_checks(snapshot, expected))
    assert [r.name for r in failed] == ["'Completed' count"]


def test_credits_dialog_mismatch_is_reported_per_row():
    expected = derive_expected(
        make_payload(), is_super_admin=True, companies_count=55
    )
    snapshot = make_snapshot(expected)
    snapshot.credits_by_company = [("Acme", "10"), ("Beta", "41")]
    failed = failures(credits_checks(snapshot, expected))
    assert len(failed) == 1
    assert "Beta=40" in failed[0].actual and "Beta=41" in failed[0].actual


# ------------------------------------------------------------
# Configuration and token handling
# ------------------------------------------------------------

def test_load_config_lists_every_missing_variable(monkeypatch):
    for name in ("LOGIN_URL", "LOGIN_EMAIL", "LOGIN_PASSWORD",
                 "Dashboard_API_URL", "DASHBOARD_API_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(config_module.DashboardConfigError) as excinfo:
        config_module.load_config()
    message = str(excinfo.value)
    for name in ("LOGIN_URL", "LOGIN_EMAIL", "LOGIN_PASSWORD", "Dashboard_API_URL"):
        assert name in message


def test_load_config_derives_endpoints_and_strips_whitespace(monkeypatch):
    monkeypatch.setenv("LOGIN_URL", "https://app.example/ ")
    monkeypatch.setenv("LOGIN_EMAIL", "qa@example.com")
    monkeypatch.setenv("LOGIN_PASSWORD", "secret")
    monkeypatch.setenv("Dashboard_API_URL", " https://api.example/dashboard-data/")
    monkeypatch.setenv("ACCESS_TOKEN", " tok ")
    monkeypatch.delenv("LOGIN_API_URL", raising=False)
    monkeypatch.delenv("DASHBOARD_URL", raising=False)
    config = config_module.load_config()
    assert config.dashboard_api_url == "https://api.example/dashboard-data"
    assert config.login_api_url == "https://api.example/auth/sign-in"
    assert config.dashboard_page_url == "https://app.example/dashboard"
    assert config.access_token == "tok"


def test_token_validity_states():
    assert token_validity(None)[0] is False
    assert token_validity("not-a-jwt")[0] is False

    import base64
    import json

    def jwt(exp):
        payload = base64.urlsafe_b64encode(
            json.dumps({"exp": exp}).encode()
        ).decode().rstrip("=")
        return f"eyJhbGciOiJSUzI1NiJ9.{payload}.sig"

    assert token_validity(jwt(1), now=100)[0] is False
    assert token_validity(jwt(200), now=100)[0] is True
