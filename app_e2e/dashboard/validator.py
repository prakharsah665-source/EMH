"""
Compare what the dashboard page rendered (UiSnapshot) with what the
API payload says it must render (ExpectedDashboard).

Every comparison is a CheckResult row with the API-derived expected
value, the UI's actual value and PASS/FAIL, so a mismatch pinpoints
the widget and the numbers.
"""

from app_e2e.dashboard.checks import CheckResult, check, not_applicable
from app_e2e.dashboard.config import DashboardConfig
from app_e2e.dashboard.expected import ExpectedDashboard, fmt_int
from app_e2e.dashboard.schema import INTEGER_PATHS, CREDITS_SUM_PATH, get_path, is_int
from app_e2e.dashboard.ui import UiSnapshot, parse_number


CAT_SOURCE = "Dashboard data source"
CAT_KPI = "Dashboard KPI cards"
CAT_STAGES = "Dashboard stage breakdown"
CAT_PIPELINE = "Dashboard pipeline chart"
CAT_DONUTS = "Dashboard donut charts"
CAT_CREDITS = "Dashboard credits by company"


def _norm(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


# ------------------------------------------------------------
# Where the page got its data from
# ------------------------------------------------------------

def source_checks(
    snapshot: UiSnapshot, config: DashboardConfig
) -> list[CheckResult]:
    results: list[CheckResult] = []
    request = snapshot.intercepted_request

    results.append(
        check(
            CAT_SOURCE,
            "dashboard page called getDashboardData",
            request is not None,
            expected="1+ call",
            actual=f"{len(snapshot.dashboard_requests)} call(s)",
        )
    )
    if request is None:
        return results

    expected_url = f"{config.dashboard_api_url}/getDashboardData"
    actual_url = request["url"].split("?", 1)[0]
    results.append(
        check(
            CAT_SOURCE,
            "page fetches from the configured Dashboard_API_URL",
            actual_url == expected_url,
            expected=expected_url,
            actual=actual_url,
            detail=(
                "" if actual_url == expected_url else
                "The deployed web app talks to a different backend than "
                "Dashboard_API_URL - the API being validated is not the "
                "one behind the dashboard."
            ),
        )
    )
    results.append(
        check(
            CAT_SOURCE,
            "page request carried a bearer token",
            bool(request.get("authorized")),
            expected="Authorization header present",
            actual="present" if request.get("authorized") else "absent",
        )
    )
    results.append(
        check(
            CAT_SOURCE,
            "page request succeeded",
            request.get("status") == 200,
            expected="HTTP 200",
            actual=f"HTTP {request.get('status')}",
        )
    )
    results.append(
        check(
            CAT_SOURCE,
            "page received a JSON object payload",
            isinstance(request.get("payload"), dict),
            expected="object",
            actual=type(request.get("payload")).__name__,
        )
    )

    variables = (request.get("variables") or {}).get("variables") \
        if isinstance(request.get("variables"), dict) else None
    candidates_where = (
        (variables or {}).get("candidatesWhere") if variables else None
    )
    results.append(
        check(
            CAT_SOURCE,
            "default view excludes individual users "
            "(candidatesWhere.is_individual_user._eq == false)",
            candidates_where == {"is_individual_user": {"_eq": False}},
            expected='{"is_individual_user": {"_eq": false}}',
            actual=str(candidates_where),
        )
    )
    return results


def drift_checks(
    intercepted: dict | None,
    independent: dict | None,
    tolerance: int,
) -> list[CheckResult]:
    """
    The harness's own API read vs the payload the page received.
    Both are the same query seconds apart, so any difference is
    either live data changing or a non-deterministic backend.
    """

    results: list[CheckResult] = []
    if not isinstance(intercepted, dict) or not isinstance(independent, dict):
        results.append(
            not_applicable(
                CAT_SOURCE,
                "independent API read matches the page's payload",
                "one of the two payloads is unavailable",
            )
        )
        return results

    fields = [*INTEGER_PATHS, CREDITS_SUM_PATH]

    differences: list[str] = []
    for path in fields:
        _, a = get_path(intercepted, path)
        _, b = get_path(independent, path)
        if is_int(a) and is_int(b):
            if abs(a - b) > tolerance:
                differences.append(f"{path}: page={a} api={b}")
        elif a != b:
            differences.append(f"{path}: page={a!r} api={b!r}")

    page_credits = intercepted.get("credits")
    api_credits = independent.get("credits")
    if page_credits != api_credits:
        differences.append(
            f"credits: page={len(page_credits) if isinstance(page_credits, list) else page_credits!r} rows, "
            f"api={len(api_credits) if isinstance(api_credits, list) else api_credits!r} rows or values differ"
        )

    results.append(
        check(
            CAT_SOURCE,
            "independent API read matches the page's payload "
            f"(tolerance ±{tolerance})",
            not differences,
            expected="identical counts",
            actual="identical" if not differences else "; ".join(differences),
            detail=(
                "" if not differences else
                "Live data changed between the page load and the harness "
                "read, or the backend is non-deterministic. UI checks use "
                "the page's own payload, so this does not affect them."
            ),
        )
    )
    return results


# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------

def kpi_checks(
    snapshot: UiSnapshot, expected: ExpectedDashboard
) -> list[CheckResult]:
    results: list[CheckResult] = []

    for key, kpi in expected.kpis.items():
        card = snapshot.kpi_cards.get(key)
        results.append(
            check(
                CAT_KPI,
                f"'{kpi.title}' card is displayed",
                card is not None,
                expected="card present",
                actual="present" if card else (
                    "absent; cards shown: "
                    + ", ".join(c.title for c in snapshot.kpi_cards.values())
                ),
            )
        )
        if card is None:
            continue

        results.append(
            check(
                CAT_KPI,
                f"'{kpi.title}' value",
                card.value == kpi.value,
                expected=fmt_int(kpi.value),
                actual=card.value_text or "(empty)",
            )
        )
        results.append(
            check(
                CAT_KPI,
                f"'{kpi.title}' subtitle",
                _norm(card.subtitle) == _norm(kpi.subtitle),
                expected=kpi.subtitle or "(none)",
                actual=card.subtitle or "(none)",
            )
        )
        results.append(
            check(
                CAT_KPI,
                f"'{kpi.title}' badge",
                _norm(card.badge) == _norm(kpi.badge),
                expected=kpi.badge or "(none)",
                actual=card.badge or "(none)",
            )
        )

    unexpected = [
        c.title for k, c in snapshot.kpi_cards.items() if k not in expected.kpis
    ]
    results.append(
        check(
            CAT_KPI,
            "no unexpected KPI cards",
            not unexpected,
            expected=", ".join(k.title for k in expected.kpis.values()),
            actual=(
                "extra: " + ", ".join(unexpected) if unexpected
                else "none"
            ),
            detail=(
                "" if not unexpected or expected.is_super_admin else
                "The credits card must only be shown to super admins."
            ),
        )
    )
    return results


# ------------------------------------------------------------
# Stage breakdown
# ------------------------------------------------------------

def stage_checks(
    snapshot: UiSnapshot, expected: ExpectedDashboard
) -> list[CheckResult]:
    results: list[CheckResult] = []
    ui_rows = [
        row for row in snapshot.stage_rows
        if _norm(row[0]) != "remaining credits"
    ]
    results.append(
        check(
            CAT_STAGES,
            "stage rows displayed in order",
            [_norm(r[0]) for r in ui_rows]
            == [_norm(label) for label, _, _ in expected.stage_rows],
            expected=", ".join(label for label, _, _ in expected.stage_rows),
            actual=", ".join(r[0] for r in ui_rows) or "(no rows)",
        )
    )
    ui_by_label = {_norm(r[0]): r for r in ui_rows}
    for label, value, pct in expected.stage_rows:
        row = ui_by_label.get(_norm(label))
        if row is None:
            results.append(
                check(
                    CAT_STAGES, f"'{label}' count", False,
                    expected=fmt_int(value), actual="row missing",
                )
            )
            continue
        results.append(
            check(
                CAT_STAGES,
                f"'{label}' count",
                parse_number(row[1]) == value,
                expected=fmt_int(value),
                actual=row[1],
            )
        )
        results.append(
            check(
                CAT_STAGES,
                f"'{label}' percentage",
                _norm(row[2]) == f"{pct}%",
                expected=f"{pct}%",
                actual=row[2] or "(none)",
                detail=f"round({value} / {expected.stage_denominator} * 100)",
            )
        )
    return results


# ------------------------------------------------------------
# Pipeline bar chart
# ------------------------------------------------------------

def pipeline_checks(
    snapshot: UiSnapshot, expected: ExpectedDashboard
) -> list[CheckResult]:
    results: list[CheckResult] = []
    labels = set(snapshot.pipeline_labels)
    results.append(
        check(
            CAT_PIPELINE,
            "pipeline chart rendered with numeric labels",
            bool(labels),
            expected="labels present",
            actual=f"{len(snapshot.pipeline_labels)} numeric texts",
        )
    )
    for name, value in expected.pipeline_counts.items():
        if value <= 0:
            results.append(
                not_applicable(
                    CAT_PIPELINE,
                    f"'{name}' bar label",
                    "value is 0 - the chart draws no label",
                )
            )
            continue
        text = fmt_int(value)
        results.append(
            check(
                CAT_PIPELINE,
                f"'{name}' bar label",
                text in labels,
                expected=text,
                actual="present" if text in labels else
                f"not found among {sorted(labels)}",
            )
        )
    return results


# ------------------------------------------------------------
# Donut charts
# ------------------------------------------------------------

def donut_checks(
    snapshot: UiSnapshot, expected: ExpectedDashboard
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for title, (total, legend) in expected.donuts.items():
        donut = snapshot.donuts.get(title)
        results.append(
            check(
                CAT_DONUTS,
                f"'{title}' card is displayed",
                donut is not None,
                expected="card present",
                actual="present" if donut else "absent",
            )
        )
        if donut is None:
            continue
        results.append(
            check(
                CAT_DONUTS,
                f"'{title}' TOTAL",
                parse_number(donut.total_text) == total,
                expected=fmt_int(total),
                actual=donut.total_text or "(none)",
            )
        )
        ui_legend = [(_norm(n), parse_number(v)) for n, v in donut.legend]
        exp_legend = [(_norm(n), v) for n, v in legend]
        results.append(
            check(
                CAT_DONUTS,
                f"'{title}' legend entries",
                ui_legend == exp_legend,
                expected=", ".join(f"{n}={fmt_int(v)}" for n, v in legend)
                or "(none)",
                actual=", ".join(f"{n}={v}" for n, v in donut.legend)
                or "(none)",
            )
        )
    return results


# ------------------------------------------------------------
# Credits by company dialog
# ------------------------------------------------------------

def credits_checks(
    snapshot: UiSnapshot, expected: ExpectedDashboard
) -> list[CheckResult]:
    results: list[CheckResult] = []
    if expected.credits_by_company is None:
        results.append(
            not_applicable(
                CAT_CREDITS,
                "credits by company dialog",
                "user is not a super admin - the credits card is not shown",
            )
        )
        results.append(
            check(
                CAT_CREDITS,
                "'View by company' hidden for non-super-admin",
                not snapshot.credits_button_present,
                expected="hidden",
                actual="shown" if snapshot.credits_button_present else "hidden",
            )
        )
        return results

    if not expected.credits_by_company:
        results.append(
            check(
                CAT_CREDITS,
                "'View by company' hidden when the API returns no credits",
                not snapshot.credits_button_present,
                expected="hidden",
                actual="shown" if snapshot.credits_button_present else "hidden",
            )
        )
        return results

    results.append(
        check(
            CAT_CREDITS,
            "'View by company' button is displayed",
            snapshot.credits_button_present,
            expected="shown",
            actual="shown" if snapshot.credits_button_present else "hidden",
        )
    )
    rows = snapshot.credits_by_company or []
    results.append(
        check(
            CAT_CREDITS,
            "dialog row count equals API credits rows",
            len(rows) == len(expected.credits_by_company),
            expected=len(expected.credits_by_company),
            actual=len(rows),
        )
    )
    mismatches: list[str] = []
    for index, (name, value) in enumerate(expected.credits_by_company):
        if index >= len(rows):
            mismatches.append(f"#{index + 1} {name}={value}: row missing")
            continue
        ui_name, ui_value = rows[index]
        if _norm(ui_name) != _norm(name) or parse_number(ui_value) != value:
            mismatches.append(
                f"#{index + 1} expected {name}={value}, "
                f"shown {ui_name}={ui_value}"
            )
    results.append(
        check(
            CAT_CREDITS,
            "every company's remaining credits match the API (in API order)",
            not mismatches,
            expected=f"{len(expected.credits_by_company)} rows identical",
            actual=(
                "; ".join(mismatches[:10])
                + (" …" if len(mismatches) > 10 else "")
                if mismatches
                else f"{len(rows)} rows identical"
            ),
        )
    )
    return results


def all_ui_checks(
    snapshot: UiSnapshot, expected: ExpectedDashboard
) -> list[CheckResult]:
    return (
        kpi_checks(snapshot, expected)
        + stage_checks(snapshot, expected)
        + pipeline_checks(snapshot, expected)
        + donut_checks(snapshot, expected)
        + credits_checks(snapshot, expected)
    )


__all__ = [
    "all_ui_checks",
    "credits_checks",
    "donut_checks",
    "drift_checks",
    "kpi_checks",
    "pipeline_checks",
    "source_checks",
    "stage_checks",
]
