"""
Derive the values the dashboard page MUST display from the API
payload, using the same formulas as the web app's Dashboard
component (read from the deployed QA bundle, 2026-09-03):

  P = payload.pipeline (every field `|| 0`)

  total        = P.started            ("Total interviews" KPI)
  opened_total = P.opened + P.startedOnly + P.inProgress + P.paused
                 + P.completed + P.candidateExit
  invites      = opened_total + P.notOpened + P.expired
  completion   = min(100, round(P.completed / total * 100)), 0 if total == 0

  KPI "Active candidates"  = candidates.aggregate.count
        subtitle "across N companies"  (super admin, N = lookup companies)
  KPI "Active jobs"        = jobs.aggregate.count
  KPI "Total interviews"   = total, badge "<completion>%",
        subtitle "completion rate"
  KPI "Credits allocated"  = totalCreditsDisbursed.aggregate.sum
        .remaining_credits (super admin only),
        subtitle "<n> companies below 25" when n > 0
        (n = credits rows with remaining_credits < 25)

  Stage breakdown rows (label, value, pct = round(value / D * 100)),
  D = sum of the rows, or 1:
        Link not opened=notOpened, Link opened=opened, Started=startedOnly,
        In progress, Paused, Completed, Candidate exit, Invite expired=expired

  Interview pipeline bars (label shown when count > 0):
        Not opened, Opened, Started(=startedOnly), In progress, Paused,
        Completed, Candidate exit, Expired

  Donuts (center TOTAL = sum of the shown entries):
        Communications  [Email = invites]                    (when > 0)
        Open rate       [Opened = opened_total, Not opened, Expired]
        Response mix    [Completed, Candidate exit, Paused, In progress,
                         Started]
        (entries with value 0 are dropped)

  "View by company" dialog: credits rows in API order
        (company_name, remaining_credits)

Numbers render with toLocaleString() -> "3,635"; JS Math.round
rounds halves up, so js_round() is used instead of Python's round().
"""

import math
from dataclasses import dataclass, field
from typing import Any

from app_e2e.dashboard.schema import get_path, is_int


SUPER_ADMIN_ROLE = "admin"
LOW_CREDIT_THRESHOLD = 25


def js_round(value: float) -> int:
    """JavaScript Math.round semantics (halves toward +infinity)."""

    return int(math.floor(value + 0.5))


def fmt_int(value: int) -> str:
    """toLocaleString() for integers in an en locale."""

    return f"{int(value):,}"


def _count(payload: Any, path: str) -> int:
    """Mirror the UI's `?? 0` / `|| 0` fallbacks."""

    found, value = get_path(payload, path)
    return value if found and is_int(value) else 0


@dataclass(frozen=True)
class DashboardStats:
    total: int
    not_opened: int
    opened: int
    started: int
    in_progress: int
    paused: int
    expired: int
    completed: int
    candidate_exit: int

    @property
    def opened_total(self) -> int:
        return (
            self.opened
            + self.started
            + self.in_progress
            + self.paused
            + self.completed
            + self.candidate_exit
        )

    @property
    def invites(self) -> int:
        return self.opened_total + self.not_opened + self.expired

    @property
    def completion_rate(self) -> int:
        if self.total > 0:
            return min(100, js_round(self.completed / self.total * 100))
        return 0


def stats_from_payload(payload: Any) -> DashboardStats:
    return DashboardStats(
        total=_count(payload, "pipeline.started"),
        not_opened=_count(payload, "pipeline.notOpened"),
        opened=_count(payload, "pipeline.opened"),
        started=_count(payload, "pipeline.startedOnly"),
        in_progress=_count(payload, "pipeline.inProgress"),
        paused=_count(payload, "pipeline.paused"),
        expired=_count(payload, "pipeline.expired"),
        completed=_count(payload, "pipeline.completed"),
        candidate_exit=_count(payload, "pipeline.candidateExit"),
    )


@dataclass(frozen=True)
class ExpectedKpi:
    title: str
    value: int
    subtitle: str | None = None
    badge: str | None = None

    @property
    def key(self) -> str:
        return self.title.lower()


@dataclass
class ExpectedDashboard:
    stats: DashboardStats
    kpis: dict[str, ExpectedKpi]
    stage_rows: list[tuple[str, int, int]]
    stage_denominator: int
    pipeline_counts: dict[str, int]
    donuts: dict[str, tuple[int, list[tuple[str, int]]]]
    credits_by_company: list[tuple[str, int]] | None
    is_super_admin: bool
    companies_count: int
    low_credit_companies: int = 0
    notes: list[str] = field(default_factory=list)


def stage_rows_raw(stats: DashboardStats) -> list[tuple[str, int]]:
    return [
        ("Link not opened", stats.not_opened),
        ("Link opened", stats.opened),
        ("Started", stats.started),
        ("In progress", stats.in_progress),
        ("Paused", stats.paused),
        ("Completed", stats.completed),
        ("Candidate exit", stats.candidate_exit),
        ("Invite expired", stats.expired),
    ]


def stage_breakdown(
    stats: DashboardStats,
) -> tuple[list[tuple[str, int, int]], int]:
    rows = stage_rows_raw(stats)
    denominator = sum(v for _, v in rows) or 1
    return (
        [
            (label, value, js_round(value / denominator * 100))
            for label, value in rows
        ],
        denominator,
    )


def derive_expected(
    payload: Any,
    *,
    is_super_admin: bool,
    companies_count: int,
) -> ExpectedDashboard:
    stats = stats_from_payload(payload)

    credits_raw = payload.get("credits") if isinstance(payload, dict) else None
    credits = credits_raw if isinstance(credits_raw, list) else []

    low = sum(
        1
        for entry in credits
        if (
            (entry.get("remaining_credits") if isinstance(entry, dict) else 0)
            or 0
        )
        < LOW_CREDIT_THRESHOLD
    )

    kpis: dict[str, ExpectedKpi] = {}

    candidates = ExpectedKpi(
        title="Active candidates",
        value=_count(payload, "candidates.aggregate.count"),
        subtitle=(
            f"across {companies_count} companies"
            if is_super_admin and companies_count > 0
            else None
        ),
    )
    jobs = ExpectedKpi(
        title="Active jobs",
        value=_count(payload, "jobs.aggregate.count"),
    )
    interviews = ExpectedKpi(
        title="Total interviews",
        value=stats.total,
        subtitle="completion rate",
        badge=f"{stats.completion_rate}%",
    )
    for kpi in (candidates, jobs, interviews):
        kpis[kpi.key] = kpi

    credits_by_company: list[tuple[str, int]] | None = None
    if is_super_admin:
        allocated = _count(
            payload, "totalCreditsDisbursed.aggregate.sum.remaining_credits"
        )
        credits_kpi = ExpectedKpi(
            title="Credits allocated",
            value=allocated,
            subtitle=(
                f"{low} companies below {LOW_CREDIT_THRESHOLD}"
                if low > 0
                else None
            ),
        )
        kpis[credits_kpi.key] = credits_kpi
        credits_by_company = [
            (
                str(
                    (entry.get("company") or {}).get("company_name")
                    if isinstance(entry, dict)
                    else ""
                ),
                int(entry.get("remaining_credits") or 0)
                if isinstance(entry, dict)
                else 0,
            )
            for entry in credits
        ]

    pipeline_counts = {
        "Not opened": stats.not_opened,
        "Opened": stats.opened,
        "Started": stats.started,
        "In progress": stats.in_progress,
        "Paused": stats.paused,
        "Completed": stats.completed,
        "Candidate exit": stats.candidate_exit,
        "Expired": stats.expired,
    }

    def non_zero(rows: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return [(name, value) for name, value in rows if value > 0]

    donuts: dict[str, tuple[int, list[tuple[str, int]]]] = {}
    communications = non_zero([("Email", stats.invites)])
    donuts["Communications"] = (
        sum(v for _, v in communications), communications
    )
    open_rate = non_zero(
        [
            ("Opened", stats.opened_total),
            ("Not opened", stats.not_opened),
            ("Expired", stats.expired),
        ]
    )
    donuts["Open rate"] = (sum(v for _, v in open_rate), open_rate)
    response_mix = non_zero(
        [
            ("Completed", stats.completed),
            ("Candidate exit", stats.candidate_exit),
            ("Paused", stats.paused),
            ("In progress", stats.in_progress),
            ("Started", stats.started),
        ]
    )
    donuts["Response mix"] = (sum(v for _, v in response_mix), response_mix)

    rows, denominator = stage_breakdown(stats)

    return ExpectedDashboard(
        stats=stats,
        kpis=kpis,
        stage_rows=rows,
        stage_denominator=denominator,
        pipeline_counts=pipeline_counts,
        donuts=donuts,
        credits_by_company=credits_by_company,
        is_super_admin=is_super_admin,
        companies_count=companies_count,
        low_credit_companies=low,
    )
