"""
Schema and value validation of the getDashboardData payload.

Observed shape (QA, deployed 2026-09-03):

  candidates            {"aggregate": {"count": int}}
  jobs                  {"aggregate": {"count": int}}
  totalInterviews       {"aggregate": {"count": int}}
  credits               [{"remaining_credits": int,
                          "company": {"company_id": int,
                                      "company_name": str}}, ...]
  totalCreditsDisbursed {"aggregate": {"sum": {"remaining_credits": int}}}
  pipeline              {"notOpened": int, "opened": int, "started": int,
                         "startedOnly": int, "inProgress": int,
                         "paused": int, "expired": int, "completed": int,
                         "candidateExit": int}

Four groups of checks, each producing CheckResult rows:
  required fields, data types, value ranges, cross-field consistency.
"""

import re
from typing import Any

from app_e2e.dashboard.checks import CheckResult, check


CAT_FIELDS = "API required fields"
CAT_TYPES = "API data types"
CAT_VALUES = "API values"
CAT_CONSISTENCY = "API consistency"

COUNT_FIELDS = ("candidates", "jobs", "totalInterviews")

PIPELINE_FIELDS: tuple[str, ...] = (
    "notOpened",
    "opened",
    "started",
    "startedOnly",
    "inProgress",
    "paused",
    "expired",
    "completed",
    "candidateExit",
)

CREDITS_SUM_PATH = "totalCreditsDisbursed.aggregate.sum.remaining_credits"

REQUIRED_TOP_LEVEL = (
    *COUNT_FIELDS,
    "credits",
    "totalCreditsDisbursed",
    "pipeline",
)

INTEGER_PATHS = (
    *(f"{field}.aggregate.count" for field in COUNT_FIELDS),
    *(f"pipeline.{field}" for field in PIPELINE_FIELDS),
)


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def get_path(obj: Any, path: str) -> tuple[bool, Any]:
    """
    Resolve "a.b[0].c" against nested dicts/lists.
    Returns (found, value).
    """

    current = obj
    for match in _PATH_TOKEN.finditer(path):
        key, index = match.group(1), match.group(2)
        if key is not None:
            if not isinstance(current, dict) or key not in current:
                return False, None
            current = current[key]
        else:
            position = int(index)
            if not isinstance(current, list) or position >= len(current):
                return False, None
            current = current[position]
    return True, current


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    return type(value).__name__


def _list_or_more(items: list[str], limit: int = 8) -> str:
    return ", ".join(items[:limit]) + (" …" if len(items) > limit else "")


# ------------------------------------------------------------
# Required fields
# ------------------------------------------------------------

def validate_required_fields(payload: Any) -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(
        check(
            CAT_FIELDS,
            "payload is a JSON object",
            isinstance(payload, dict),
            expected="object",
            actual=type_name(payload),
        )
    )
    if not isinstance(payload, dict):
        return results

    missing = [key for key in REQUIRED_TOP_LEVEL if key not in payload]
    results.append(
        check(
            CAT_FIELDS,
            "all top-level fields present",
            not missing,
            expected=", ".join(REQUIRED_TOP_LEVEL),
            actual=(
                "missing: " + ", ".join(missing) if missing else "all present"
            ),
        )
    )

    for path in (*INTEGER_PATHS, CREDITS_SUM_PATH):
        found, _ = get_path(payload, path)
        results.append(
            check(
                CAT_FIELDS,
                f"{path} present",
                found,
                expected="present",
                actual="present" if found else "MISSING",
            )
        )

    credits = payload.get("credits")
    if isinstance(credits, list):
        bad = []
        for index, entry in enumerate(credits):
            for sub in (
                "remaining_credits",
                "company.company_id",
                "company.company_name",
            ):
                found, _ = get_path(entry, sub)
                if not found:
                    bad.append(f"credits[{index}].{sub}")
        results.append(
            check(
                CAT_FIELDS,
                "credits[*] entries carry remaining_credits, "
                "company.company_id, company.company_name",
                not bad,
                expected=f"{len(credits)} complete entries",
                actual=(
                    "missing: " + _list_or_more(bad)
                    if bad
                    else f"{len(credits)} complete entries"
                ),
            )
        )

    return results


# ------------------------------------------------------------
# Data types
# ------------------------------------------------------------

def validate_data_types(payload: Any) -> list[CheckResult]:
    results: list[CheckResult] = []
    if not isinstance(payload, dict):
        return results

    def typed(path: str, predicate, expected: str) -> None:
        found, value = get_path(payload, path)
        results.append(
            check(
                CAT_TYPES,
                f"{path} is {expected}",
                found and predicate(value),
                expected=expected,
                actual=type_name(value) if found else "MISSING",
            )
        )

    typed("pipeline", lambda v: isinstance(v, dict), "object")
    for path in INTEGER_PATHS:
        typed(path, is_int, "integer")

    typed(
        CREDITS_SUM_PATH,
        lambda v: v is None or is_int(v),
        "integer (or null when no credits)",
    )
    typed("credits", lambda v: isinstance(v, list), "array")

    credits = payload.get("credits")
    if isinstance(credits, list):
        wrong: list[str] = []
        for index, entry in enumerate(credits):
            _, remaining = get_path(entry, "remaining_credits")
            _, company_id = get_path(entry, "company.company_id")
            _, company_name = get_path(entry, "company.company_name")
            if not is_int(remaining):
                wrong.append(
                    f"[{index}].remaining_credits={type_name(remaining)}"
                )
            if not is_int(company_id):
                wrong.append(
                    f"[{index}].company.company_id={type_name(company_id)}"
                )
            if not isinstance(company_name, str):
                wrong.append(
                    f"[{index}].company.company_name="
                    f"{type_name(company_name)}"
                )
        results.append(
            check(
                CAT_TYPES,
                "credits[*] remaining_credits:int, company_id:int, "
                "company_name:string",
                not wrong,
                expected=f"{len(credits)} well-typed entries",
                actual=(
                    "wrong: " + _list_or_more(wrong)
                    if wrong
                    else f"{len(credits)} well-typed entries"
                ),
            )
        )

    return results


# ------------------------------------------------------------
# Values
# ------------------------------------------------------------

def validate_values(payload: Any) -> list[CheckResult]:
    results: list[CheckResult] = []
    if not isinstance(payload, dict):
        return results

    for path in INTEGER_PATHS:
        found, value = get_path(payload, path)
        results.append(
            check(
                CAT_VALUES,
                f"{path} >= 0",
                found and is_int(value) and value >= 0,
                expected=">= 0",
                actual=value if found else "MISSING",
            )
        )

    pipeline = payload.get("pipeline")
    if isinstance(pipeline, dict):
        unknown = sorted(set(pipeline) - set(PIPELINE_FIELDS))
        results.append(
            check(
                CAT_VALUES,
                "pipeline has no unknown stage fields "
                "(the dashboard would silently ignore them)",
                not unknown,
                expected=", ".join(PIPELINE_FIELDS),
                actual=(
                    "unknown: " + ", ".join(unknown) if unknown
                    else "only known fields"
                ),
            )
        )

    credits = payload.get("credits")
    if isinstance(credits, list):
        negative = [
            entry for entry in credits
            if isinstance(entry, dict)
            and is_int(entry.get("remaining_credits"))
            and entry["remaining_credits"] < 0
        ]
        results.append(
            check(
                CAT_VALUES,
                "credits[*].remaining_credits >= 0",
                not negative,
                expected=">= 0 for all",
                actual=(
                    f"{len(negative)} negative: "
                    + ", ".join(
                        f"{e['company'].get('company_name')}="
                        f"{e['remaining_credits']}"
                        for e in negative[:5]
                        if isinstance(e.get("company"), dict)
                    )
                    if negative
                    else f"{len(credits)} entries >= 0"
                ),
            )
        )

        blank = [
            index for index, entry in enumerate(credits)
            if not (
                isinstance(entry, dict)
                and isinstance(entry.get("company"), dict)
                and isinstance(entry["company"].get("company_name"), str)
                and entry["company"]["company_name"].strip()
            )
        ]
        results.append(
            check(
                CAT_VALUES,
                "credits[*].company.company_name non-empty",
                not blank,
                expected="non-empty for all",
                actual=(
                    f"blank at indexes {blank[:8]}" if blank
                    else f"{len(credits)} named"
                ),
            )
        )

        ids = [
            entry["company"].get("company_id")
            for entry in credits
            if isinstance(entry, dict) and isinstance(entry.get("company"), dict)
        ]
        duplicates = sorted(
            {cid for cid in ids if ids.count(cid) > 1}, key=str
        )
        results.append(
            check(
                CAT_VALUES,
                "credits[*].company.company_id unique",
                not duplicates,
                expected="no duplicate company ids",
                actual=(
                    f"duplicates: {duplicates}" if duplicates
                    else f"{len(set(ids))} unique ids"
                ),
            )
        )

    return results


# ------------------------------------------------------------
# Cross-field consistency
# ------------------------------------------------------------

def validate_consistency(payload: Any) -> list[CheckResult]:
    results: list[CheckResult] = []
    if not isinstance(payload, dict):
        return results

    credits = payload.get("credits")
    _, reported_sum = get_path(payload, CREDITS_SUM_PATH)
    if isinstance(credits, list) and all(
        isinstance(e, dict) and is_int(e.get("remaining_credits"))
        for e in credits
    ):
        computed = sum(e["remaining_credits"] for e in credits)
        reported = reported_sum if is_int(reported_sum) else 0
        results.append(
            check(
                CAT_CONSISTENCY,
                "totalCreditsDisbursed equals the sum of credits[*]"
                ".remaining_credits",
                computed == reported,
                expected=computed,
                actual=reported,
                detail=f"{len(credits)} company credit rows",
            )
        )

    pipeline = payload.get("pipeline")
    _, total = get_path(payload, "totalInterviews.aggregate.count")
    if isinstance(pipeline, dict) and all(
        is_int(pipeline.get(f)) for f in PIPELINE_FIELDS
    ) and is_int(total):
        started = pipeline["started"]
        completed = pipeline["completed"]
        results.append(
            check(
                CAT_CONSISTENCY,
                "pipeline.started equals totalInterviews.aggregate.count "
                "(the 'Total interviews' KPI shows pipeline.started)",
                started == total,
                expected=total,
                actual=started,
                detail=(
                    "" if started == total else
                    "The API reports two different totals for interviews; "
                    "the dashboard KPI uses pipeline.started."
                ),
            )
        )
        results.append(
            check(
                CAT_CONSISTENCY,
                "pipeline.completed <= pipeline.started "
                "(completion rate cannot exceed 100%)",
                completed <= started,
                expected=f"<= {started}",
                actual=completed,
                detail=(
                    "" if completed <= started else
                    "The dashboard clamps the completion badge to 100%, "
                    "hiding the inconsistency."
                ),
            )
        )
        exclusive = {
            f: pipeline[f]
            for f in ("opened", "startedOnly", "inProgress", "paused",
                      "completed", "candidateExit")
        }
        for name, value in exclusive.items():
            results.append(
                check(
                    CAT_CONSISTENCY,
                    f"pipeline.{name} <= pipeline.started",
                    value <= started,
                    expected=f"<= {started}",
                    actual=value,
                )
            )

    return results


def validate_payload(payload: Any) -> list[CheckResult]:
    """All four groups, in report order."""

    return (
        validate_required_fields(payload)
        + validate_data_types(payload)
        + validate_values(payload)
        + validate_consistency(payload)
    )
