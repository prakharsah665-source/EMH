#!/usr/bin/env python3
"""
Run ONLY the application E2E Dashboard data validation and print its
PASS/FAIL summary.

Usage:

    python scripts/run_dashboard_validation.py [--headed] [pytest args]

Outputs:

    reports/dashboard_validation_report.html   PASS/FAIL report with
                                               every check + screenshot
    reports/dashboard_validation_results.json  machine-readable rows
    reports/dashboard_pytest_report.html       pytest-html report
    reports/junit_dashboard.xml                JUnit XML

Independent of scripts/run_all_tests.py: no interview session
pre-flight, no interview URLs needed. Configuration comes from .env
(LOGIN_URL, LOGIN_EMAIL, LOGIN_PASSWORD, Dashboard_API_URL,
ACCESS_TOKEN) - see docs/dashboard_validation.md.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
RESULTS_JSON = REPORT_DIR / "dashboard_validation_results.json"
REPORT_HTML = REPORT_DIR / "dashboard_validation_report.html"


def run_pytest(extra_args: list[str]) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        str(ROOT / "tests" / "app_e2e"),
        "--junitxml",
        str(REPORT_DIR / "junit_dashboard.xml"),
        "-o",
        "junit_logging=all",
        "-o",
        "junit_family=xunit2",
        "--html",
        str(REPORT_DIR / "dashboard_pytest_report.html"),
        "--self-contained-html",
        *extra_args,
    ]
    print("Running Dashboard data validation...")
    print(" ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


def print_summary(results: dict) -> None:
    counts = results["counts"]
    print()
    print("=" * 64)
    print(f"DASHBOARD DATA VALIDATION: {results['overall']}")
    print("=" * 64)
    print(
        f"{counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['N/A']} n/a ({counts['total']} checks)"
    )

    by_category: dict[str, dict[str, int]] = {}
    for row in results["checks"]:
        bucket = by_category.setdefault(
            row["category"], {"PASS": 0, "FAIL": 0, "N/A": 0}
        )
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1

    print()
    print(f"{'Category':<36} {'PASS':>5} {'FAIL':>5} {'N/A':>5}")
    for category, bucket in by_category.items():
        print(
            f"{category:<36} {bucket['PASS']:>5} {bucket['FAIL']:>5} "
            f"{bucket['N/A']:>5}"
        )

    failed = [r for r in results["checks"] if r["status"] == "FAIL"]
    if failed:
        print()
        print("FAILED CHECKS:")
        for row in failed:
            print(f"  - {row['category']}: {row['name']}")
            print(f"      expected: {row['expected']}")
            print(f"      actual:   {row['actual']}")
            if row.get("detail"):
                print(f"      note:     {row['detail']}")

    if results.get("setup_errors"):
        print()
        print("SETUP ERRORS:")
        for stage, message in results["setup_errors"].items():
            print(f"  - {stage}: {message}")

    print()
    print(f"HTML report: {REPORT_HTML}")
    print(f"JSON results: {RESULTS_JSON}")


def main() -> int:
    args = sys.argv[1:]
    if "--headed" in args:
        args.remove("--headed")
        os.environ["DASHBOARD_HEADLESS"] = "0"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_JSON.exists():
        RESULTS_JSON.unlink()

    exit_code = run_pytest(args)

    if not RESULTS_JSON.exists():
        print(
            "ERROR: no dashboard_validation_results.json was produced - "
            "pytest did not reach the session fixture (collection error?)."
        )
        return exit_code or 1

    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    print_summary(results)

    return 0 if exit_code == 0 and results["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
