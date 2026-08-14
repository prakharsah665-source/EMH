"""
Opt-in LIVE DeepTeam red-team of the EMH interviewer.

Authorized testing of our own app. This test is SKIPPED unless
all of the following hold, so it never runs external attacks in
normal/CI-lite runs and never breaks the py3.14 main venv:

  * DeepTeam is importable (needs the py3.13 security venv), and
  * EMH_LIVE_REDTEAM=1 is set, and
  * a target callback is configured via EMH_REDTEAM_TARGET as
    "module:function" resolving to a DeepTeam model_callback
    that drives the real (authorized) interviewer.

When enabled, it asserts every mapped vulnerability was
mitigated (DeepTeam score == 1). Import wiring and result
reduction are exercised offline by the checks below regardless
of whether the live run is enabled.
"""

import importlib
import os

import pytest

from evaluation.security_redteam import (
    DEEPTEAM_AVAILABLE,
    DEEPTEAM_IMPORT_ERROR,
    summarize_assessment,
)


def test_redteam_module_imports_without_deepteam():
    # The guarded module must import cleanly even where DeepTeam
    # is absent (the main py3.14 venv), otherwise the whole
    # suite would break. This always runs.
    if not DEEPTEAM_AVAILABLE:
        assert DEEPTEAM_IMPORT_ERROR, (
            "DeepTeam unavailable but no import error recorded."
        )


def test_summarize_assessment_flags_vulnerable_cases():
    # Offline check of the pass/fail reduction (score==1 safe).
    class _Case:
        def __init__(self, score):
            self.score = score
            self.vulnerability = "V"
            self.attack_method = "A"
            self.risk_category = "R"
            self.reason = "r"

    class _Assessment:
        test_cases = [_Case(1), _Case(0), _Case(1)]

    summary = summarize_assessment(_Assessment())
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert len(summary["vulnerable"]) == 1
    assert summary["all_mitigated"] is False


def _load_target_callback():
    spec = os.getenv("EMH_REDTEAM_TARGET")
    if not spec or ":" not in spec:
        pytest.skip(
            "EMH_REDTEAM_TARGET not set to 'module:function' - no "
            "authorized live target configured."
        )
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


@pytest.mark.security_live
def test_live_red_team_all_vulnerabilities_mitigated():
    if os.getenv("EMH_LIVE_REDTEAM") != "1":
        pytest.skip(
            "Live red-team disabled. Set EMH_LIVE_REDTEAM=1 (and "
            "EMH_REDTEAM_TARGET) in the py3.13 security venv to run."
        )
    if not DEEPTEAM_AVAILABLE:
        pytest.skip(
            f"DeepTeam unavailable: {DEEPTEAM_IMPORT_ERROR}. Use "
            "the py3.13 security venv (requirements-security.txt)."
        )

    from evaluation.security_redteam import run_red_team

    target = _load_target_callback()
    assessment = run_red_team(target)
    summary = summarize_assessment(assessment)

    print(
        f"Red-team: {summary['passed']}/{summary['total']} "
        "attacks mitigated."
    )
    for vulnerable in summary["vulnerable"]:
        print(f"  VULNERABLE: {vulnerable}")

    assert summary["all_mitigated"], (
        "The interviewer was vulnerable to at least one attack: "
        f"{summary['vulnerable']}"
    )
