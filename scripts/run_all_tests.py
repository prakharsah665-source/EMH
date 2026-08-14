#!/usr/bin/env python3
"""
Run the full EMH test suite and generate a detailed,
self-contained HTML report.

Usage:

    python scripts/run_all_tests.py [extra pytest args]

The report is written to:

    reports/emh_full_test_report.html

How it works:

1. pytest runs with output capturing ENABLED (no -s) so
   every test's stdout - GEval scores, judge reasoning,
   page-state dumps - is captured per test.
2. pytest writes machine-readable results to a JUnit XML
   file (junit_logging=all embeds captured stdout/stderr).
3. This script parses the XML and renders one
   self-contained HTML file with:
       - run summary and environment
       - per-suite breakdown
       - per-test status, duration, captured output,
         and full failure tracebacks
"""

from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree
import html
import os
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
REPORT_FILE = REPORT_DIR / "emh_full_test_report.html"
JUNIT_FILE = REPORT_DIR / "junit_results.xml"


# ============================================================
# Suite descriptions
#
# Shown in the report so a reader understands what each
# group of tests actually verifies.
# ============================================================

SUITES = {
    "tests.e2e": (
        "End-to-End (Browser)",
        "Playwright drives the real EMH interview page: "
        "media permissions, system configuration, audio "
        "setup, recording consent, interview room, "
        "socket.io and LiveKit connectivity.",
    ),
    "tests.api": (
        "API / Interview Events",
        "LLM-judged checks that the AI interviewer handles "
        "interview events (start, follow-ups, "
        "clarifications, off-topic answers, completion) "
        "appropriately. Judge: NVIDIA Nemotron, "
        "temperature=0.",
    ),
    "tests.ai_quality": (
        "AI Quality",
        "LLM-judged quality checks for individual "
        "interviewer behaviors: relevance, adaptability, "
        "professionalism, technical accuracy, context "
        "retention, question repetition and more. "
        "Judge: NVIDIA Nemotron, temperature=0.",
    ),
    "tests.evaluation": (
        "Interview Quality Gate (CI)",
        "Full-transcript rubric evaluation via NVIDIA "
        "Nemotron with structured JSON output, discrete "
        "scores (0, 0.25, 0.5, 0.75, 1.0), 3-run median "
        "scoring and per-criterion CI thresholds.",
    ),
    "tests.audio_eval": (
        "Audio Metrics (Local)",
        "Deterministic jiwer WER and pyannote.metrics "
        "diarization checks over local fixtures and, when "
        "present, real captured audio-turn artifacts. "
        "No network calls.",
    ),
    "tests.collectors": (
        "Transcript Capture (Local)",
        "Unit tests for the transcript collector that turns "
        "E2E page observations into "
        "artifacts/transcripts/actual_transcript.json.",
    ),
    "tests.security": (
        "Security (Defensive)",
        "Prompt-injection, instruction-override, system-prompt "
        "disclosure, PII/secret-leakage and malicious-input "
        "detectors, unit-tested offline and used to audit the "
        "real captured transcript. Plus an opt-in DeepTeam live "
        "red-team (py3.13 security venv).",
    ),
}


# ============================================================
# Test execution
# ============================================================

def ordered_test_paths() -> list[str]:
    """
    Dependency-ordered test paths for a full run.

    pytest's default alphabetical order is wrong for this
    suite in two ways:

    1. The LLM-judged suites (ai_quality/evaluation/security
       audit) and the real-artifact audio tests consume the
       transcript captured by tests/e2e/test_bot_responsiveness
       - alphabetically they run BEFORE it, silently judging
       the PREVIOUS run's stale capture.
    2. test_bot_responsiveness drives the interview to
       completion, consuming the one-shot session -
       alphabetically it runs second, so every later E2E test
       hits an already-finished interview and fails for
       session reasons, not product reasons.

    So: the full-interview evaluation FIRST - it is the only
    consumer that needs a fresh, never-joined interview room
    (the agent greeting fires on room join), and it produces
    the capture everything else judges. Setup-screen tests
    then re-enter the same session as intentional
    continuation (they never join the room). Room-joining
    tests (continue_to_interview / interview_room / livekit)
    run last within E2E and skip with SESSION ALREADY
    CONSUMED unless EMH_ROOM_TESTS_URL provides an isolated
    second session. Judging suites come after the capture.
    """

    capture = ROOT / "tests/e2e/test_bot_responsiveness.py"

    room_joiners = [
        ROOT / "tests/e2e/test_continue_to_interview.py",
        ROOT / "tests/e2e/test_interview_room.py",
        ROOT / "tests/e2e/test_livekit_connection.py",
    ]

    setup_screen_tests = sorted(
        path
        for path in (ROOT / "tests/e2e").glob("test_*.py")
        if path != capture and path not in room_joiners
    )

    ordered = [
        capture,
        *setup_screen_tests,
        *room_joiners,
        ROOT / "tests/config",
        ROOT / "tests/collectors",
        ROOT / "tests/audio_eval",
        ROOT / "tests/security",
        ROOT / "tests/ai_quality",
        ROOT / "tests/api",
        ROOT / "tests/evaluation",
    ]

    return [str(path) for path in ordered if path.exists()]


def run_pytest(extra_args: list[str]) -> int:
    """
    Run pytest with capturing enabled and JUnit output.

    Note: -s must NOT be used here. It disables output
    capturing, which would strip all per-test detail
    (scores, judge reasoning, page dumps) from the report.
    """

    # Only impose the dependency ordering when the caller did
    # not select specific tests/paths themselves.
    has_path_selection = any(
        not arg.startswith("-") for arg in extra_args
    )

    ordered_paths = (
        [] if has_path_selection else ordered_test_paths()
    )

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--junitxml",
        str(JUNIT_FILE),
        "-o",
        "junit_logging=all",
        "-o",
        "junit_family=xunit2",
        # pytest-html self-contained report (kept alongside the
        # custom HTML report built from the JUnit XML below).
        "--html",
        str(REPORT_DIR / "emh_pytest_report.html"),
        "--self-contained-html",
        *extra_args,
        *ordered_paths,
    ]

    print("Running full EMH test suite...")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=ROOT,
    )

    return result.returncode


# ============================================================
# JUnit XML parsing
# ============================================================

def parse_results(junit_path: Path) -> dict:
    """
    Parse the JUnit XML into a plain dict structure.
    """

    tree = ElementTree.parse(junit_path)

    tests = []

    totals = {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "duration": 0.0,
    }

    for suite in tree.iter("testsuite"):

        totals["duration"] += float(
            suite.get("time", 0)
        )

        for case in suite.iter("testcase"):

            classname = case.get("classname", "")
            name = case.get("name", "")
            duration = float(case.get("time", 0))

            outcome = "passed"
            detail = ""

            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")

            if failure is not None:
                outcome = "failed"
                detail = (
                    (failure.get("message") or "")
                    + "\n\n"
                    + (failure.text or "")
                )

            elif error is not None:
                outcome = "error"
                detail = (
                    (error.get("message") or "")
                    + "\n\n"
                    + (error.text or "")
                )

            elif skipped is not None:
                outcome = "skipped"
                detail = skipped.get("message") or ""

            totals[
                outcome if outcome in totals else "error"
            ] += 1

            stdout_node = case.find("system-out")
            stderr_node = case.find("system-err")

            tests.append(
                {
                    "classname": classname,
                    "name": name,
                    "duration": duration,
                    "outcome": outcome,
                    "detail": detail.strip(),
                    "stdout": (
                        stdout_node.text or ""
                    ).strip()
                    if stdout_node is not None
                    else "",
                    "stderr": (
                        stderr_node.text or ""
                    ).strip()
                    if stderr_node is not None
                    else "",
                }
            )

    totals["total"] = len(tests)

    return {
        "tests": tests,
        "totals": totals,
    }


def suite_key(classname: str) -> str:
    """
    Map a JUnit classname (tests.e2e.test_x) to its suite.
    """

    for key in SUITES:
        if classname.startswith(key):
            return key

    return "other"


# ============================================================
# HTML rendering
# ============================================================

STYLE = """
:root {
    --bg: #f6f7f9;
    --card: #ffffff;
    --text: #1e2530;
    --muted: #5c6672;
    --border: #dde2e8;
    --pass: #1a7f4b;
    --pass-bg: #e4f4eb;
    --fail: #b3312d;
    --fail-bg: #fbe9e8;
    --skip: #8a6d1a;
    --skip-bg: #faf3dc;
    --mono: ui-monospace, SFMono-Regular, Menlo,
            Consolas, monospace;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont,
          "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.wrap {
    max-width: 1080px;
    margin: 0 auto;
    padding: 32px 20px 64px;
}

h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 36px 0 12px; }

.meta { color: var(--muted); font-size: 0.9rem; }

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 20px 0 8px;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}

.card .num { font-size: 1.7rem; font-weight: 700; }
.card .label {
    color: var(--muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.card.pass .num { color: var(--pass); }
.card.fail .num { color: var(--fail); }
.card.skip .num { color: var(--skip); }

table {
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

th, td {
    text-align: left;
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

th {
    background: #eef1f4;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
}

tr:last-child td { border-bottom: none; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge.passed { color: var(--pass); background: var(--pass-bg); }
.badge.failed, .badge.error {
    color: var(--fail); background: var(--fail-bg);
}
.badge.skipped { color: var(--skip); background: var(--skip-bg); }

.suite-desc { color: var(--muted); font-size: 0.88rem; }

details {
    margin: 6px 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: #fafbfc;
}

details summary {
    cursor: pointer;
    padding: 6px 10px;
    font-size: 0.85rem;
    color: var(--muted);
    user-select: none;
}

details[open] summary { border-bottom: 1px solid var(--border); }

pre {
    margin: 0;
    padding: 10px 12px;
    font-family: var(--mono);
    font-size: 0.78rem;
    line-height: 1.5;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 480px;
    overflow-y: auto;
}

pre.failure { color: var(--fail); }

.testname { font-family: var(--mono); font-size: 0.85rem; }
.duration { color: var(--muted); white-space: nowrap; }

.overall {
    display: inline-block;
    margin-top: 10px;
    padding: 6px 16px;
    border-radius: 8px;
    font-weight: 700;
}

.overall.pass { color: var(--pass); background: var(--pass-bg); }
.overall.fail { color: var(--fail); background: var(--fail-bg); }
"""


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_detail_block(
    title: str,
    content: str,
    css_class: str = "",
    open_by_default: bool = False,
) -> str:

    if not content:
        return ""

    open_attr = " open" if open_by_default else ""

    return (
        f"<details{open_attr}>"
        f"<summary>{esc(title)}</summary>"
        f'<pre class="{css_class}">{esc(content)}</pre>'
        f"</details>"
    )


def render_test_row(test: dict) -> str:

    blocks = []

    blocks.append(
        render_detail_block(
            "Failure detail",
            test["detail"],
            css_class="failure",
            open_by_default=True,
        )
    )

    blocks.append(
        render_detail_block(
            "Captured output",
            test["stdout"],
            open_by_default=(
                test["outcome"] != "passed"
            ),
        )
    )

    blocks.append(
        render_detail_block(
            "Captured stderr",
            test["stderr"],
        )
    )

    detail_html = "".join(blocks) or (
        '<span class="suite-desc">No captured output.</span>'
    )

    return f"""
<tr>
  <td>
    <span class="testname">{esc(test["name"])}</span>
    {detail_html}
  </td>
  <td><span class="badge {test["outcome"]}">
      {test["outcome"].upper()}</span></td>
  <td class="duration">{test["duration"]:.1f}s</td>
</tr>"""


def render_suite_section(
    key: str,
    tests: list[dict],
) -> str:

    title, description = SUITES.get(
        key,
        (key, ""),
    )

    passed = sum(
        1 for t in tests if t["outcome"] == "passed"
    )

    rows = "".join(
        render_test_row(test) for test in tests
    )

    return f"""
<h2>{esc(title)}
    <span class="suite-desc">
        - {passed}/{len(tests)} passed</span></h2>
<p class="suite-desc">{esc(description)}</p>
<table>
  <thead>
    <tr><th>Test</th><th>Result</th><th>Time</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


def render_report(results: dict) -> str:

    totals = results["totals"]
    tests = results["tests"]

    generated = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    failed = totals["failed"] + totals["error"]

    overall_pass = failed == 0

    overall = (
        '<span class="overall pass">'
        "ALL TESTS PASSED</span>"
        if overall_pass
        else f'<span class="overall fail">'
        f"{failed} TEST(S) FAILED</span>"
    )

    # Group tests by suite, preserving SUITES order.
    grouped: dict[str, list[dict]] = {}

    for test in tests:
        grouped.setdefault(
            suite_key(test["classname"]),
            [],
        ).append(test)

    ordered_keys = [
        key for key in SUITES if key in grouped
    ] + [
        key for key in grouped if key not in SUITES
    ]

    sections = "".join(
        render_suite_section(key, grouped[key])
        for key in ordered_keys
    )

    minutes, seconds = divmod(
        int(totals["duration"]),
        60,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>EMH Interview Test Report</title>
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">

<h1>EMH AI Interview - Full Test Report</h1>
<p class="meta">
Generated {esc(generated)}
&nbsp;|&nbsp; Python {platform.python_version()}
on {esc(platform.platform())}
&nbsp;|&nbsp; Total wall time
{minutes}m {seconds:02d}s
</p>
{overall}

<div class="cards">
  <div class="card">
    <div class="num">{totals["total"]}</div>
    <div class="label">Total</div>
  </div>
  <div class="card pass">
    <div class="num">{totals["passed"]}</div>
    <div class="label">Passed</div>
  </div>
  <div class="card fail">
    <div class="num">{failed}</div>
    <div class="label">Failed</div>
  </div>
  <div class="card skip">
    <div class="num">{totals["skipped"]}</div>
    <div class="label">Skipped</div>
  </div>
</div>

{sections}

<h2>How to read this report</h2>
<p class="suite-desc">
Every test row can be expanded: "Captured output" holds the
full per-test stdout (LLM judge scores and reasoning, page
state dumps, connection logs); failed tests additionally
show the assertion or traceback under "Failure detail".
LLM-judged suites use NVIDIA Nemotron at temperature 0 with
structured JSON output; the CI quality gate scores the
transcript three times and applies the median per criterion,
so a single inconsistent judgment cannot fail the build.
</p>

</div>
</body>
</html>"""


# ============================================================
# Main
# ============================================================

def main() -> int:

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # One fresh interview session per complete run: when a
    # provisioning endpoint is configured (EMH_PROVISION_API_URL,
    # see config/session_provisioner.py), mint the session here
    # and hand it to every test via EMH_INTERVIEW_URL. Without
    # one, the manually pasted INTERVIEW_URL/EMH_INTERVIEW_URL
    # is used unchanged.
    sys.path.insert(0, str(ROOT))
    from config.session_provisioner import provision_interview_url

    provisioned = provision_interview_url()
    if provisioned:
        os.environ["EMH_INTERVIEW_URL"] = provisioned
        print(
            "Provisioned a fresh interview session for this "
            "run (URL withheld - it contains the JWT)."
        )

    exit_code = run_pytest(sys.argv[1:])

    if not JUNIT_FILE.exists():
        print(
            "ERROR: pytest produced no JUnit XML - "
            "cannot build the HTML report."
        )
        return exit_code or 1

    results = parse_results(JUNIT_FILE)

    REPORT_FILE.write_text(
        render_report(results),
        encoding="utf-8",
    )

    totals = results["totals"]

    print()
    print(
        f"Results: {totals['passed']} passed, "
        f"{totals['failed'] + totals['error']} failed, "
        f"{totals['skipped']} skipped "
        f"({totals['total']} total)"
    )
    print(f"HTML report: {REPORT_FILE}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
