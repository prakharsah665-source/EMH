"""
PASS / FAIL report for the Dashboard data validation.

Written at the end of every pytest session that ran tests/app_e2e:

  reports/dashboard_validation_report.html   self-contained, with the
                                             dashboard screenshot
  reports/dashboard_validation_results.json  machine-readable rows

The report is built from the CheckRegistry (every check row), the
environment summary and the per-test pytest outcomes.
"""

import base64
import html
import json
from datetime import datetime
from pathlib import Path


STYLE = """
:root {
  --bg: #f6f7f9; --card: #fff; --text: #1e2530; --muted: #5c6672;
  --border: #dde2e8; --pass: #1a7f4b; --pass-bg: #e4f4eb;
  --fail: #b3312d; --fail-bg: #fbe9e8; --na: #8a6d1a; --na-bg: #faf3dc;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  Helvetica, Arial, sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 34px 0 10px; }
.meta { color: var(--muted); font-size: .9rem; }
.overall { display: inline-block; margin: 12px 0 4px; padding: 6px 16px;
  border-radius: 8px; font-weight: 700; }
.overall.PASS { color: var(--pass); background: var(--pass-bg); }
.overall.FAIL { color: var(--fail); background: var(--fail-bg); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; margin: 18px 0 8px; }
.card { background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; }
.card .num { font-size: 1.7rem; font-weight: 700; }
.card .label { color: var(--muted); font-size: .8rem; text-transform: uppercase;
  letter-spacing: .04em; }
.card.PASS .num { color: var(--pass); } .card.FAIL .num { color: var(--fail); }
.card.NA .num { color: var(--na); }
table { width: 100%; border-collapse: collapse; background: var(--card);
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
  vertical-align: top; font-size: .9rem; }
th { background: #eef1f4; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted); }
tr:last-child td { border-bottom: none; }
td.val { font-family: var(--mono); font-size: .8rem; word-break: break-word; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: .76rem; font-weight: 600; white-space: nowrap; }
.badge.PASS { color: var(--pass); background: var(--pass-bg); }
.badge.FAIL { color: var(--fail); background: var(--fail-bg); }
.badge.N\\/A { color: var(--na); background: var(--na-bg); }
.detail { color: var(--muted); font-size: .8rem; }
.desc { color: var(--muted); font-size: .88rem; }
ul.notes { padding-left: 20px; } ul.notes li { margin: 3px 0; font-size: .9rem; }
img.shot { max-width: 100%; border: 1px solid var(--border); border-radius: 10px; }
details { margin: 8px 0; border: 1px solid var(--border); border-radius: 8px;
  background: #fafbfc; }
summary { cursor: pointer; padding: 6px 10px; font-size: .85rem; color: var(--muted); }
pre { margin: 0; padding: 10px 12px; font-family: var(--mono); font-size: .76rem;
  white-space: pre-wrap; word-break: break-word; max-height: 420px; overflow: auto; }
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_report(
    *,
    registry_results: list[dict],
    environment: dict,
    notes: list[str],
    test_outcomes: list[dict],
    screenshot_path: str | None,
    api_payload: dict | None,
    setup_errors: dict[str, str],
) -> dict:
    counts = {"PASS": 0, "FAIL": 0, "N/A": 0}
    for row in registry_results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    tests_failed = sum(
        1 for t in test_outcomes if t["outcome"] in ("failed", "error")
    )
    overall = (
        "FAIL"
        if counts["FAIL"] or tests_failed or setup_errors
        else "PASS"
    )
    screenshot_b64 = None
    if screenshot_path and Path(screenshot_path).exists():
        screenshot_b64 = base64.b64encode(
            Path(screenshot_path).read_bytes()
        ).decode("ascii")
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall": overall,
        "counts": {**counts, "total": len(registry_results)},
        "environment": environment,
        "notes": notes,
        "setup_errors": setup_errors,
        "checks": registry_results,
        "tests": test_outcomes,
        "screenshot_path": screenshot_path,
        "screenshot_b64": screenshot_b64,
        "api_payload": api_payload,
    }


def render_html(report: dict) -> str:
    counts = report["counts"]
    grouped: dict[str, list[dict]] = {}
    for row in report["checks"]:
        grouped.setdefault(row["category"], []).append(row)

    sections = []
    for category, rows in grouped.items():
        passed = sum(1 for r in rows if r["status"] == "PASS")
        failed = sum(1 for r in rows if r["status"] == "FAIL")
        body = "".join(
            f"<tr><td>{esc(r['name'])}"
            + (f"<div class='detail'>{esc(r['detail'])}</div>" if r["detail"] else "")
            + f"</td><td class='val'>{esc(r['expected'])}</td>"
            f"<td class='val'>{esc(r['actual'])}</td>"
            f"<td><span class='badge {esc(r['status'])}'>{esc(r['status'])}</span></td></tr>"
            for r in rows
        )
        sections.append(
            f"<h2>{esc(category)} <span class='desc'>- {passed} passed"
            + (f", {failed} FAILED" if failed else "")
            + f" of {len(rows)}</span></h2>"
            "<table><thead><tr><th>Check</th><th>Expected (API)</th>"
            "<th>Actual</th><th>Result</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    env_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='val'>{esc(v)}</td></tr>"
        for k, v in report["environment"].items()
    )
    notes = "".join(f"<li>{esc(n)}</li>" for n in report["notes"])
    setup = "".join(
        f"<li><b>{esc(stage)}</b>: {esc(message)}</li>"
        for stage, message in report["setup_errors"].items()
    )
    tests = "".join(
        f"<tr><td class='val'>{esc(t['nodeid'])}</td>"
        f"<td><span class='badge {'PASS' if t['outcome'] == 'passed' else 'N/A' if t['outcome'] == 'skipped' else 'FAIL'}'>"
        f"{esc(t['outcome'].upper())}</span></td>"
        f"<td class='val'>{t['duration']:.1f}s</td></tr>"
        for t in report["tests"]
    )
    shot = (
        f"<h2>Dashboard as rendered</h2><img class='shot' alt='dashboard screenshot' "
        f"src='data:image/png;base64,{report['screenshot_b64']}'>"
        if report.get("screenshot_b64")
        else ""
    )
    payload = (
        "<details><summary>API payload used for the expected values</summary>"
        f"<pre>{esc(json.dumps(report['api_payload'], indent=1)[:60000])}</pre></details>"
        if report.get("api_payload") is not None
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMH Dashboard Data Validation</title><style>{STYLE}</style></head>
<body><div class="wrap">
<h1>EMH Dashboard - Data Validation Report</h1>
<p class="meta">Generated {esc(report['generated'])}</p>
<span class="overall {esc(report['overall'])}">DASHBOARD VALIDATION {esc(report['overall'])}</span>
<div class="cards">
  <div class="card"><div class="num">{counts['total']}</div><div class="label">Checks</div></div>
  <div class="card PASS"><div class="num">{counts['PASS']}</div><div class="label">Passed</div></div>
  <div class="card FAIL"><div class="num">{counts['FAIL']}</div><div class="label">Failed</div></div>
  <div class="card NA"><div class="num">{counts['N/A']}</div><div class="label">Not applicable</div></div>
</div>
<h2>Environment</h2>
<table><tbody>{env_rows}</tbody></table>
{('<h2>Setup errors</h2><ul class="notes">' + setup + '</ul>') if setup else ''}
{('<h2>Run notes</h2><ul class="notes">' + notes + '</ul>') if notes else ''}
{''.join(sections)}
<h2>Test outcomes</h2>
<table><thead><tr><th>Test</th><th>Outcome</th><th>Time</th></tr></thead><tbody>{tests}</tbody></table>
{shot}
{payload}
<h2>How to read this report</h2>
<p class="desc">"Expected (API)" is the value the backend returned (or the number
derived from it with the dashboard's own formula); "Actual" is what the
dashboard page rendered, or what the API response contained for schema
checks. Any FAIL row is a data mismatch, a schema violation or a broken
step; N/A rows are checks that do not apply to this account/data.</p>
</div></body></html>"""


def write_reports(report: dict, html_path: Path, json_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(report), encoding="utf-8")
    slim = {k: v for k, v in report.items() if k != "screenshot_b64"}
    json_path.write_text(json.dumps(slim, indent=1), encoding="utf-8")
