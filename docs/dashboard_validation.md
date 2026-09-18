# Dashboard data validation (application E2E)

Automated PASS/FAIL validation of the recruiter/admin **Dashboard** page
against the backend `dashboard-data` API. It lives in its own module
(`app_e2e/`, `tests/app_e2e/`) and is fully independent of the interview
evaluation flow: no interview session, no one-tab lock, no transcript.

## Run

```bash
# Dashboard validation only (no interview URLs needed)
python scripts/run_dashboard_validation.py            # headless
python scripts/run_dashboard_validation.py --headed   # watch the browser

# or plain pytest
python -m pytest tests/app_e2e -v
```

The full suite (`scripts/run_all_tests.py`) also runs `tests/app_e2e`
last, after every interview suite, and lists it as its own section.

Outputs:

| File | Content |
|---|---|
| `reports/dashboard_validation_report.html` | PASS/FAIL report: every check with expected (API) vs actual (UI) value, environment, run notes, per-test outcomes, full-page screenshot |
| `reports/dashboard_validation_results.json` | the same rows, machine-readable |
| `artifacts/dashboard/dashboard_validation.png` | screenshot evidence |

## Configuration (.env only, nothing hardcoded)

| Variable | Meaning |
|---|---|
| `LOGIN_URL` | dashboard web-app login page (its origin serves `/dashboard`) |
| `LOGIN_EMAIL`, `LOGIN_PASSWORD` | dashboard user credentials |
| `Dashboard_API_URL` | dashboard-data API base, e.g. `https://<api-host>/dashboard-data` |
| `ACCESS_TOKEN` | optional pre-issued idToken |
| `LOGIN_API_URL` | optional; default `<Dashboard_API_URL origin>/auth/sign-in` |
| `DASHBOARD_URL` | optional; default `<LOGIN_URL origin>/dashboard` |
| `DASHBOARD_HEADLESS` | `0` to watch the browser |
| `DASHBOARD_TIMEOUT_MS` | per-step browser timeout (default 60000) |
| `DASHBOARD_DRIFT_TOLERANCE` | allowed count difference between the page's payload and the harness's own API read (default 0) |

**Token handling.** `ACCESS_TOKEN` is used only while it is an unexpired
JWT *and* the API accepts it. Dashboard tokens live one hour and are
environment-specific (a token minted by the production-facing dashboard
host is rejected by the QA API as `Invalid token`), so in practice the
harness signs in with `LOGIN_EMAIL` / `LOGIN_PASSWORD` through the same
`POST /auth/sign-in` the login form uses and records why in the report.

**Environment pairing.** `LOGIN_URL` must be the web-app build that talks
to the host in `Dashboard_API_URL` (QA app -> QA API). The "page fetches
from the configured Dashboard_API_URL" check fails loudly if they are
paired wrongly.

## What is validated

API side (independent read, same request body the page sends):

1. Authentication: token accepted, JSON object, latency, `GetLookup` too.
2. Required fields: every top-level and nested path the page reads
   (`candidates`, `jobs`, `totalInterviews`, `credits`,
   `totalCreditsDisbursed`, and all nine `pipeline.*` counters).
3. Data types: integer counters, object/array containers, well-typed
   credits rows.
4. Values: non-negative counters, no unknown `pipeline` fields (the page
   would silently ignore them), non-empty company names, unique company
   ids, no negative credits.
5. Cross-field consistency: `totalCreditsDisbursed` = sum of credits rows;
   `pipeline.started` = `totalInterviews.aggregate.count` (the KPI shows
   `pipeline.started`); `completed` and every exclusive stage <= `started`
   (the page clamps the completion badge to 100%, which would hide it).

UI side (Playwright logs in through the real form, intercepts the
`getDashboardData` payload the page itself received, waits for the
count-up animation to settle):

6. Login reached `/dashboard`; page fetched from `Dashboard_API_URL`
   with a bearer token; default filter excludes individual users.
7. Page payload vs the harness's own API read (real-time drift).
8. KPI cards: Active candidates (+ "across N companies"), Active jobs,
   Total interviews (+ completion badge), Credits allocated
   (+ "n companies below 25").
9. Stage breakdown: eight rows (Link not opened, Link opened, Started,
   In progress, Paused, Completed, Candidate exit, Invite expired),
   counts and percentages.
10. Interview pipeline chart: a bar label for every non-zero stage.
11. Donut charts: TOTAL and legend of Communications, Open rate,
    Response mix.
12. "View by company" dialog: every company's remaining credits.

The expected UI values are derived with the dashboard's own formulas
(read from the deployed bundle), documented at the top of
`app_e2e/dashboard/expected.py` - e.g. *Total interviews* =
`pipeline.started`; *Opened* (donuts) = opened + startedOnly + inProgress
+ paused + completed + candidateExit; completion =
`min(100, Math.round(completed / total * 100))`; stage percentages divide
by the sum of the eight rows and use `Math.round` (halves up).

## API shape history

- **2026-09-03 (current)**: `pipeline` object with `notOpened`, `opened`,
  `started`, `startedOnly`, `inProgress`, `paused`, `expired`,
  `completed`, `candidateExit`. Deployed to QA during the afternoon of
  2026-09-03 together with a dashboard build that adds the Started /
  Candidate exit / Invite expired rows.
- **before**: six per-stage arrays (`linkNotOpened[0].count.aggregate.count`
  etc.) whose counts overlapped (they summed to 3,969 against 3,465 total
  interviews, so stage percentages added up to ~189%). That shape is no
  longer accepted: the required-field checks fail loudly if the API
  reverts to it.

Latest result (2026-09-03, QA): 128 checks PASS, 0 FAIL, 1 N/A (the
"Opened" bar has count 0, so the chart draws no label).

## Notes

- `pytest` with no arguments collects `tests/app_e2e` too (like
  `tests/e2e`, it opens a real browser); use the runner script or a path
  to run only what you need.
- The dashboard session is a normal admin login - it does not touch the
  interview one-tab lock or any interview URL.

## Layout

```
app_e2e/dashboard/
  config.py     .env -> DashboardConfig
  auth.py       ACCESS_TOKEN / sign-in resolution, JWT claims
  api_client.py getDashboardData + GetLookup client
  schema.py     required fields / types / values / consistency checks
  expected.py   API payload -> values the page must render
  ui.py         Playwright capture (login, intercept, read widgets)
  validator.py  UI vs expected comparisons
  checks.py     CheckResult rows + registry
  report.py     HTML + JSON PASS/FAIL report
tests/app_e2e/
  conftest.py                     one login/API read/browser capture per session; writes the report
  test_dashboard_data.py          live checks
  test_dashboard_rules_offline.py unit tests of the rules (no network)
scripts/run_dashboard_validation.py
```
