# Interview session audit & provisioning design

Audited 2026-08-14 against a complete `scripts/run_all_tests.py` run.

## 1. How many times is the same INTERVIEW_URL opened per complete run?

**12 distinct navigations of the same interview URL** (13 page loads counting
the second session in the permissions test), every one in a fresh browser
context, plus up to 12 bounded lock-screen reloads per page-object launch:

| # | Test / helper | Navigation site | Enters interview room? |
|---|---|---|---|
| 1 | `test_bot_responsiveness` | `pages/interview_launch.py:155` (shared launch) | **Yes — full interview** |
| 2 | `test_audio_configuration` | `test_audio_configuration.py:794` | No (setup screens) |
| 3 | `test_complete_interview` | `test_complete_interview.py:272` | No (clicks launch, dumps state) |
| 4 | `test_interview_launch` | `test_interview_launch.py:29` | No (page-load smoke) |
| 5–6 | `test_interview_permissions` (2 sessions) | `test_interview_permissions.py:315` via `:468,:477` | No (setup screens) |
| 7 | `test_recording_consent` | `test_recording_consent.py:616` | No (setup screens) |
| 8 | `test_socket_connection` | `test_socket_connection.py:139` | No (launch + consent) |
| 9 | `test_system_configuration` | `test_system_configuration.py:291` | No (setup screens) |
| 10 | `test_continue_to_interview` | `pages/interview_launch.py:155` | **Yes** |
| 11 | `test_interview_room` | `test_interview_room.py:998` | **Yes** |
| 12 | `test_livekit_connection` | `test_livekit_connection.py:110` (+ lock reloads `:225`) | **Yes** |

## 2. Intentional continuation or incorrect fresh-session restart?

Empirical facts (verified live against a consumed session):

- The launch page shows **"Start interview"** only for a never-opened session;
  any prior open flips it to **"Continue interview"** — even a fully driven,
  mid-interview-frozen session still shows the normal launch page with
  "Continue interview", never a "completed" screen.
- Clicking Continue on a consumed session re-enters the **full System
  Configuration setup flow** normally.
- The agent greeting (and therefore interview state) fires on **room join**
  ("Continue to Interview"), not on the launch click.

Classification:

- **Setup-screen tests (rows 2–9): intentional continuation.** They exercise
  setup UI that works identically on a re-entered session. They were
  *incorrectly written* as fresh-session restarts (exact-match "Start
  Interview" buttons, one-shot queries) — that is a test bug, fixed by
  accepting `Start|Continue` (`pages/interview_launch.py:LAUNCH_BUTTON_RE`)
  with bounded waits.
- **`test_bot_responsiveness` (row 1): the one legitimate fresh-session
  consumer.** The full-interview evaluation must own the run's fresh session
  — it runs FIRST in `scripts/run_all_tests.py` ordering.
- **Room-joining tests (rows 10–12): incorrect fresh-session restarts.**
  Each join fires the greeting and mutates interview state, so they can
  neither share the full-interview session before it runs (they would consume
  its greeting) nor meaningfully re-enter it afterwards (dead/finished room →
  false "bot" failures, e.g. "LiveKit WebSocket was not established"). These
  genuinely require an isolated session.

## 3. Design implemented

- **One fresh session per run for the full interview.**
  `scripts/run_all_tests.py` orders `test_bot_responsiveness` first; it
  requires an *unconsumed* session and records the room join in
  `artifacts/session_ledger.json` (`config/interview_session.py:
  mark_session_consumed`). No session is created per test.
- **Used-session guard.** `require_unconsumed_session()` raises
  `SESSION ALREADY CONSUMED … session/environment condition, not a bot
  failure` when a ledger entry exists for the URL's
  `(candidate_id, job_id, iat)`. Re-running a complete suite against an
  already-driven URL now fails at the session layer instead of producing
  fake bot failures. (Residual risk: consumption by parties outside this
  harness is not detectable from the launch page.)
- **Two sessions per run (2026-08-17).** `INTERVIEW_URL` is reserved
  EXCLUSIVELY for `test_bot_responsiveness`; **every other E2E test** uses
  `EMH_TESTS_URL` (legacy name `EMH_ROOM_TESTS_URL` still honoured) via
  `require_fresh_tests_url()` / `tests/e2e/session_policy.py`. Setup-screen
  tests re-enter it as continuation. `require_fresh_tests_url()` fails if it
  is the same session as `INTERVIEW_URL`.
- **One shared room join per run (2026-08-24).** The three room-joining
  tests (`test_continue_to_interview`, `test_interview_room`,
  `test_livekit_connection`) share a SINGLE join of `EMH_TESTS_URL`
  (`tests/e2e/shared_room.py` + the `shared_room` fixture in
  `tests/e2e/conftest.py`): the first to run joins through the unchanged
  policy (unconsumed session required; join ledgered), the others attach
  to the same live page and validate their own responsibility, and the
  browser leaves the room gracefully exactly once after the last of them.
  Previously each test joined on its own, so with a budget of one joinable
  session per run the second and third skipped `SESSION ALREADY CONSUMED`
  deterministically. That skip now only occurs when `EMH_TESTS_URL` was
  consumed *before* the run (the guard is re-checked per consumer).
- **No concurrent runs on one session.** The app has a one-tab lock (a
  second joiner ejects the first — seen live 2026-08-17). `config/session_lock.py`
  + `tests/e2e/conftest.py` hold a per-session pid lock for each E2E test;
  a live holder → `SESSION IN USE` (environment). `scripts/run_all_tests.py`
  runs a **pre-flight** requiring both URLs fresh, distinct, unused and
  unlocked (override with `EMH_ALLOW_STALE_INTERVIEW=1`).
- **ONE primary URL.** `INTERVIEW_URL` (.env) is the primary session.
  `EMH_INTERVIEW_URL` is a per-run override for the CLI/provisioner ONLY -
  never put it in `.env` next to `INTERVIEW_URL`. Every test resolves the URL
  via `config.interview_session.get_interview_url()` /
  `require_fresh_interview_url()` (no test reads `config.settings.INTERVIEW_URL`
  directly), so the full interview, setup-screen tests, transcript and
  evaluation always target the SAME session. If both vars are set to
  different sessions, `get_interview_url()` prints a loud `[WARNING]` naming
  the winner (fixed 2026-08-17: `.env` had two links, so 4 setup tests silently
  opened a different candidate than the evaluated interview).
- **Provisioner hook.** `config/session_provisioner.py` mints a per-run
  session when `EMH_PROVISION_API_URL` is configured
  (`scripts/run_all_tests.py` exports it as `EMH_INTERVIEW_URL` for the
  whole run).

## 4. Provisioning endpoint the EMH backend needs to expose

No session-provisioning API exists in this repo today. The harness is ready
for:

```
POST {EMH_PROVISION_API_URL}
Authorization: Bearer <ACCESS_TOKEN>            # dashboard JWT (.env)
Content-Type: application/json
{"job_id": <int>, "candidate": {"name": "...", "email": "..."}}

200 → {"interview_url": "https://hiring.easemyhiring.ai/interview/<JWT>?version=v1"}
```

The returned URL must embed the standard interview JWT
(`candidate_id/job_id/company_id/iat/exp`) so the existing freshness,
expiry and used-session guards work unchanged. Optional env:
`EMH_PROVISION_JOB_ID` (job to mint against), `EMH_TESTS_URL`
(the isolated second session for all non-full-interview E2E tests).
