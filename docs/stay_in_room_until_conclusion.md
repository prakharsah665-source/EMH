# Full-interview drive: stay in the room until the AI interviewer concludes

Implemented 2026-08-17 in `tests/e2e/test_bot_responsiveness.py`.

## Behaviour

- The evaluator joins the interview room once and **stays for the entire
  interview**. The drive loop exits ONLY when the AI interviewer concludes
  the interview (`interview_concluded()`: socket.io `isExit` /
  `INTERVIEW_COMPLETED` frames, or end-of-interview UI text). There is no
  fixed question/turn count (the old `MAX_TURNS=25` cap is gone;
  `EMH_MAX_TURNS` defaults to 0 = unlimited).
- A bot that does not answer is **never** a reason to leave the room:
  1. the stall is recorded as a *deferred failure* with the full pipeline
     matrix / RTC diagnostics / screenshot (`capture_failure(keep_trace=True)`,
     so the Playwright trace keeps recording);
  2. the evaluator waits `EMH_BOT_RECOVERY_TIMEOUT` (default 120 s) for a late
     reply or a conclusion signal;
  3. if still silent it re-speaks the same candidate answer (the app's own
     watchdog has returned to `USER_WAITING`) and repeats.
- A missing greeting / agent that never joined is likewise deferred (turn 0)
  and the loop proceeds so the bot can recover.
- The only exits that are not the bot's conclusion are safety caps:
  `EMH_INTERVIEW_MAX_S` wall-clock (default 45 min) and the optional
  `EMH_MAX_TURNS`. Hitting one marks the transcript INCOMPLETE and fails with
  `INTERVIEW DID NOT COMPLETE`.
- Failure reporting is preserved but **deferred**: after the bot concludes,
  the transcript is saved with `complete=true` (scorable), then the test
  fails with `BOT RESPONSIVENESS DEGRADED - interview completed` listing every
  stall (turn/attempt/label, recovered or not) and the first stall's full
  diagnostics. With zero stalls it passes as before.

## Artifacts

- `artifacts/debug/bot_responsiveness_<ts>/deferred_failures.json` — every
  stall with its diagnostics message.
- `artifacts/screenshots/bot_responsiveness_turnNN_attemptK_stall.png`.
- `artifacts/reports/bot_responsiveness_report.json` — `stalls`, `recoveries`,
  `conclusion_reason`, `deferred_failures`, per-attempt `turns`.
- `artifacts/transcripts/actual_transcript_status.json` — `complete` is true
  only when the bot concluded.

## Offline proof

`tests/e2e_offline/test_stay_in_room_until_conclusion.py` drives the real
`run_multi_turn()` against a scripted fake bot: stall → recovery wait → late
reply, stall → re-prompt → answer, 14+ turns without stopping, and a dead bot
that is left only at the wall-clock cap.

## The one non-bot exit: server room teardown

Observed live on 2026-08-17 (sessions 8386, runs 16:36 and 16:37): about
120 s after the agent stops answering, room-api closes the socket.io
transport (`Disconnected: transport close`) and LiveKit goes
`connected -> disconnected`; the app unpublishes the mic. No conclusion
signal is ever sent, so re-prompting into that page can never recover.
`room_gone()` (all peer connections closed/failed/disconnected + no live
remote track) sustained for `EMH_ROOM_DISCONNECT_GRACE` (default 180 s)
makes the drive leave with `ROOM DISCONNECTED BY SERVER (no conclusion
signal)` — an ENVIRONMENT classification, transcript marked incomplete.
Set `EMH_ROOM_DISCONNECT_GRACE=0` to disable and rely on the wall-clock cap
only.
