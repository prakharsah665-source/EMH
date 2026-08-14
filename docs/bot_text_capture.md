# How does agent text reach the browser? (Phase 2 investigation)

Investigated 2026-08-14 against the live 11-turn capture run
(`artifacts/debug/bot_responsiveness_20260814_114414/`) plus a live
data-channel probe.

## Findings

1. **DOM: no interviewer text and no candidate STT during the interview.**
   Across 11 real turns the pipeline matrix consistently reports "answer
   transcript not visible in page text" / "no new agent text visible in page
   body". The bot's replies are audio-only in the UI.
2. **The greeting text that renders on screen is NOT agent-generated text.**
   It is the static onboarding copy of the pre-interview tour ("Welcome to
   Your AI Interview / Let's take a quick tour…") plus UI chrome ("Elapsed
   time", "AI is speaking", "Jamie (Interviewer)"). It exists before the
   agent says anything, so it is not a usable transcript source.
3. **socket.io (`room-api-v1-dev…/socket.io/`) carries state, not text —
   audited exhaustively, no backend warranted.** Full decode of every
   frame from the 11-turn run: `interview-state-change` (99 sent / 23
   received, `{"state": ...}` only), `bot-speech-started` /
   `bot-speech-ended` (11 each, boolean flags only — these fire once per
   bot utterance, so if the server attached utterance text anywhere it
   would be here; the payloads are complete JSON with no text field),
   `user-speech-detected` / `last-user-video-chunk` (null args),
   `initiate-chat` / `startInterview` (JWT only),
   `fetch-job-candidate-details` → `job-candidate-details` (candidate/job
   METADATA — name, role — sent once at page load, not conversation). A
   scan of every string field found zero multi-word conversation text.
   Conclusion: socket.io is NOT an authoritative conversation-text
   source; no transcript-capture backend was added for it. (The frame
   recorder now keeps text frames to 2000 chars so future runs stay
   conclusive.)
4. **Browser console: audio-pipeline logs only** (chunk processing, state
   machine, watchdog) — no text.
5. **LiveKit signalling WebSocket is binary protobuf**; transcriptions
   (`lk.transcription` text streams) and chat, if the agent publishes them,
   travel over the **WebRTC data channel**, which WebSocket capture cannot
   see. A live probe that hooked `RTCPeerConnection.createDataChannel` /
   `ondatachannel` on a **consumed** session recorded zero channels — 
   inconclusive, because the agent does not rejoin a consumed room (no
   PeerConnection activity at all).
6. **A recruiter-facing conversation-log API very likely exists** (the app
   records the full interview; the dashboard presumably displays it) but no
   endpoint is documented in this repo.

## What was built (collectors/transcript_capture.py)

`TranscriptCapture.get_turns() -> [{role, text, ts, source, confidence}]`,
selected via `EMH_TRANSCRIPT_CAPTURE` (`livekit | app-api | dom | stt |
auto`), PII-redacted at capture time (`evaluation/redaction.py`):

| Backend | Source | Status |
|---|---|---|
| `LiveKitEventsCapture` (preferred) | Data-channel messages recorded in-page by `TRANSCRIPT_HOOK_JS` (installed by the capture run; drained to `artifacts/transcripts/livekit_transcript_events.json`) | Ready — the next fresh capture run definitively answers whether the agent publishes text, and harvests it if so |
| `AppApiCapture` | `EMH_TRANSCRIPT_API_URL` (+ `ACCESS_TOKEN`) returning `[{role, text|content, ts}]` | Ready, awaiting an endpoint |
| `DomScrapeCapture` (stopgap) | Existing DOM capture metadata; honest confidences: `app-stt` high, `dom-diff` medium, `dom-greeting`/`injected-audio` low | Working today (but yields ~no assistant text, per finding 1) |
| `SttCapture` (fallback) | Local STT over recorded bot audio | Documented, not implemented — the capture run does not record bot audio to disk yet |

`test_bot_responsiveness` now installs the hook and prints
`LiveKit data-channel transcript hook: N events, M with text` at the end of
every capture, so the channel question is answered on every run.

## Next steps if the data channel turns out to be silent

If a fresh capture run reports 0 text events, the agent does not publish
transcriptions, and the two realistic paths are (a) the conversation-log
API (ask the EMH team for the endpoint; `AppApiCapture` is ready) or
(b) recording the remote audio track to disk during capture and running
local STT (`SttCapture`).
