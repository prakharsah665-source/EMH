"""
TranscriptCapture: pluggable sources of the REAL interview
transcript.

Investigation summary (2026-08-14, live run + artifact mining;
see docs/bot_text_capture.md):

- The app renders NO interviewer text and NO candidate STT in
  the DOM during the interview; bot replies are audio-only.
  The only DOM "assistant" text is the pre-interview greeting
  screen copy.
- The room-api socket.io channel carries state events only
  (bot-speech-started/ended, interview-state-change) - no text.
- Browser console: audio-pipeline logs only - no text.
- LiveKit transcriptions/chat, if the agent publishes them,
  travel over the WebRTC DATA CHANNEL (invisible to WebSocket
  capture). TRANSCRIPT_HOOK_JS below records every data-channel
  message so the capture run can both answer whether that
  channel exists and harvest its text.
- A recruiter-facing conversation-log API likely exists but is
  undocumented here; AppApiCapture is env-configured for it.

Interface: every backend returns
    get_turns() -> [{role, text, ts, source, confidence}]
with role in {assistant, user}, source naming the channel and
confidence in {high, medium, low}. Text is PII-redacted AT
CAPTURE TIME (evaluation.redaction).

Backend selection: EMH_TRANSCRIPT_CAPTURE = livekit | app-api |
dom | stt | auto (default). "auto" picks the first available of
livekit -> app-api -> dom.
"""

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from evaluation.redaction import redact_pii


LIVEKIT_EVENTS_PATH = Path(
    "artifacts/transcripts/livekit_transcript_events.json"
)
DOM_METADATA_PATH = Path(
    "artifacts/transcripts/actual_transcript_capture_metadata.json"
)


# ============================================================
# In-page hook (installed by the E2E capture run)
#
# Records every WebRTC data-channel message into
# window.__emhTranscriptEvents:
#   - string payloads (LiveKit chat / lk.transcription text
#     streams) verbatim,
#   - binary payloads as printable UTF-8 runs (best-effort
#     text extraction from the protobuf DataPacket).
# The capture run drains this via drain_transcript_events().
# ============================================================

TRANSCRIPT_HOOK_JS = """
(() => {
  if (window.__emhTranscriptEvents) return;
  window.__emhTranscriptEvents = [];
  const push = (entry) => {
    if (window.__emhTranscriptEvents.length < 5000) {
      window.__emhTranscriptEvents.push(entry);
    }
  };
  const printableRuns = (buf) => {
    try {
      const bytes = buf instanceof ArrayBuffer
        ? new Uint8Array(buf) : new Uint8Array(buf.buffer || buf);
      let out = [], run = '';
      for (const b of bytes) {
        if (b >= 32 && b < 127) run += String.fromCharCode(b);
        else { if (run.length >= 4) out.push(run); run = ''; }
      }
      if (run.length >= 4) out.push(run);
      return out.join(' ');
    } catch (e) { return ''; }
  };
  const hook = (dc, origin) => {
    push({ev: 'channel', label: dc.label, origin, ts: Date.now()});
    dc.addEventListener('message', (m) => {
      const d = m.data;
      push({
        ev: 'message',
        label: dc.label,
        origin,
        ts: Date.now(),
        kind: typeof d === 'string' ? 'string' : 'binary',
        size: d.byteLength ?? (d.length ?? 0),
        text: typeof d === 'string'
          ? d.slice(0, 2000)
          : printableRuns(d).slice(0, 2000),
      });
    });
  };
  const OrigPC = window.RTCPeerConnection;
  if (!OrigPC) return;
  const Wrapped = function(...args) {
    const pc = new OrigPC(...args);
    const origCreate =
      pc.createDataChannel && pc.createDataChannel.bind(pc);
    if (origCreate) {
      pc.createDataChannel = (label, opts) => {
        const dc = origCreate(label, opts);
        hook(dc, 'local');
        return dc;
      };
    }
    pc.addEventListener('datachannel', (e) => hook(e.channel, 'remote'));
    return pc;
  };
  Wrapped.prototype = OrigPC.prototype;
  Object.setPrototypeOf(Wrapped, OrigPC);
  window.RTCPeerConnection = Wrapped;
})();
"""


async def drain_transcript_events(page) -> list[dict]:
    """
    Pull the recorded data-channel events out of the page and
    persist them (PII-redacted) for LiveKitEventsCapture.
    Returns the drained events. Safe to call on a dead page.
    """

    try:
        events = await page.evaluate(
            "() => window.__emhTranscriptEvents || []"
        )
    except Exception:
        events = []

    for event in events:
        if event.get("text"):
            event["text"] = redact_pii(event["text"])

    LIVEKIT_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVEKIT_EVENTS_PATH.write_text(
        json.dumps(events, indent=2), encoding="utf-8"
    )
    return events


# ============================================================
# Interface
# ============================================================

def make_turn(
    role: str,
    text: str,
    ts: float | None,
    source: str,
    confidence: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "text": redact_pii(text),
        "ts": ts,
        "source": source,
        "confidence": confidence,
    }


class TranscriptCapture(ABC):
    """A source of real interview transcript turns."""

    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool:
        """Can this backend produce turns right now?"""

    @abstractmethod
    def get_turns(self) -> list[dict[str, Any]]:
        """[{role, text, ts, source, confidence}] in order."""


class LiveKitEventsCapture(TranscriptCapture):
    """
    PREFERRED. Parses the data-channel events recorded by
    TRANSCRIPT_HOOK_JS during the capture run. String payloads
    (LiveKit chat / lk.transcription streams) are high
    confidence; printable runs extracted from binary protobuf
    packets are medium confidence.
    """

    name = "livekit"

    # Binary printable-run noise that is protocol framing, not
    # speech (participant identities, stream ids, topics).
    _NOISE_PREFIXES = (
        "PA_", "TR_", "lk.", "agent-", "identity",
    )

    def __init__(self, path: Path | None = None):
        # Default resolved at construction so path overrides
        # (tests, alternate artifact roots) take effect.
        self.path = Path(path or LIVEKIT_EVENTS_PATH)

    def _events(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _speech_events(self) -> list[dict]:
        speech = []
        for event in self._events():
            if event.get("ev") != "message":
                continue
            text = (event.get("text") or "").strip()
            if len(text) < 4:
                continue
            speech.append(event)
        return speech

    def available(self) -> bool:
        return bool(self._speech_events())

    def get_turns(self) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for event in self._speech_events():
            text = event["text"].strip()
            role = "assistant"
            confidence = (
                "high" if event.get("kind") == "string" else "medium"
            )

            if event.get("kind") == "string":
                # LiveKit chat/transcription payloads are JSON.
                try:
                    payload = json.loads(text)
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    inner = (
                        payload.get("text")
                        or payload.get("message")
                        or ""
                    )
                    if not inner.strip():
                        continue
                    identity = str(
                        payload.get("participantIdentity")
                        or payload.get("identity")
                        or ""
                    )
                    role = (
                        "user"
                        if "candidate" in identity.lower()
                        else "assistant"
                    )
                    text = inner.strip()
            else:
                # Binary extraction: drop framing-only runs.
                words = [
                    w
                    for w in text.split()
                    if not w.startswith(self._NOISE_PREFIXES)
                ]
                text = " ".join(words)
                if len(text) < 12:
                    continue

            turns.append(
                make_turn(
                    role=role,
                    text=text,
                    ts=event.get("ts"),
                    source=f"livekit-{event.get('label') or 'datachannel'}",
                    confidence=confidence,
                )
            )
        return turns


class AppApiCapture(TranscriptCapture):
    """
    Conversation-log API backend. Configure:

        EMH_TRANSCRIPT_API_URL - endpoint returning the
            interview conversation log as JSON:
            [{"role": "assistant"|"user", "text"|"content": ...,
              "ts": <epoch-ms, optional>}]
        ACCESS_TOKEN - bearer token (.env)

    No such endpoint is documented today; this backend makes
    the harness ready the moment one exists.
    """

    name = "app-api"

    def __init__(self) -> None:
        self.url = os.getenv("EMH_TRANSCRIPT_API_URL")
        self.token = os.getenv("ACCESS_TOKEN")

    def available(self) -> bool:
        return bool(self.url)

    def get_turns(self) -> list[dict[str, Any]]:
        if not self.url:
            return []

        request = urllib.request.Request(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            }
            if self.token
            else {"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        turns = []
        for item in payload if isinstance(payload, list) else []:
            role = item.get("role")
            text = item.get("text") or item.get("content") or ""
            if role not in ("assistant", "user") or not text.strip():
                continue
            turns.append(
                make_turn(
                    role=role,
                    text=text.strip(),
                    ts=item.get("ts"),
                    source="app-api",
                    confidence="high",
                )
            )
        return turns


class DomScrapeCapture(TranscriptCapture):
    """
    STOPGAP. Wraps the existing DOM-diff capture metadata
    (actual_transcript_capture_metadata.json). Confidence
    reflects provenance honestly:

      app-stt       high   (the app rendered its own STT)
      dom-diff      medium (DOM text attributed by heuristic)
      dom-greeting  low    (greeting screen chrome)
      injected-audio low   (what the harness SAID, not what
                            the app heard - reference text)
    """

    name = "dom"

    _CONFIDENCE = {
        "app-stt": "high",
        "dom-diff": "medium",
        "dom-greeting": "low",
        "injected-audio": "low",
    }

    def __init__(self, path: Path | None = None):
        self.path = Path(path or DOM_METADATA_PATH)

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def available(self) -> bool:
        return bool(self._entries())

    def get_turns(self) -> list[dict[str, Any]]:
        turns = []
        for entry in self._entries():
            role = entry.get("role")
            text = (entry.get("content") or "").strip()
            if role not in ("assistant", "user") or not text:
                continue
            source = entry.get("source") or "dom-diff"
            turns.append(
                make_turn(
                    role=role,
                    text=text,
                    ts=None,
                    source=source,
                    confidence=self._CONFIDENCE.get(source, "low"),
                )
            )
        return turns


class SttCapture(TranscriptCapture):
    """
    FALLBACK (not implemented). Would run local STT (e.g.
    Whisper) over recorded bot audio. The capture run does not
    yet record the bot's audio track to disk; when it does,
    implement get_turns() with source="stt-local",
    confidence="medium". Declared so EMH_TRANSCRIPT_CAPTURE=stt
    fails with a clear reason instead of a KeyError.
    """

    name = "stt"

    def available(self) -> bool:
        return False

    def get_turns(self) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "SttCapture requires recorded bot audio, which the "
            "capture run does not produce yet. Use "
            "EMH_TRANSCRIPT_CAPTURE=livekit|app-api|dom."
        )


# ============================================================
# Selection
# ============================================================

_BACKENDS = {
    "livekit": LiveKitEventsCapture,
    "app-api": AppApiCapture,
    "dom": DomScrapeCapture,
    "stt": SttCapture,
}

AUTO_ORDER = ("livekit", "app-api", "dom")


def select_capture() -> TranscriptCapture:
    """
    Resolve the transcript source from EMH_TRANSCRIPT_CAPTURE
    (default "auto": first available of livekit -> app-api ->
    dom). Raises RuntimeError - an evaluator/environment
    classification, never a bot failure - when nothing is
    available.
    """

    choice = os.getenv("EMH_TRANSCRIPT_CAPTURE", "auto").lower()

    if choice != "auto":
        if choice not in _BACKENDS:
            raise RuntimeError(
                f"Unknown EMH_TRANSCRIPT_CAPTURE={choice!r}; "
                f"expected one of {sorted(_BACKENDS)} or 'auto'."
            )
        return _BACKENDS[choice]()

    for name in AUTO_ORDER:
        backend = _BACKENDS[name]()
        if backend.available():
            return backend

    raise RuntimeError(
        "No transcript capture backend is available: no LiveKit "
        "data-channel text was recorded, no EMH_TRANSCRIPT_API_URL "
        "is configured, and no DOM capture metadata exists. Run "
        "the E2E capture first (pytest tests/e2e/"
        "test_bot_responsiveness.py -s). This is a capture/"
        "environment condition, not a bot failure."
    )
