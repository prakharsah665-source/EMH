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
AUDIO_DIR = Path("artifacts/audio")
STT_LOCAL_PATH = Path(
    "artifacts/transcripts/stt_local_transcript.json"
)
CAPTURE_EVIDENCE_PATH = Path("artifacts/capture_evidence.json")


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


# ============================================================
# Remote (bot) audio recording for the stt-local fallback.
#
# Wraps RTCPeerConnection 'track' events to keep a registry of
# live remote audio tracks, and exposes a per-turn recorder:
#   window.__emhBotRec.start()          - begin recording the
#       newest live remote audio track (MediaRecorder, webm/
#       opus, 1s timeslice).
#   window.__emhBotRec.stop() -> base64 - stop and return the
#       recorded clip (empty string when nothing was captured).
# ============================================================

AUDIO_RECORD_JS = """
(() => {
  if (window.__emhBotRec) return;
  const remoteTracks = [];
  const OrigPC = window.RTCPeerConnection;
  if (OrigPC) {
    const Wrapped = function(...args) {
      const pc = new OrigPC(...args);
      pc.addEventListener('track', (event) => {
        if (event.track && event.track.kind === 'audio') {
          remoteTracks.push(event.track);
        }
      });
      return pc;
    };
    Wrapped.prototype = OrigPC.prototype;
    Object.setPrototypeOf(Wrapped, OrigPC);
    window.RTCPeerConnection = Wrapped;
  }

  let recorder = null;
  let chunks = [];

  const liveTrack = () => {
    for (let i = remoteTracks.length - 1; i >= 0; i--) {
      if (remoteTracks[i].readyState === 'live') {
        return remoteTracks[i];
      }
    }
    return null;
  };

  window.__emhBotRec = {
    start() {
      this._stopSync();
      const track = liveTrack();
      if (!track) return false;
      chunks = [];
      try {
        recorder = new MediaRecorder(
          new MediaStream([track]),
          { mimeType: 'audio/webm;codecs=opus' }
        );
      } catch (e) {
        recorder = null;
        return false;
      }
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      recorder.start(1000);
      return true;
    },
    _stopSync() {
      if (recorder && recorder.state !== 'inactive') {
        try { recorder.stop(); } catch (e) {}
      }
      recorder = null;
    },
    async stop() {
      if (!recorder) return '';
      const rec = recorder;
      recorder = null;
      const done = new Promise((resolve) => {
        rec.onstop = resolve;
        setTimeout(resolve, 2000);
      });
      try { rec.stop(); } catch (e) {}
      await done;
      if (!chunks.length) return '';
      const blob = new Blob(chunks, { type: 'audio/webm' });
      chunks = [];
      const buf = await blob.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let binary = '';
      const step = 0x8000;
      for (let i = 0; i < bytes.length; i += step) {
        binary += String.fromCharCode(
          ...bytes.subarray(i, i + step)
        );
      }
      return btoa(binary);
    },
  };
})();
"""


async def start_bot_audio_recording(page) -> bool:
    """Begin recording the bot's remote audio track."""

    try:
        return bool(
            await page.evaluate(
                "() => window.__emhBotRec && "
                "window.__emhBotRec.start()"
            )
        )
    except Exception:
        return False


async def stop_bot_audio_recording(
    page, label: str
) -> Path | None:
    """
    Stop the current recording and persist it under
    artifacts/audio/<label>.webm. Returns the path, or None
    when nothing was captured. Never raises.
    """

    try:
        encoded = await page.evaluate(
            "() => window.__emhBotRec ? "
            "window.__emhBotRec.stop() : ''"
        )
    except Exception:
        return None

    if not encoded:
        return None

    import base64 as _base64

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{label}.webm"
    path.write_bytes(_base64.b64decode(encoded))
    return path


# ============================================================
# Local Whisper transcription (faster-whisper)
# ============================================================

def transcribe_captured_audio(
    entries: list[dict],
    model_name: str | None = None,
) -> dict:
    """
    Transcribe captured audio files with local faster-whisper
    and write STT_LOCAL_PATH. `entries` is a list of
    {"role": "assistant"|"user", "turn": int, "audio_path": str}.
    Returns a summary dict (also used in the evidence
    manifest). Degrades gracefully - a missing model or
    library is an ENVIRONMENT limitation, never a test error.
    """

    summary = {
        "requested": len(entries),
        "transcribed": 0,
        "skipped_reason": None,
        "path": str(STT_LOCAL_PATH),
    }

    if os.getenv("EMH_DISABLE_STT_LOCAL") == "1":
        summary["skipped_reason"] = "EMH_DISABLE_STT_LOCAL=1"
        return summary

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        summary["skipped_reason"] = (
            "faster-whisper is not installed (pip install "
            "faster-whisper) - stt-local capture unavailable"
        )
        return summary

    usable = [
        entry
        for entry in entries
        if entry.get("audio_path")
        and Path(entry["audio_path"]).exists()
    ]
    if not usable:
        summary["skipped_reason"] = "no captured audio files"
        return summary

    model = WhisperModel(
        model_name or os.getenv("EMH_STT_MODEL", "base"),
        compute_type="int8",
    )

    results = []
    for entry in usable:
        segments, _info = model.transcribe(
            entry["audio_path"], language="en"
        )
        text = " ".join(
            segment.text.strip() for segment in segments
        ).strip()
        results.append(
            {
                "turn": entry.get("turn"),
                "role": entry.get("role"),
                "audio_path": entry["audio_path"],
                "text": redact_pii(text),
            }
        )
        summary["transcribed"] += 1

    STT_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    STT_LOCAL_PATH.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return summary


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
# Capture-run artifact lifecycle + evidence manifest
# ============================================================

def clear_previous_capture_artifacts() -> list[str]:
    """
    Remove the PREVIOUS session's capture artifacts so a new
    capture run can never leave a mixed old/new state behind
    (a stale transcript beside a fresh sidecar, old bot audio
    beside new turns). Called by the capture run AFTER its
    session guards pass - a guard failure must not wipe the
    last good run's evidence.
    """

    from collectors.transcript_collector import (
        AUDIO_RECORDS_PATH,
        TRANSCRIPT_PATH,
        TRANSCRIPT_STATUS_PATH,
    )

    removed = []
    targets = [
        TRANSCRIPT_PATH,
        TRANSCRIPT_STATUS_PATH,
        AUDIO_RECORDS_PATH,
        DOM_METADATA_PATH,
        LIVEKIT_EVENTS_PATH,
        STT_LOCAL_PATH,
        CAPTURE_EVIDENCE_PATH,
    ]
    if AUDIO_DIR.exists():
        targets.extend(sorted(AUDIO_DIR.glob("*.webm")))

    for path in targets:
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        except OSError:
            pass
    return removed


def write_capture_evidence(
    *,
    dom_metadata: list[dict],
    datachannel_events: list[dict],
    socketio_frames: list[dict],
    audio_files: list[str],
    stt_summary: dict,
) -> dict:
    """
    Write artifacts/capture_evidence.json: one manifest per
    capture run stating which sources produced USABLE BOT TEXT
    and which were empty, so nobody has to re-derive it from
    raw artifacts.
    """

    dom_bot = [
        entry
        for entry in dom_metadata
        if entry.get("role") == "assistant"
        and (entry.get("content") or "").strip()
    ]

    dc_text = [
        event
        for event in datachannel_events
        if event.get("ev") == "message"
        and len((event.get("text") or "").strip()) >= 4
    ]

    socketio_event_names = set()
    for frame in socketio_frames:
        payload = frame.get("payload") or ""
        if payload.startswith('42["'):
            parts = payload.split('"')
            if len(parts) >= 2:
                socketio_event_names.add(parts[1])
    socketio_event_names = sorted(socketio_event_names)

    stt_turns = []
    if STT_LOCAL_PATH.exists():
        try:
            stt_turns = json.loads(
                STT_LOCAL_PATH.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            stt_turns = []
    stt_bot = [
        entry
        for entry in stt_turns
        if entry.get("role") == "assistant"
        and (entry.get("text") or "").strip()
    ]

    evidence = {
        "sources": {
            "dom": {
                "assistant_entries_with_text": len(dom_bot),
                "usable_bot_text": bool(dom_bot),
                "note": (
                    "DOM renders no bot text during the "
                    "interview; chrome is filtered."
                ),
            },
            "livekit_datachannel": {
                "events": len(datachannel_events),
                "events_with_text": len(dc_text),
                "usable_bot_text": bool(dc_text),
            },
            "socketio": {
                "frames": len(socketio_frames),
                "event_names": socketio_event_names,
                "usable_bot_text": False,
                "note": (
                    "state events + candidate/job metadata "
                    "only (audited 2026-08-14, docs/"
                    "bot_text_capture.md)"
                ),
            },
            "remote_audio": {
                "files": audio_files,
                "count": len(audio_files),
            },
            "stt_local": {
                **stt_summary,
                "assistant_turns_with_text": len(stt_bot),
                "usable_bot_text": bool(stt_bot),
            },
        },
    }

    CAPTURE_EVIDENCE_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    CAPTURE_EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return evidence


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
    FALLBACK. Serves the transcript produced by local
    faster-whisper over audio captured during the interview:
    the bot's remote audio track (recorded per turn to
    artifacts/audio/ by the capture run) and the actual
    candidate answer clips that were injected. All turns carry
    source="stt-local" and confidence="medium" - honest for
    STT-derived text, and deliberately BELOW livekit/app-api so
    this backend can never override a higher-confidence
    capture. Metrics that reject stt-derived text
    (context_retention, question_repetition,
    technical_accuracy) keep skipping via their existing
    provenance gates.
    """

    name = "stt"

    def __init__(self, path: Path | None = None):
        self.path = Path(path or STT_LOCAL_PATH)

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def available(self) -> bool:
        return any(
            (entry.get("text") or "").strip()
            for entry in self._entries()
        )

    def get_turns(self) -> list[dict[str, Any]]:
        turns = []
        for entry in sorted(
            self._entries(),
            key=lambda item: (
                item.get("turn") or 0,
                # greeting/assistant before the same-numbered
                # candidate answer keeps conversational order.
                0 if item.get("role") == "user" else 1,
            ),
        ):
            text = (entry.get("text") or "").strip()
            role = entry.get("role")
            if not text or role not in ("assistant", "user"):
                continue
            turns.append(
                make_turn(
                    role=role,
                    text=text,
                    ts=None,
                    source="stt-local",
                    confidence="medium",
                )
            )
        return turns


# ============================================================
# Selection
# ============================================================

_BACKENDS = {
    "livekit": LiveKitEventsCapture,
    "app-api": AppApiCapture,
    "dom": DomScrapeCapture,
    "stt": SttCapture,
}

# stt (medium confidence) sits BELOW livekit/app-api (high) so
# it can never override them, and ABOVE the dom stopgap (whose
# assistant text is chrome-filtered to ~nothing).
AUTO_ORDER = ("livekit", "app-api", "stt", "dom")


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
