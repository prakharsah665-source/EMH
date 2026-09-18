"""
Socket.io audio transport adapter for the E2E evaluator.

The production stack (app.easemyhiring.ai -> room-api-v1) does
NOT use LiveKit/WebRTC for interview audio: bot speech arrives
as socket.io ``bot-audio-chunk`` events followed by binary
frames (the last one flagged ``isLastChunk:true``), the SPA
decodes and plays them via an AudioBufferSourceNode, and the
candidate's microphone is streamed back as ``user-audio-chunk``
binary frames. The QA stack (dev-qa -> room-api-v1-qa) keeps
using LiveKit; socket.io there carries state only.

This module adds a parallel capture/observation path so the
existing evaluator works on both stacks WITHOUT touching the
LiveKit path:

    SOCKETIO_AUDIO_HOOK_JS  - init script that wraps
        window.WebSocket for /socket.io/ URLs, assembles bot
        utterances from bot-audio-chunk binary frames (utterance
        boundary = isLastChunk:true), counts outgoing
        user-audio-chunk frames, and tracks the app's own
        playback window (outgoing interview-state-change
        BOT_SPEAKING -> USER_WAITING).
    SocketIOTransport       - python-side adapter: activity
        counters, utterance drain (reconstruct -> file -> local
        whisper STT), and the interviewer-question event stream
        for the candidate simulator.
    SocketIOBotWatcher      - same poll()/rebase() contract as
        the test's BotAudioWatcher, built from socket.io
        evidence instead of WebRTC stats/analysers.
    SocketIOAnswerSource    - LiveSimulatorAnswerSource over the
        transport's STT-derived interviewer events (the prod
        stack has no caption/data channel; whisper over the
        reconstructed bot audio is the only question source).

Evidence honesty: chunk bytes prove the browser RECEIVED bot
audio; the BOT_SPEAKING -> USER_WAITING cycle proves the app
DECODED AND PLAYED it; whisper text proves it was real speech.
Missing LiveKit/RTC data is expected on this transport and is
never an error.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from evaluation.redaction import redact_pii
from simulator.live_answers import LiveSimulatorAnswerSource

# Reconstructed bot utterances live beside the LiveKit-path
# .webm clips so the existing stt-local pipeline picks them up.
SIO_AUDIO_DIR = Path("artifacts/audio")
SIO_FILE_PREFIX = "bot_sio_"

# Same "real audio" floor the RTC stats use (~>1s of audio).
SIO_MIN_BYTES = 4_000

# A chunk within this window means the bot is actively sending.
SIO_CHUNK_ACTIVE_MS = 1_500


SOCKETIO_AUDIO_HOOK_JS = """
(() => {
  if (window.__emhSocketIO) return;
  const S = {
    sockets: 0, closed: 0,
    botChunks: 0, botBytes: 0, lastBotChunkTs: 0,
    userChunks: 0, userBytes: 0, lastUserChunkTs: 0,
    botSpeaking: false, playbackStartTs: 0,
    lastPlaybackEndTs: 0, playbackMsTotal: 0,
    utterancesCompleted: 0,
    utterances: [],
    stateEvents: [],
  };
  window.__emhSocketIO = S;

  let current = null;        // utterance being assembled
  const pendingBinary = [];  // FIFO: metadata for expected binary frames

  // base64 in 3-byte-aligned chunks so per-chunk btoa results
  // concatenate into one valid base64 string.
  const B64_CHUNK = 32766;
  const b64FromBuffer = (buf) => {
    const bytes = new Uint8Array(buf);
    let out = '';
    for (let i = 0; i < bytes.length; i += B64_CHUNK) {
      let bin = '';
      const end = Math.min(i + B64_CHUNK, bytes.length);
      for (let j = i; j < end; j++) bin += String.fromCharCode(bytes[j]);
      out += btoa(bin);
    }
    return out;
  };

  const noteState = (dir, text) => {
    if (S.stateEvents.length < 800) {
      S.stateEvents.push({ ts: Date.now(), dir, text: String(text).slice(0, 300) });
    }
  };

  const beginUtterance = () => {
    if (!current) {
      current = {
        index: S.utterances.length, startTs: Date.now(), endTs: null,
        chunkCount: 0, bytes: 0, b64: [], drained: false,
      };
    }
  };

  const onIncomingText = (text) => {
    // socket.io binary event header: 45<attachments>-["event",...]
    const m = /^45(\\d+)-(\\[[\\s\\S]*)$/.exec(text);
    if (m) {
      const attachments = parseInt(m[1], 10) || 1;
      const body = m[2];
      const isBotAudio = body.indexOf('"bot-audio-chunk"') !== -1;
      const isLast = /"isLastChunk"\\s*:\\s*true/.test(body);
      for (let i = 0; i < attachments; i++) {
        pendingBinary.push({
          bot: isBotAudio,
          isLastChunk: isBotAudio && isLast && i === attachments - 1,
        });
      }
      if (isBotAudio) beginUtterance();
      noteState('recv', text.slice(0, 200));
      return;
    }
    if (text.indexOf('interview-state-change') !== -1
        || text.indexOf('bot-speech-') !== -1
        || text.indexOf('sessionInitialised') !== -1) {
      noteState('recv', text);
    }
  };

  const onIncomingBinary = (buf) => {
    const meta = pendingBinary.shift();
    if (!meta || !meta.bot) return;
    const size = buf.byteLength || 0;
    S.botChunks += 1;
    S.botBytes += size;
    S.lastBotChunkTs = Date.now();
    beginUtterance();
    current.chunkCount += 1;
    current.bytes += size;
    try { current.b64.push(b64FromBuffer(buf)); } catch (e) {}
    if (meta.isLastChunk) {
      current.endTs = Date.now();
      S.utterances.push(current);
      S.utterancesCompleted += 1;
      current = null;
    }
  };

  const handleIncoming = (data) => {
    try {
      if (typeof data === 'string') onIncomingText(data);
      else if (data instanceof ArrayBuffer) onIncomingBinary(data);
      else if (data && typeof data.arrayBuffer === 'function') {
        // Blob: async read; frames of one utterance resolve in
        // arrival order because the FIFO metadata is consumed
        // inside onIncomingBinary.
        data.arrayBuffer().then(onIncomingBinary).catch(() => {});
      }
    } catch (e) {}
  };

  const handleOutgoing = (data) => {
    try {
      if (typeof data === 'string') {
        if (data.indexOf('interview-state-change') !== -1) {
          noteState('sent', data);
          if (data.indexOf('BOT_SPEAKING') !== -1 && !S.botSpeaking) {
            S.botSpeaking = true;
            S.playbackStartTs = Date.now();
          }
          if (data.indexOf('USER_WAITING') !== -1 && S.botSpeaking) {
            S.botSpeaking = false;
            S.lastPlaybackEndTs = Date.now();
            S.playbackMsTotal += S.lastPlaybackEndTs - S.playbackStartTs;
          }
        }
      } else {
        // Binary frames the app sends are the candidate's
        // user-audio-chunk payloads.
        const size = data.byteLength || data.size || 0;
        S.userChunks += 1;
        S.userBytes += size;
        S.lastUserChunkTs = Date.now();
      }
    } catch (e) {}
  };

  // Offline-test ingestion hook (no real socket needed).
  window.__emhSocketIOIngest = (dir, data) => (
    dir === 'recv' ? handleIncoming(data) : handleOutgoing(data)
  );

  const instrument = (ws) => {
    S.sockets += 1;
    ws.addEventListener('message', (e) => handleIncoming(e.data));
    ws.addEventListener('close', () => { S.closed += 1; });
    const origSend = ws.send.bind(ws);
    ws.send = (data) => { handleOutgoing(data); return origSend(data); };
  };

  const NativeWS = window.WebSocket;
  if (NativeWS) {
    const Wrapped = function (url, protocols) {
      const ws = protocols === undefined
        ? new NativeWS(url) : new NativeWS(url, protocols);
      try {
        if (String(url).indexOf('/socket.io/') !== -1) instrument(ws);
      } catch (e) {}
      return ws;
    };
    Wrapped.prototype = NativeWS.prototype;
    Object.setPrototypeOf(Wrapped, NativeWS);
    ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'].forEach((k) => {
      Wrapped[k] = NativeWS[k];
    });
    window.WebSocket = Wrapped;
  }

  window.__emhSocketIOSnapshot = () => ({
    now: Date.now(),
    sockets: S.sockets,
    closed: S.closed,
    botChunks: S.botChunks,
    botBytes: S.botBytes,
    lastBotChunkTs: S.lastBotChunkTs,
    userChunks: S.userChunks,
    userBytes: S.userBytes,
    lastUserChunkTs: S.lastUserChunkTs,
    botSpeaking: S.botSpeaking,
    playbackStartTs: S.playbackStartTs,
    lastPlaybackEndTs: S.lastPlaybackEndTs,
    playbackMsTotal: S.playbackMsTotal,
    utterancesCompleted: S.utterancesCompleted,
  });

  window.__emhSocketIODrain = () => {
    const out = [];
    for (const u of S.utterances) {
      if (u.drained) continue;
      u.drained = true;
      // Per-frame base64 strings: each frame was encoded
      // independently (its own padding), so they must be
      // DECODED separately and byte-concatenated - joining the
      // base64 text would corrupt the payload.
      out.push({
        index: u.index, startTs: u.startTs, endTs: u.endTs,
        chunkCount: u.chunkCount, bytes: u.bytes, b64Chunks: u.b64,
      });
      u.b64 = [];  // free memory - payload leaves the page once
    }
    return out;
  };
})();
"""


def sniff_audio_extension(data: bytes) -> str:
    """Container sniff for a reconstructed utterance payload."""

    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:4] == b"\x1aE\xdf\xa3":
        return ".webm"
    if data[:3] == b"ID3" or (
        len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    ):
        return ".mp3"
    # Whisper/PyAV probe by content; the extension is cosmetic.
    return ".mp3"


_WHISPER_MODEL = None


def transcribe_audio_file(path: Path | str) -> str | None:
    """
    Local faster-whisper STT of one audio file. Returns None
    when the model/library is unavailable (environment
    limitation, never an error) - shares the model/env knobs
    with collectors.transcript_capture.transcribe_captured_audio.
    """

    global _WHISPER_MODEL

    if os.getenv("EMH_DISABLE_STT_LOCAL") == "1":
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel(
            os.getenv("EMH_STT_MODEL", "base"), compute_type="int8"
        )
    segments, _info = _WHISPER_MODEL.transcribe(str(path), language="en")
    return " ".join(segment.text.strip() for segment in segments).strip()


class SocketIOTransport:
    """
    Python-side adapter over the in-page socket.io audio hook.

    ``active`` flips to True the moment bot-audio-chunk traffic
    is observed (transport detection); until then every consumer
    stays on the LiveKit path.
    """

    def __init__(
        self,
        audio_dir: Path = SIO_AUDIO_DIR,
        log=print,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.log = log
        self.active = False
        # Interviewer events shaped like __emhTranscriptEvents
        # string payloads so live_answers' extraction works
        # unchanged (see SocketIOAnswerSource).
        self.events: list[dict] = []
        self.records: list[dict] = []
        self._unmanifested: list[dict] = []
        self._stt_warned = False

    async def activity(self, page) -> dict:
        """Counter snapshot from the page hook; {} when absent."""

        try:
            snapshot = await page.evaluate(
                "() => window.__emhSocketIOSnapshot"
                " ? window.__emhSocketIOSnapshot() : null"
            )
        except Exception:
            snapshot = None
        return snapshot or {}

    async def drain(self, page) -> list[dict]:
        """
        Pull every COMPLETED (isLastChunk-terminated) bot
        utterance not drained yet: reconstruct the audio bytes,
        persist them beside the LiveKit-path clips, transcribe
        with local whisper, and append the text as an
        interviewer event for the candidate simulator. Returns
        the new records. Never raises for STT problems.
        """

        try:
            raw = await page.evaluate(
                "() => window.__emhSocketIODrain"
                " ? window.__emhSocketIODrain() : []"
            )
        except Exception:
            raw = []

        new_records: list[dict] = []
        for utterance in raw or []:
            try:
                data = b"".join(
                    base64.b64decode(chunk)
                    for chunk in (utterance.get("b64Chunks") or [])
                )
            except Exception:
                data = b""
            if not data:
                continue
            self.active = True
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            path = self.audio_dir / (
                f"{SIO_FILE_PREFIX}{utterance['index']:02d}"
                f"{sniff_audio_extension(data)}"
            )
            path.write_bytes(data)

            text = None
            try:
                text = await asyncio.to_thread(
                    transcribe_audio_file, path
                )
            except Exception as error:
                self.log(
                    "[socketio] whisper STT failed for "
                    f"{path.name} (environment issue): {error}"
                )
            if text is None and not self._stt_warned:
                self._stt_warned = True
                self.log(
                    "[socketio] faster-whisper unavailable - bot "
                    "utterances are captured but the LIVE QUESTION "
                    "cannot be derived on the socket.io transport."
                )
            if text:
                text = redact_pii(text)
                self.events.append(
                    {
                        "ev": "message",
                        "kind": "string",
                        "label": "socketio-stt",
                        "ts": utterance.get("endTs"),
                        "text": json.dumps(
                            {
                                "text": text,
                                "participantIdentity": "agent-socketio",
                            }
                        ),
                    }
                )

            record = {
                "index": utterance["index"],
                "path": str(path),
                "text": text,
                "ts": utterance.get("endTs"),
                "bytes": utterance.get("bytes"),
                "chunks": utterance.get("chunkCount"),
            }
            self.records.append(record)
            self._unmanifested.append(record)
            new_records.append(record)
            self.log(
                f"[socketio] bot utterance #{utterance['index']}: "
                f"{utterance.get('chunkCount')} chunk(s), "
                f"{utterance.get('bytes')} bytes -> {path.name}"
                + (
                    f", STT {len(text.split())} words"
                    if text else ", STT unavailable"
                )
            )
        return new_records

    def take_unmanifested(self) -> list[dict]:
        """Records not yet handed to the audio manifest."""

        taken, self._unmanifested = self._unmanifested, []
        return taken


class SocketIOBotWatcher:
    """
    Drop-in replacement for the test's BotAudioWatcher on the
    socket.io transport - same rebase()/poll() contract, so
    wait_for_bot_silence / the response-wait loops work
    unchanged. Detection channels:

      analyser_heard - a COMPLETED bot utterance arrived since
                       rebase (isLastChunk seen): the bot
                       definitively replied.
      stats_heard    - raw bot-audio-chunk bytes since rebase
                       crossed the real-audio floor (utterance
                       still in flight).
      active_now     - the app is inside a playback window
                       (BOT_SPEAKING -> USER_WAITING) or a chunk
                       arrived within the last 1.5s.
    """

    def __init__(self, page):
        self.page = page
        self.base_bytes = 0
        self.base_chunks = 0
        self.base_utterances = 0
        self.base_playback_ms = 0
        self.max_level = 0.0  # not measurable on this transport
        self.last_stats_activity_ts = 0.0

    async def _snap(self) -> dict:
        try:
            snapshot = await self.page.evaluate(
                "() => window.__emhSocketIOSnapshot"
                " ? window.__emhSocketIOSnapshot() : null"
            )
        except Exception:
            snapshot = None
        return snapshot or {}

    @staticmethod
    def _connection_states(snapshot: dict) -> list[dict]:
        """
        The room socket's state expressed like a WebRTC
        connectionState so room_gone() works unchanged: one
        "connected" entry while any instrumented /socket.io/
        socket is open, one "closed" entry once every socket has
        closed, [] before any socket existed.
        """

        sockets = snapshot.get("sockets") or 0
        if not sockets:
            return []
        closed = snapshot.get("closed") or 0
        state = "closed" if closed >= sockets else "connected"
        return [{"connection": state, "ice": state, "signaling": state}]

    @staticmethod
    def _playback_ms(snapshot: dict) -> float:
        total = snapshot.get("playbackMsTotal") or 0
        if snapshot.get("botSpeaking"):
            total += max(
                0.0,
                (snapshot.get("now") or 0)
                - (snapshot.get("playbackStartTs") or 0),
            )
        return total

    async def rebase(self) -> None:
        snapshot = await self._snap()
        self.base_bytes = snapshot.get("botBytes") or 0
        self.base_chunks = snapshot.get("botChunks") or 0
        self.base_utterances = snapshot.get("utterancesCompleted") or 0
        self.base_playback_ms = self._playback_ms(snapshot)
        self.max_level = 0.0
        self.last_stats_activity_ts = 0.0

    async def poll(self) -> dict:
        snapshot = await self._snap()
        now = snapshot.get("now") or (time.time() * 1000)

        bytes_delta = (snapshot.get("botBytes") or 0) - self.base_bytes
        packets_delta = (
            (snapshot.get("botChunks") or 0) - self.base_chunks
        )
        utterance_delta = (
            (snapshot.get("utterancesCompleted") or 0)
            - self.base_utterances
        )
        speech_ms = int(
            self._playback_ms(snapshot) - self.base_playback_ms
        )

        playing = bool(snapshot.get("botSpeaking"))
        last_chunk_ts = snapshot.get("lastBotChunkTs") or 0
        chunk_active = (
            last_chunk_ts > 0
            and now - last_chunk_ts < SIO_CHUNK_ACTIVE_MS
        )
        active_now = playing or chunk_active
        if active_now:
            self.last_stats_activity_ts = time.monotonic()

        last_activity_ts = (
            now if playing
            else max(
                last_chunk_ts,
                snapshot.get("lastPlaybackEndTs") or 0,
            )
        )

        # Shape-compatible with BotAudioWatcher.poll(): the RTC
        # fields are empty BY DESIGN on this transport (that is
        # expected data absence, not a failure signal).
        return {
            "bot": {
                "speechMs": speech_ms,
                "elementSpeechMs": 0,
                "trackSpeechMs": speech_ms,
                "lastSpeechTs": last_activity_ts or None,
                "lastElementSpeechTs": 0,
                "lastTrackSpeechTs": last_activity_ts or 0,
                "monitoredElements": 0,
            },
            "snapshot": {
                "ts": now,
                "pcCount": 0,
                # Socket state in the same vocabulary the drive
                # loop's room_gone() already understands, so a
                # server-side room teardown (socket closed, no
                # conclusion signal) triggers the existing
                # ROOM DISCONNECTED environment exit instead of
                # re-prompting into a dead room until the
                # wall-clock cap.
                "connectionStates": self._connection_states(snapshot),
                "remoteAudioTracks": [],
                "inboundAudio": [],
                "outboundAudio": [],
                "mediaSources": [],
                "remoteInboundAudio": [],
                "localMicTracks": [],
                "audioElements": [],
            },
            "totals": {
                "bytesReceived": snapshot.get("botBytes") or 0,
                "packetsReceived": snapshot.get("botChunks") or 0,
                "packetsLost": 0,
                "audioLevel": None,
                "totalAudioEnergy": None,
                "reportCount": 0,
            },
            "speech_ms": speech_ms,
            "bytes_delta": bytes_delta,
            "packets_delta": packets_delta,
            "energy_delta": None,
            "max_level": self.max_level,
            "analyser_heard": utterance_delta >= 1,
            "stats_heard": bytes_delta >= SIO_MIN_BYTES,
            "active_now": active_now,
            "element_ms": 0,
            "track_ms": speech_ms,
            "element_heard": False,
            "track_heard": False,
            "utterance_delta": utterance_delta,
            "sio": snapshot,
        }


class SocketIOAnswerSource(LiveSimulatorAnswerSource):
    """
    LiveSimulatorAnswerSource whose interviewer events come from
    the transport's whisper STT of reconstructed bot utterances
    (the socket.io stack publishes no caption/data channel).
    The cursor/stability/simulator logic is inherited unchanged.
    """

    name = "live-simulator-socketio"

    def __init__(self, simulator, transport: SocketIOTransport, **kwargs):
        super().__init__(simulator, **kwargs)
        self.transport = transport

    async def _events_since_cursor(self, page) -> list[dict]:
        await self.transport.drain(page)
        return self.transport.events[self.cursor:]
