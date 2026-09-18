"""
Offline validation of the socket.io audio transport adapter
(collectors/socketio_capture.py) and its integration with the
bot-responsiveness drive loop.

Covers, without any network/backend:

  1. The in-page hook: bot-audio-chunk headers + binary frames
     are assembled into utterances (boundary = isLastChunk:true),
     outgoing user-audio-chunk frames are counted, and the
     BOT_SPEAKING -> USER_WAITING playback window is tracked
     (real headless Chromium, frames fed via the test ingestion
     hook __emhSocketIOIngest).
  2. Reconstruction + REAL local whisper STT of a bot utterance
     synthesized with macOS `say` (the exact question-derivation
     path the prod transport depends on).
  3. The interviewer-question event flow feeding the candidate
     simulator (SocketIOTransport.drain -> SocketIOAnswerSource
     -> extract_interviewer_question).
  4. run_multi_turn driven end-to-end over a fake socket.io page:
     turn detection, stall + re-prompt recovery, late reply,
     conclusion via the socket.io exit signal, transport-tagged
     pipeline verdicts, and the assistant audio manifest.
  5. validate_greeting transport detection: no RTC tracks +
     bot-audio-chunk traffic => socket.io greeting validation
     passes and never reports AGENT NEVER JOINED.
"""

import asyncio
import base64
import json
import subprocess
import time
from pathlib import Path

import pytest

import collectors.socketio_capture as sio_mod
from collectors.socketio_capture import (
    SOCKETIO_AUDIO_HOOK_JS,
    SocketIOBotWatcher,
    SocketIOTransport,
    sniff_audio_extension,
    transcribe_audio_file,
)
from simulator.live_answers import extract_interviewer_question
from tests.e2e import test_bot_responsiveness as bot


# ------------------------------------------------------------
# 0. Pure helpers
# ------------------------------------------------------------

def test_sniff_audio_extension():
    assert sniff_audio_extension(b"RIFF1234WAVE") == ".wav"
    assert sniff_audio_extension(b"OggS...") == ".ogg"
    assert sniff_audio_extension(b"\x1aE\xdf\xa3xxx") == ".webm"
    assert sniff_audio_extension(b"ID3\x04rest") == ".mp3"
    assert sniff_audio_extension(b"\xff\xf3\x82data") == ".mp3"
    assert sniff_audio_extension(b"unknownpayload") == ".mp3"


# ------------------------------------------------------------
# Shared: a real page with the hook installed, frames fed via
# the __emhSocketIOIngest test entry point.
# ------------------------------------------------------------

BOT_CHUNK_HEADER = (
    '451-["bot-audio-chunk",{"buffer":{"_placeholder":true,'
    '"num":0},"isLastChunk":false},false,false]'
)
BOT_CHUNK_HEADER_LAST = (
    '451-["bot-audio-chunk",{"buffer":{"_placeholder":true,'
    '"num":0},"isLastChunk":true},false,false]'
)

FEED_BINARY_JS = """(b64) => {
  const raw = atob(b64);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
  window.__emhSocketIOIngest('recv', arr.buffer);
}"""


async def _feed_utterance(page, payload: bytes, pieces: int = 3):
    """Feed one bot utterance as header+binary chunk frames."""

    size = max(1, len(payload) // pieces)
    chunks = [
        payload[i:i + size] for i in range(0, len(payload), size)
    ]
    for i, chunk in enumerate(chunks):
        header = (
            BOT_CHUNK_HEADER_LAST
            if i == len(chunks) - 1 else BOT_CHUNK_HEADER
        )
        await page.evaluate(
            "(t) => window.__emhSocketIOIngest('recv', t)", header
        )
        await page.evaluate(
            FEED_BINARY_JS,
            base64.b64encode(chunk).decode("ascii"),
        )


def _say_wav(tmp_path: Path, text: str) -> Path:
    """macOS say -> mono 22.05kHz WAV (skip if unavailable)."""

    aiff = tmp_path / "utt.aiff"
    wav = tmp_path / "utt.wav"
    try:
        subprocess.run(
            ["say", "-o", str(aiff), text],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050",
             "-c", "1", str(aiff), str(wav)],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("macOS say/afconvert unavailable")
    return wav


@pytest.mark.asyncio
async def test_hook_assembles_utterances_and_counters(tmp_path, monkeypatch):
    from playwright.async_api import async_playwright

    payload = b"RIFF" + bytes(range(256)) * 200  # ~51KB fake wav

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.add_init_script(SOCKETIO_AUDIO_HOOK_JS)
        await page.goto("data:text/html,<h1>hook</h1>")

        # Playback window + first utterance.
        await page.evaluate(
            "(t) => window.__emhSocketIOIngest('send', t)",
            '42["interview-state-change",{"state":"BOT_SPEAKING"}]',
        )
        await _feed_utterance(page, payload, pieces=3)
        await asyncio.sleep(0.05)
        await page.evaluate(
            "(t) => window.__emhSocketIOIngest('send', t)",
            '42["interview-state-change",{"state":"USER_WAITING"}]',
        )
        # Candidate audio going out (binary user-audio-chunk).
        await page.evaluate(
            """(b64) => {
              const raw = atob(b64);
              const arr = new Uint8Array(raw.length);
              for (let i = 0; i < raw.length; i++) {
                arr[i] = raw.charCodeAt(i);
              }
              window.__emhSocketIOIngest('send', arr.buffer);
            }""",
            base64.b64encode(b"\x00" * 10922).decode("ascii"),
        )

        snap = await page.evaluate("() => window.__emhSocketIOSnapshot()")
        assert snap["botChunks"] == 3
        assert snap["botBytes"] == len(payload)
        assert snap["utterancesCompleted"] == 1
        assert snap["botSpeaking"] is False
        assert snap["playbackMsTotal"] > 0
        assert snap["userChunks"] == 1
        assert snap["userBytes"] == 10922

        # Watcher contract over the live page state.
        watcher = SocketIOBotWatcher(page)
        await watcher.rebase()
        result = await watcher.poll()
        assert result["analyser_heard"] is False  # nothing new
        assert result["bytes_delta"] == 0

        # Second utterance arrives -> watcher hears a reply.
        await _feed_utterance(page, payload, pieces=2)
        result = await watcher.poll()
        assert result["analyser_heard"] is True
        assert result["utterance_delta"] == 1
        assert result["bytes_delta"] == len(payload)
        assert result["stats_heard"] is True
        # RTC-shaped fields exist and are empty (never an error).
        assert result["snapshot"]["remoteAudioTracks"] == []

        # Drain reconstructs both utterances byte-exactly.
        monkeypatch.setattr(
            sio_mod, "transcribe_audio_file", lambda p: "stub text"
        )
        transport = SocketIOTransport(
            audio_dir=tmp_path / "audio", log=lambda *_: None
        )
        records = await transport.drain(page)
        assert transport.active is True
        assert [r["index"] for r in records] == [0, 1]
        for record in records:
            data = Path(record["path"]).read_bytes()
            assert data == payload
            assert record["path"].endswith(".wav")  # RIFF sniffed
        # Idempotent: nothing left to drain.
        assert await transport.drain(page) == []

        await browser.close()


@pytest.mark.asyncio
async def test_reconstructed_utterance_real_whisper_stt(tmp_path):
    """
    The full prod question path: say-synthesized bot speech ->
    chunked bot-audio-chunk frames -> reconstruction -> REAL
    local faster-whisper STT.
    """

    pytest.importorskip("faster_whisper")
    from playwright.async_api import async_playwright

    wav = _say_wav(
        tmp_path,
        "Hi, my name is Jamie and I am your interviewer today.",
    )
    payload = wav.read_bytes()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.add_init_script(SOCKETIO_AUDIO_HOOK_JS)
        await page.goto("data:text/html,<h1>stt</h1>")
        await _feed_utterance(page, payload, pieces=4)

        transport = SocketIOTransport(
            audio_dir=tmp_path / "audio", log=lambda *_: None
        )
        records = await transport.drain(page)
        await browser.close()

    assert len(records) == 1
    text = records[0]["text"]
    if text is None:
        pytest.skip("faster-whisper model unavailable")
    assert "interviewer" in text.lower()
    # The event stream carries the question for the simulator.
    question, diagnostics = extract_interviewer_question(
        transport.events, previous_question=None
    )
    assert "interviewer" in question.lower()
    assert diagnostics["interviewer_words"] >= 4


# ------------------------------------------------------------
# 3. Question event flow with a stub page (no browser)
# ------------------------------------------------------------

class _StubDrainPage:
    def __init__(self, pending):
        self.pending = pending

    async def evaluate(self, expr, arg=None):
        if "__emhSocketIODrain" in expr:
            out, self.pending = self.pending, []
            return out
        if "__emhSocketIOSnapshot" in expr:
            return {"now": time.time() * 1000}
        if "__emhTranscriptEvents" in expr:
            return 0 if ".length" in expr else []
        return None


@pytest.mark.asyncio
async def test_answer_source_event_flow(tmp_path, monkeypatch):
    texts = iter(
        [
            "Tell me about your experience with JavaScript.",
            "How do you approach debugging a slow web page?",
        ]
    )
    monkeypatch.setattr(
        sio_mod, "transcribe_audio_file", lambda p: next(texts)
    )
    payload_b64 = base64.b64encode(b"RIFFfakeaudio").decode("ascii")
    page = _StubDrainPage(
        [
            {"index": 0, "startTs": 1, "endTs": 2, "chunkCount": 1,
             "bytes": 13, "b64Chunks": [payload_b64]},
        ]
    )
    transport = SocketIOTransport(
        audio_dir=tmp_path, log=lambda *_: None
    )

    # SocketIOAnswerSource reads events through the transport;
    # exercise the cursor slice + extraction exactly as
    # next_answer does (without the LLM call).
    source_events = None

    class _Source(sio_mod.SocketIOAnswerSource):
        def __init__(self, transport):
            # Bypass the simulator (not needed for this flow).
            sio_mod.LiveSimulatorAnswerSource.__init__(
                self, simulator=None, log=lambda *_: None
            )
            self.transport = transport

    source = _Source(transport)
    events = await source._events_since_cursor(page)
    question, diagnostics = extract_interviewer_question(
        events, previous_question=None
    )
    assert question == "Tell me about your experience with JavaScript."
    source.cursor += diagnostics["events_since_last_answer"]

    # Second utterance -> only NEW events are seen after the
    # cursor advances (stale/duplicate protection intact).
    page.pending = [
        {"index": 1, "startTs": 3, "endTs": 4, "chunkCount": 1,
         "bytes": 13, "b64Chunks": [payload_b64]},
    ]
    events = await source._events_since_cursor(page)
    question2, _ = extract_interviewer_question(
        events, previous_question=question
    )
    assert question2 == "How do you approach debugging a slow web page?"


# ------------------------------------------------------------
# 4. Drive loop over a fake socket.io page
# ------------------------------------------------------------

class FakeSocketIOPage:
    """
    Fake page for the socket.io transport: bot audio exists ONLY
    as socket.io counters/utterances; every RTC probe stays
    empty (as on prod). behaviour[turn] scripts each attempt:
    "reply" | "silent" | "late" | "conclude".
    """

    def __init__(self, behaviour, recorder):
        self.behaviour = behaviour
        self.recorder = recorder
        self.turn = 0
        self.attempt = 0
        self.turn_hint = 0
        self.mode = None
        self.spoke_at = None
        self.replied_this_attempt = False
        self.room_closed = False
        self.mic_ms = 0
        self.bot_bytes = 0
        self.bot_chunks = 0
        self.utterances_completed = 0
        self.playback_ms = 0
        self.last_chunk_ts = 0.0
        self.user_bytes = 0
        self.user_chunks = 0
        self.pending_drain = []
        self.body = "Interview room"
        self.injections = []

    def _now_ms(self):
        return time.time() * 1000

    def _tick(self):
        now = time.monotonic()
        delay = {"reply": 0.2, "late": 1.6}.get(self.mode)
        if (
            delay is not None
            and self.spoke_at is not None
            and not self.replied_this_attempt
            and now - self.spoke_at > delay
        ):
            self.replied_this_attempt = True
            index = self.utterances_completed
            self.utterances_completed += 1
            self.bot_chunks += 4
            self.bot_bytes += 16_000
            self.playback_ms += 500
            self.last_chunk_ts = self._now_ms()
            self.pending_drain.append(
                {
                    "index": index,
                    "startTs": self._now_ms() - 500,
                    "endTs": self._now_ms(),
                    "chunkCount": 4,
                    "bytes": 16_000,
                    "b64Chunks": [
                        base64.b64encode(
                            b"RIFF" + b"\x00" * 64
                        ).decode("ascii")
                    ],
                }
            )
        if (
            self.mode == "exit-state"
            and self.spoke_at is not None
            and time.monotonic() - self.spoke_at > 0.5
            and not any(
                "EXITING" in (f.get("payload") or "")
                for f in self.recorder.websocket_frames
            )
        ):
            # Prod exit dialect of run 20260902_122950: no
            # isExit string; the app announces EXITING and the
            # server confirms REPORT_GENERATED, then the socket
            # closes.
            self.recorder.websocket_frames.extend(
                [
                    {
                        "direction": "sent",
                        "url": "wss://x/socket.io/?EIO=4",
                        "payload": '42["interview-state-change",'
                                   '{"state":"EXITING"}]',
                        "ts": self._now_ms(),
                    },
                    {
                        "direction": "received",
                        "url": "wss://x/socket.io/?EIO=4",
                        "payload": '42["status","REPORT_GENERATED"]',
                        "ts": self._now_ms(),
                    },
                ]
            )
            self.room_closed = True
        if (
            self.mode == "gone"
            and self.spoke_at is not None
            and time.monotonic() - self.spoke_at > 0.5
        ):
            # Server tears the room down with NO conclusion
            # signal of any dialect.
            self.room_closed = True
        if (
            self.mode == "conclude"
            and self.spoke_at is not None
            and time.monotonic() - self.spoke_at > 0.5
            and not any(
                "isExit" in (f.get("payload") or "")
                for f in self.recorder.websocket_frames
            )
        ):
            self.recorder.websocket_frames.append(
                {
                    "direction": "received",
                    "url": "wss://x/socket.io/?EIO=4",
                    "payload": '42["bot-speech-ended",{"isExit":true}]',
                    "ts": self._now_ms(),
                }
            )

    async def evaluate(self, expr, arg=None):
        self._tick()
        if "__emhSocketIOSnapshot" in expr:
            return {
                "now": self._now_ms(),
                "sockets": 1,
                "closed": 1 if self.room_closed else 0,
                "botChunks": self.bot_chunks,
                "botBytes": self.bot_bytes,
                "lastBotChunkTs": self.last_chunk_ts,
                "userChunks": self.user_chunks,
                "userBytes": self.user_bytes,
                "botSpeaking": False,
                "playbackStartTs": 0,
                "lastPlaybackEndTs": self.last_chunk_ts,
                "playbackMsTotal": self.playback_ms,
                "utterancesCompleted": self.utterances_completed,
            }
        if "__emhSocketIODrain" in expr:
            out, self.pending_drain = self.pending_drain, []
            return out
        if "__emhSpeak" in expr:
            self.injections.append((self.turn, self.attempt))
            self.mic_ms += 1500
            self.user_chunks += 3
            self.user_bytes += 32_766
            # Server VAD ack for the candidate stream.
            self.recorder.websocket_frames.append(
                {
                    "direction": "received",
                    "url": "wss://x/socket.io/?EIO=4",
                    "payload": '42["interview-state-change","USER_SPEAKING"]',
                    "ts": self._now_ms(),
                }
            )
            return 1500
        if "window.__emh.bot" in expr:
            return {
                "speechMs": 0, "elementSpeechMs": 0,
                "trackSpeechMs": 0, "lastSpeechTs": None,
            }
        if "window.__emh.mic" in expr:
            return {"speechMs": self.mic_ms}
        if "__emhRtcSnapshot" in expr:
            # Prod reality: zero peer connections, empty stats.
            return {
                "ts": self._now_ms(), "pcCount": 0,
                "connectionStates": [], "remoteAudioTracks": [],
                "inboundAudio": [], "outboundAudio": [],
                "mediaSources": [], "remoteInboundAudio": [],
                "localMicTracks": [], "audioElements": [],
            }
        if "__emhBotRec" in expr:
            return ""
        if "__emh.events" in expr:
            return []
        if "__emhTranscriptEvents" in expr:
            return 0 if ".length" in expr else []
        if "Date.now" in expr:
            return self._now_ms()
        return None

    def locator(self, _sel):
        fake = self

        class _Loc:
            async def inner_text(self_inner):
                return fake.body

        return _Loc()

    async def screenshot(self, **_):
        return None

    def begin_attempt(self, turn, attempt):
        self.turn, self.attempt = turn, attempt
        script = self.behaviour.get(turn, ["reply"])
        self.mode = script[min(attempt - 1, len(script) - 1)]
        self.spoke_at = time.monotonic()
        self.replied_this_attempt = False


class FakeContext:
    class _Tracing:
        async def stop(self, **_):
            return None

    tracing = _Tracing()


class SpySocketWatcher(SocketIOBotWatcher):
    def __init__(self, page):
        super().__init__(page)
        self._attempt_by_turn = {}

    async def rebase(self):
        turn = self.page.turn_hint
        self._attempt_by_turn[turn] = (
            self._attempt_by_turn.get(turn, 0) + 1
        )
        self.page.begin_attempt(turn, self._attempt_by_turn[turn])
        await super().rebase()


@pytest.fixture
def fast(monkeypatch, tmp_path):
    monkeypatch.setattr(bot, "BOT_RESPONSE_TIMEOUT_S", 1)
    monkeypatch.setattr(bot, "BOT_RECOVERY_TIMEOUT_S", 3)
    monkeypatch.setattr(bot, "BOT_SILENCE_MS", 300)
    monkeypatch.setattr(bot, "BOT_UTTERANCE_MAX_S", 3)
    monkeypatch.setattr(bot, "INTERVIEW_MAX_S", 60)
    monkeypatch.setattr(bot, "MAX_TURNS", 0)
    monkeypatch.setattr(bot, "FORCED_TURNS", None)
    monkeypatch.setattr(bot, "POST_EXIT_GRACE_S", 0)
    monkeypatch.setattr(bot, "fixture_base64", lambda _p: "AAAA")
    monkeypatch.setattr(bot, "SCREENSHOT_DIR", tmp_path / "shots")
    monkeypatch.setattr(
        bot.TranscriptCollector, "save", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        sio_mod, "transcribe_audio_file", lambda p: "stub bot text"
    )
    return tmp_path


def _fixtures():
    return [Path("data/audio_fixtures/answer_01.wav"),
            Path("data/audio_fixtures/answer_02.wav")]


async def _drive_socketio(behaviour, tmp_path):
    recorder = bot.PipelineRecorder()
    page = FakeSocketIOPage(behaviour, recorder)
    watcher = SpySocketWatcher(page)
    transport = SocketIOTransport(
        audio_dir=tmp_path / "audio", log=lambda *_: None
    )
    transport.active = True
    stages = bot.StageLog()
    collector = bot.TranscriptCollector()
    deferred = []
    status = {"complete": False, "turns_completed": 0,
              "reached_cap": False}
    turn_reports = []
    audio_manifest = []

    orig_stamp = stages.stamp

    def stamp(msg):
        if "Speaking candidate answer" in msg:
            page.turn_hint = int(
                msg.split("[Turn ")[1].split("]")[0]
            )
        orig_stamp(msg)

    stages.stamp = stamp

    result = await bot.run_multi_turn(
        page, FakeContext(), recorder, stages, tmp_path / "run",
        watcher, _fixtures(), True, turn_reports, collector, [],
        interview_status=status, audio_manifest=audio_manifest,
        deferred=deferred, sio=transport,
    )
    return (result, deferred, turn_reports, stages, page,
            audio_manifest)


@pytest.mark.asyncio
async def test_socketio_drive_stalls_recovery_and_conclusion(fast):
    behaviour = {
        1: ["reply"],
        2: ["silent", "reply"],
        3: ["late"],
        4: ["conclude"],
    }
    (result, deferred, reports, stages, page,
     manifest) = await _drive_socketio(behaviour, fast)

    assert result["complete"] is True
    assert "socket.io exit signal" in result["conclusion_reason"]
    assert result["turns_completed"] >= 3
    labels = [(d["turn"], d["attempt"], d["recovered"]) for d in deferred]
    assert (2, 1, True) in labels
    assert (3, 1, True) in labels
    assert result["stalls"] == 2 and result["recoveries"] == 2

    # Transport-tagged pipeline evidence on an answered turn.
    answered = next(r for r in reports if r["turn"] == 1)
    st = answered["pipeline"]["stages"]
    assert "[transport=socketio]" in st["agent_publish"]["detail"]
    assert st["agent_publish"]["status"] == "PASS"
    assert "user-audio-chunk" in st["outbound_rtp"]["detail"]
    assert st["outbound_rtp"]["status"] == "PASS"
    assert "USER_SPEAKING" in st["livekit_receive"]["detail"]
    assert st["livekit_receive"]["status"] == "PASS"
    assert st["inbound_rtp"]["status"] == "PASS"
    assert st["audio_element"]["status"] == "PASS"

    # Bot utterances were reconstructed into the audio manifest.
    assistant_entries = [
        m for m in manifest if m["role"] == "assistant"
    ]
    assert assistant_entries, "no reconstructed bot audio manifested"
    assert all(
        Path(m["audio_path"]).exists() for m in assistant_entries
    )

    # No LiveKit-flavored misdiagnosis anywhere.
    all_text = "\n".join(d["message"] for d in deferred)
    assert bot.AGENT_NEVER_JOINED not in all_text
    assert "CANDIDATE AUDIO NOT PUBLISHED" not in all_text


@pytest.mark.asyncio
async def test_socketio_stall_is_bot_failure_not_transport_noise(fast):
    """A genuinely dead bot on socket.io still fails truthfully."""

    behaviour = {1: ["silent"]}
    import tests.e2e.test_bot_responsiveness as b
    orig = b.INTERVIEW_MAX_S
    b.INTERVIEW_MAX_S = 6
    try:
        (result, deferred, reports, stages, page,
         manifest) = await _drive_socketio(behaviour, fast)
    finally:
        b.INTERVIEW_MAX_S = orig
    assert result["complete"] is False
    assert deferred and all(
        "BOT STOPPED RESPONDING" in d["label"] for d in deferred
    )
    # The stall names socket.io evidence, not missing RTC data.
    assert "[transport=socketio]" in deferred[0]["message"]
    assert bot.AGENT_NEVER_JOINED not in deferred[0]["message"]


def test_socket_exit_markers_cover_prod_exit_dialects():
    """Both prod conclusion dialects must be recognized."""

    def rec_with(payload, direction="received", url="wss://x/socket.io/?EIO=4"):
        rec = bot.PipelineRecorder()
        rec.websocket_frames.append(
            {"direction": direction, "url": url,
             "payload": payload, "ts": 1.0}
        )
        return rec

    assert bot.socket_exit_signalled(bot.PipelineRecorder()) is None
    # Dialect 1: bot-speech-ended with isExit (run 20260902_113940).
    assert bot.socket_exit_signalled(
        rec_with('42["bot-speech-ended",{"isExit":true}]')
    )
    # Dialect 2: EXITING state + REPORT_GENERATED status
    # (run 20260902_122950 - no isExit string on the wire).
    assert bot.socket_exit_signalled(
        rec_with('42["interview-state-change",{"state":"EXITING"}]',
                 direction="sent")
    )
    assert bot.socket_exit_signalled(
        rec_with('42["status","REPORT_GENERATED"]')
    )
    # Mid-interview states never match.
    assert bot.socket_exit_signalled(
        rec_with('42["interview-state-change","USER_SPEAKING"]')
    ) is None
    # Non-socket.io frames are ignored.
    assert bot.socket_exit_signalled(
        rec_with('42["status","REPORT_GENERATED"]',
                 url="wss://x.livekit.cloud/rtc")
    ) is None


@pytest.mark.asyncio
async def test_socketio_exit_state_dialect_concludes_cleanly(fast):
    """
    The 20260902_122950 hang, replayed: the interviewer ends the
    interview via EXITING/REPORT_GENERATED (no isExit) - the
    drive must conclude instead of re-prompting into the closed
    room until the wall-clock cap.
    """

    behaviour = {1: ["reply"], 2: ["exit-state"]}
    started = time.monotonic()
    (result, deferred, reports, stages, page,
     manifest) = await _drive_socketio(behaviour, fast)
    elapsed = time.monotonic() - started

    assert result["complete"] is True
    assert "socket.io exit signal" in result["conclusion_reason"]
    assert (
        "EXITING" in result["conclusion_reason"]
        or "REPORT_GENERATED" in result["conclusion_reason"]
    )
    assert result["reached_cap"] is False
    assert deferred == []
    assert elapsed < 30


@pytest.mark.asyncio
async def test_socketio_room_close_exits_as_environment(fast, monkeypatch):
    """
    Socket closed with NO conclusion signal: the existing
    ROOM DISCONNECTED environment exit must fire (it was
    RTC-only before), not a 45-minute re-prompt crawl.
    """

    monkeypatch.setattr(bot, "ROOM_DISCONNECT_GRACE_S", 2)
    monkeypatch.setattr(bot, "INTERVIEW_MAX_S", 60)
    behaviour = {1: ["reply"], 2: ["gone"]}
    started = time.monotonic()
    (result, deferred, reports, stages, page,
     manifest) = await _drive_socketio(behaviour, fast)
    elapsed = time.monotonic() - started

    assert result["complete"] is False
    assert result["room_disconnected"] is True
    assert result["turns_completed"] == 1
    assert elapsed < 45
    assert bot.ROOM_DISCONNECTED in "\n".join(stages.entries)


def test_transcribe_captured_audio_reuses_cached_text(tmp_path, monkeypatch):
    """
    Manifest entries that already carry their whisper text
    (socket.io bot utterances) must not be re-transcribed in
    teardown - the model may not even be constructed.
    """

    import sys
    import types

    import collectors.transcript_capture as tc

    class BoomModel:
        def __init__(self, *a, **k):
            raise AssertionError(
                "WhisperModel constructed despite cached text"
            )

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = BoomModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(tc, "STT_LOCAL_PATH", tmp_path / "stt.json")
    monkeypatch.delenv("EMH_DISABLE_STT_LOCAL", raising=False)

    clip = tmp_path / "bot_sio_00.mp3"
    clip.write_bytes(b"\xff\xf3fake")
    entries = [
        {"role": "assistant", "turn": 0,
         "audio_path": str(clip), "text": "cached bot words"},
    ]
    summary = tc.transcribe_captured_audio(entries)
    assert summary["transcribed"] == 1
    assert summary["skipped_reason"] is None
    rows = json.loads((tmp_path / "stt.json").read_text())
    assert rows[0]["text"] == "cached bot words"
    assert rows[0]["role"] == "assistant"


# ------------------------------------------------------------
# 5. Greeting-phase transport detection
# ------------------------------------------------------------

class FakeGreetingPage(FakeSocketIOPage):
    """Bot greets over socket.io right away; no RTC ever."""

    def __init__(self, recorder):
        super().__init__({}, recorder)
        # Greeting already flowing when the wait loop starts.
        self.bot_chunks = 7
        self.bot_bytes = 113_295
        self.utterances_completed = 1
        self.playback_ms = 6_878
        self.last_chunk_ts = self._now_ms()


@pytest.mark.asyncio
async def test_validate_greeting_detects_socketio_transport(fast):
    recorder = bot.PipelineRecorder()
    page = FakeGreetingPage(recorder)
    watcher = bot.BotAudioWatcher(page)
    transport = SocketIOTransport(
        audio_dir=fast / "audio", log=lambda *_: None
    )
    stages = bot.StageLog()
    collector = bot.TranscriptCollector()
    deferred = []

    ok = await bot.validate_greeting(
        page, FakeContext(), recorder, stages, fast / "run",
        watcher, collector, deferred=deferred, sio=transport,
    )

    assert ok is True
    assert transport.active is True
    assert deferred == []
    timeline = "\n".join(stages.entries)
    assert "SOCKET.IO AUDIO TRANSPORT DETECTED" in timeline
    assert bot.AGENT_NEVER_JOINED not in timeline
    assert "PASSED - bot spoke first over socket.io" in timeline
