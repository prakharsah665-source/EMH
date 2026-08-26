"""
Offline simulation of the full-interview drive loop
(tests/e2e/test_bot_responsiveness.run_multi_turn) with a fake
page/bot - no browser, no network.

Proves the evaluator STAYS IN THE ROOM until the AI interviewer
concludes: a stalled turn is recorded as a deferred failure,
recovered via late-reply / re-prompt, and the loop exits only on
the bot's conclusion signal (or the wall-clock safety cap).
"""

import asyncio
import time
from pathlib import Path

import pytest

from tests.e2e import test_bot_responsiveness as bot


# ------------------------------------------------------------
# Fake browser page: scripted bot behaviour per candidate turn
# ------------------------------------------------------------

class FakeBot:
    """
    behaviour[turn] is a list of what the bot does on each
    ATTEMPT of that turn:
      "reply"     - answers within the response window
      "silent"    - no audio at all in this attempt
      "late"      - silent in the response window, speaks during
                    the recovery wait
      "conclude"  - stays silent and emits the socket.io exit
                    signal (interview over)
    Anything past the scripted list repeats the last entry.
    """

    def __init__(self, behaviour: dict[int, list[str]], recorder):
        self.behaviour = behaviour
        self.recorder = recorder
        self.turn = 0
        self.attempt = 0
        self.speech_ms = 0
        self.mic_ms = 0
        self.bytes_in = 0
        self.packets_in = 0
        self.bytes_out = 0
        self.speaking_until = 0.0
        self.mode = None
        self.spoke_at = None
        self.body = "Interview room"
        self.injections: list[tuple[int, int]] = []
        self.room_closed = False

    def _tick(self):
        now = time.monotonic()
        # Bot speaking window -> accumulate bot audio.
        if self.mode in ("reply",) and self.spoke_at is not None:
            if now - self.spoke_at > 0.2:
                self.speech_ms += 400
                self.bytes_in += 5_000
                self.packets_in += 30
        if self.mode == "late" and self.spoke_at is not None:
            # silent during the response window, speaks after
            # ~1.6s (response timeout in the test is 1s)
            if now - self.spoke_at > 1.6:
                self.speech_ms += 400
                self.bytes_in += 5_000
                self.packets_in += 30
        if self.mode == "gone" and self.spoke_at is not None:
            if now - self.spoke_at > 0.5:
                self.room_closed = True
        if self.mode == "conclude" and self.spoke_at is not None:
            if now - self.spoke_at > 0.5 and not self.recorder.websocket_frames:
                self.recorder.websocket_frames.append(
                    {
                        "direction": "received",
                        "url": "wss://x/socket.io/?EIO=4",
                        "payload": '42["bot-speech-ended",{"isExit":true}]',
                        "ts": time.time() * 1000,
                    }
                )

    async def evaluate(self, expr, arg=None):
        self._tick()
        if "__emhSpeak" in expr:
            self.turn = len({t for t, _ in self.injections}) if False else self.turn
            self.injections.append((self.turn, self.attempt))
            self.mic_ms += 1500
            return 1500
        if "window.__emh.bot" in expr:
            return {
                "speechMs": self.speech_ms,
                "elementSpeechMs": 0,
                "trackSpeechMs": self.speech_ms,
                "lastSpeechTs": None,
            }
        if "window.__emh.mic" in expr:
            return {"speechMs": self.mic_ms}
        if "__emhRtcSnapshot" in expr:
            return {
                "ts": time.time() * 1000,
                "connectionStates": [
                    {"connection": "closed" if self.room_closed else "connected",
                     "ice": "closed" if self.room_closed else "connected",
                     "signaling": "closed" if self.room_closed else "stable"}
                ],
                "remoteAudioTracks": [
                    {
                        "id": "track-1",
                        "readyState": "ended" if self.room_closed else "live",
                        "everUnmuted": True,
                        "muted": False,
                        "streamIds": [],
                        "firstSeenTs": time.time() * 1000,
                    }
                ],
                "inboundAudio": [
                    {
                        "bytesReceived": self.bytes_in,
                        "packetsReceived": self.packets_in,
                        "packetsLost": 0,
                        "audioLevel": 0.2 if self.speech_ms else 0.0,
                        "totalAudioEnergy": None,
                    }
                ],
                "outboundAudio": [
                    {"bytesSent": self.bytes_out, "packetsSent": self.bytes_out // 100}
                ],
                "mediaSources": [{"audioLevel": 0.3}],
                "peerConnections": [],
            }
        if "__emhBotRec" in expr:
            return ""
        if "__emh.events" in expr:
            return []
        if "__emhTranscriptEvents" in expr:
            return []
        if "Date.now" in expr:
            return time.time() * 1000
        return None

    # page.locator("body").inner_text()
    def locator(self, _sel):
        fake = self

        class _Loc:
            async def inner_text(self_inner):
                return fake.body

        return _Loc()

    async def screenshot(self, **_):
        return None

    # Called by the test wrapper before each injection to script
    # the bot's behaviour for this attempt.
    def begin_attempt(self, turn: int, attempt: int):
        self.turn, self.attempt = turn, attempt
        script = self.behaviour.get(turn, ["reply"])
        self.mode = script[min(attempt - 1, len(script) - 1)]
        self.spoke_at = time.monotonic()
        self.bytes_out += 4_000  # candidate audio always publishes


class FakeContext:
    class _Tracing:
        async def stop(self, **_):
            return None

    tracing = _Tracing()


class SpyWatcher(bot.BotAudioWatcher):
    """Real watcher over the fake page; rebase() marks a new
    attempt so the fake bot can script its behaviour."""

    def __init__(self, page):
        super().__init__(page)
        self._attempt_by_turn: dict[int, int] = {}

    async def rebase(self):
        # run_multi_turn calls rebase() right before injecting.
        turn = self.page.turn_hint
        self._attempt_by_turn[turn] = self._attempt_by_turn.get(turn, 0) + 1
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
    return tmp_path


def _fixtures():
    return [Path("data/audio_fixtures/answer_01.wav"),
            Path("data/audio_fixtures/answer_02.wav")]


async def _drive(behaviour, tmp_path):
    recorder = bot.PipelineRecorder()
    page = FakeBot(behaviour, recorder)
    page.turn_hint = 0
    watcher = SpyWatcher(page)
    stages = bot.StageLog()
    collector = bot.TranscriptCollector()
    deferred: list[dict] = []
    status = {"complete": False, "turns_completed": 0, "reached_cap": False}
    turn_reports: list[dict] = []

    # run_multi_turn increments `turn` internally; mirror it via
    # the number of distinct wait_for_bot_silence calls per turn
    # is fragile, so hook the stage log instead: every
    # "Speaking candidate answer" stamp precedes rebase().
    orig_stamp = stages.stamp

    def stamp(msg):
        if "Speaking candidate answer" in msg:
            page.turn_hint = int(msg.split("[Turn ")[1].split("]")[0])
        orig_stamp(msg)

    stages.stamp = stamp

    result = await bot.run_multi_turn(
        page, FakeContext(), recorder, stages, tmp_path / "run",
        watcher, _fixtures(), True, turn_reports, collector, [],
        interview_status=status, audio_manifest=[], deferred=deferred,
    )
    return result, deferred, turn_reports, stages, page


@pytest.mark.asyncio
async def test_stays_in_room_through_stalls_until_bot_concludes(fast):
    # turn 1 answers; turn 2 stalls once then answers after the
    # re-prompt; turn 3 stalls in the response window but replies
    # late (recovery wait); turn 4: bot goes silent and emits the
    # exit signal -> conclusion.
    behaviour = {
        1: ["reply"],
        2: ["silent", "reply"],
        3: ["late"],
        4: ["conclude"],
    }
    result, deferred, reports, stages, page = await _drive(behaviour, fast)

    assert result["complete"] is True
    assert "socket.io exit signal" in result["conclusion_reason"]
    assert result["reached_cap"] is False
    # 3 completed answered turns; the 4th answer got the closing
    # signal instead of a reply (natural completion path).
    assert result["turns_completed"] >= 3
    # Two stalls (turn 2 attempt 1, turn 3 attempt 1), both recovered.
    labels = [(d["turn"], d["attempt"], d["recovered"]) for d in deferred]
    assert (2, 1, True) in labels
    assert (3, 1, True) in labels
    assert all("BOT STOPPED RESPONDING" in d["label"] for d in deferred)
    assert result["stalls"] == 2 and result["recoveries"] == 2
    # Turn 2 was re-prompted exactly once (2 attempts).
    assert sorted(a for t, a in page.injections if t == 2) == [1, 2]
    # Diagnostics preserved per stall (pipeline matrix in message).
    assert "Candidate mic" in deferred[0]["message"]
    # And the loop never raised (no pytest.fail mid-interview).
    timeline = "\n".join(stages.entries)
    assert "STAYING IN THE ROOM" in timeline
    assert "concluded" in timeline and "complete=True" in timeline


@pytest.mark.asyncio
async def test_never_stops_at_a_fixed_turn_count(fast):
    # 14 answered turns (> the old default caps) then conclusion.
    behaviour = {t: ["reply"] for t in range(1, 15)}
    behaviour[15] = ["conclude"]
    result, deferred, reports, stages, page = await _drive(behaviour, fast)
    assert result["complete"] is True
    assert result["turns_completed"] >= 14
    assert deferred == []


@pytest.mark.asyncio
async def test_dead_bot_leaves_only_on_wall_clock_cap(fast, monkeypatch):
    monkeypatch.setattr(bot, "INTERVIEW_MAX_S", 8)
    behaviour = {1: ["silent"]}  # never answers, never concludes
    started = time.monotonic()
    result, deferred, reports, stages, page = await _drive(behaviour, fast)
    elapsed = time.monotonic() - started
    assert result["complete"] is False
    assert result["reached_cap"] is True
    assert result["turns_completed"] == 0
    # It kept re-prompting inside the room until the cap.
    assert len(deferred) >= 2
    assert all(not d["recovered"] for d in deferred)
    assert elapsed >= 8


@pytest.mark.asyncio
async def test_server_room_teardown_is_environment_exit_not_45min_wait(fast, monkeypatch):
    monkeypatch.setattr(bot, "ROOM_DISCONNECT_GRACE_S", 2)
    monkeypatch.setattr(bot, "INTERVIEW_MAX_S", 120)
    # turn 1 answers, then the server tears the room down (no
    # conclusion signal) after the turn-2 answer.
    behaviour = {1: ["reply"], 2: ["gone"]}
    started = time.monotonic()
    result, deferred, reports, stages, page = await _drive(behaviour, fast)
    assert result["complete"] is False
    assert result["room_disconnected"] is True
    assert result["reached_cap"] is False
    assert result["turns_completed"] == 1
    assert time.monotonic() - started < 60
    assert bot.ROOM_DISCONNECTED in "\n".join(stages.entries)
