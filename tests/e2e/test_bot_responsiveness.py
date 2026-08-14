"""
Bot-responsiveness regression test for the EMH AI interviewer.

IMPORTANT FLOW BEING TESTED
---------------------------
The bot (agent "Jamie") speaks FIRST, immediately after the
candidate enters the interview room. The candidate does NOT need
to speak before the first bot response. Jamie's expected opening:

    "Hi, my name is Jamie and I am your interviewer today.
     Let's start with your introduction."

The test therefore has two clearly separated phases:

Phase 1 - TURN 0 / GREETING VALIDATION
    Without injecting any candidate audio, verify:
      1. the agent participant joined the LiveKit room
         (a remote audio track appears on the peer connection),
      2. the agent published an audio track (track + mute state),
      3. real bot audio is actually received, using WebRTC
         inbound-rtp stats (bytesReceived, packetsReceived,
         audioLevel, totalAudioEnergy) AND a WebAudio analyser
         on the app's <audio>/<video> elements.
    If the greeting never arrives, the failure is classified as
    one of:
        AGENT NEVER JOINED
        AGENT JOINED BUT NO AUDIO
        TTS/AUDIO PUBLISH FAILED
        BOT AUDIO RECEIVED BUT DETECTOR FAILED
    each with the likely root causes spelled out (worker not
    running/registered, dispatch/room mismatch, agent waiting for
    candidate VAD, TTS produced no audio, token/permission
    prevented publishing, or a test-harness detection problem).

Phase 2 - MULTI-TURN RESPONSIVENESS (6 turns by default)
    Only after the greeting is verified does the candidate start
    answering. Per turn: inject a synthesized spoken answer into
    the fake microphone, then require an audible bot reply within
    BOT_RESPONSE_TIMEOUT_S. A failure here is reported as
    "BOT STOPPED RESPONDING AT TURN N" - explicitly distinct from
    a greeting failure.

Candidate audio never comes from a human microphone: an init
script replaces navigator.mediaDevices.getUserMedia with a
WebAudio MediaStreamDestination the test feeds on demand, using
answers synthesized once with macOS `say` into
data/audio_fixtures/*.wav.

On any failure the test dumps: room name and identities (decoded
from the LiveKit access-token JWT in the signalling WebSocket
URL - the token itself is never printed), peer-connection states,
remote audio track list with mute state, inbound audio stats
(bytesReceived / packetsReceived / audioLevel / totalAudioEnergy),
agent/worker-looking console errors, per-stage timestamps, a
screenshot, a Playwright trace and all collected logs.

Every turn additionally produces a pipeline stage matrix that
isolates exactly where the audio pipeline fails:

    Candidate Mic -> Outbound RTP -> LiveKit receives audio
    -> STT transcript -> AI response -> TTS audio generated
    -> Agent publishes audio track -> Browser inbound RTP
    -> Correct audio element plays

Each stage gets an independent PASS/FAIL/WARN/UNKNOWN verdict
with a timestamp (STT/AI/TTS are observed via rendered
transcript text and inferred from downstream audio when not
directly visible; they run inside the agent worker). On failure
the report names the FIRST stage where data stops moving.

Run with the existing command:

    pytest tests/e2e/test_bot_responsiveness.py -s
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from collectors.transcript_collector import (
    TranscriptCollector,
    save_audio_turn_records,
    save_transcript_status,
)
from config.interview_session import (
    InterviewSessionError,
    mark_session_consumed,
    require_fresh_interview_url,
    require_unconsumed_session,
)
from config.settings import INTERVIEW_URL
from pages.interview_launch import (
    launch_into_interview_room as _launch_shared,
)


# ============================================================
# Configuration
# ============================================================

# The interview is driven to REAL completion (the bot signalling
# it is done), not a fixed question count. MAX_TURNS is a safety
# cap so a never-ending / looping agent cannot hang the test.
# EMH_INTERVIEW_TURNS, if set, still forces an exact turn count
# (kept for backwards-compatible single-question debugging).
MAX_TURNS = max(1, min(40, int(os.getenv("EMH_MAX_TURNS", "25"))))
_FORCED_TURNS = os.getenv("EMH_INTERVIEW_TURNS")
FORCED_TURNS = (
    max(1, min(MAX_TURNS, int(_FORCED_TURNS))) if _FORCED_TURNS else None
)

# Phrases with which the interviewer signals the interview is
# over. Kept specific so a mid-interview "thank you" does not
# end the drive-to-completion loop prematurely.
INTERVIEW_COMPLETE_RE = re.compile(
    r"this concludes|that concludes|concludes (our|the) interview"
    r"|interview is (now )?(complete|over|finished|ended)"
    r"|interview has (now )?ended"
    r"|no further questions from me"
    r"|thank you for (completing|participating in|your time today)"
    r"|end of (the|our) interview"
    r"|we have reached the end"
    r"|that (brings us to|is) the end of",
    re.IGNORECASE,
)

# Phase 1 timeouts (per stage, so a failure names the exact
# stage instead of hiding behind one generic timeout).
AGENT_JOIN_TIMEOUT_S = int(os.getenv("EMH_AGENT_JOIN_TIMEOUT", "60"))
GREETING_TIMEOUT_S = int(os.getenv("EMH_BOT_FIRST_SPEECH_TIMEOUT", "90"))

# Phase 2: how long the bot may take to reply to an answer.
BOT_RESPONSE_TIMEOUT_S = int(os.getenv("EMH_BOT_RESPONSE_TIMEOUT", "60"))

# The bot is considered finished speaking after this much
# continuous output silence.
BOT_SILENCE_MS = 2_500

# Minimum accumulated bot speech (ms of analyser frames above
# the energy threshold) to count as a real utterance, so a
# single noise blip does not pass as a response.
MIN_BOT_SPEECH_MS = 300

# Longest we will wait for the bot to finish one utterance.
BOT_UTTERANCE_MAX_S = 180

# WebRTC-stats thresholds for "real bot audio was received":
# roughly > 1 s of actual Opus audio, with a non-silent level.
STATS_MIN_BYTES = 4_000
STATS_MIN_PACKETS = 25
STATS_AUDIO_LEVEL = 0.01       # inbound-rtp audioLevel (0..1)
STATS_ENERGY_DELTA = 0.01      # inbound-rtp totalAudioEnergy delta

EXPECTED_GREETING = (
    "Hi, my name is Jamie and I am your interviewer today. "
    "Let's start with your introduction."
)

# Clear failure labels (turn 0).
AGENT_NEVER_JOINED = "AGENT NEVER JOINED"
AGENT_JOINED_BUT_NO_AUDIO = "AGENT JOINED BUT NO AUDIO"
TTS_AUDIO_PUBLISH_FAILED = "TTS/AUDIO PUBLISH FAILED"
DETECTOR_FAILED = "BOT AUDIO RECEIVED BUT DETECTOR FAILED"

FIXTURE_DIR = Path("data/audio_fixtures")
SCREENSHOT_DIR = Path("artifacts/screenshots")
DEBUG_ROOT = Path("artifacts/debug")
REPORT_DIR = Path("artifacts/reports")

# Console lines that look like the audio/STT/AI/TTS pipeline.
PIPELINE_LOG_PATTERN = re.compile(
    r"processAudioData|AudioStreamHandler|transcri|stt\b|tts\b"
    r"|speech|deepgram|whisper|livekit|websocket|socket"
    r"|agent|audio",
    re.IGNORECASE,
)

# Console lines that look like agent/worker/dispatch/permission
# problems - surfaced first in failure output.
AGENT_ERROR_PATTERN = re.compile(
    r"worker|dispatch|agent|token|permission|unauthoriz|forbidden"
    r"|publish|tts|synthes",
    re.IGNORECASE,
)

# Candidate answers spoken into the fake microphone AFTER the
# greeting, one per turn (cycled if TURNS exceeds the list).
CANDIDATE_ANSWERS = [
    "Hello Jamie, thank you. I am Alex, a software engineer with six years of experience, mostly building backend services in Python and Node.",
    "In my last role I led the migration of a monolith to microservices, which reduced our deployment time significantly.",
    "My biggest strength is debugging complex distributed systems, and I enjoy mentoring junior engineers.",
    "A challenging project I worked on was a real time analytics pipeline that processed millions of events per day.",
    "I usually approach problems by breaking them into smaller parts and validating each assumption with data.",
    "I collaborate closely with product managers and designers, and I value clear written communication.",
    "In five years I see myself leading a small engineering team while still contributing to the codebase.",
    "I handle disagreements by focusing on the shared goal and backing my position with evidence.",
    "I am motivated by shipping products that people actually use every day.",
    "That covers my experience. Thank you for the conversation, I have no further questions.",
]


def now_stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class StageLog:
    """Timestamped stage log printed live and dumped on failure."""

    def __init__(self):
        self.entries: list[str] = []

    def stamp(self, message: str) -> None:
        line = f"[{now_stamp()}] {message}"
        self.entries.append(line)
        print(line)


# ============================================================
# Audio fixtures (no human microphone)
# ============================================================

def ensure_answer_fixtures(count: int) -> list[Path]:
    """
    Synthesize spoken candidate answers as mono 48 kHz WAV files
    using macOS `say` + `afconvert`. Cached in
    data/audio_fixtures/ so generation happens once.
    """

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = []

    for index in range(count):
        text = CANDIDATE_ANSWERS[index % len(CANDIDATE_ANSWERS)]
        wav = FIXTURE_DIR / f"answer_{index + 1:02d}.wav"

        if not wav.exists():
            aiff = wav.with_suffix(".aiff")

            try:
                subprocess.run(
                    ["say", "-o", str(aiff), text],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [
                        "afconvert",
                        "-f", "WAVE",
                        "-d", "LEI16@48000",
                        "-c", "1",
                        str(aiff),
                        str(wav),
                    ],
                    check=True,
                    capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                pytest.skip(
                    "Could not synthesize candidate audio fixtures "
                    f"with macOS say/afconvert: {error}"
                )
            finally:
                aiff.unlink(missing_ok=True)

        fixtures.append(wav)

    return fixtures


def fixture_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ============================================================
# In-page instrumentation
#
# Installed with add_init_script so it runs before any
# application code on every navigation. Three responsibilities:
#   1. Fake candidate microphone (WebAudio destination fed by
#      window.__emhSpeak).
#   2. Bot playback energy detection (analysers on the app's
#      <audio>/<video> elements).
#   3. WebRTC instrumentation: hook RTCPeerConnection to record
#      remote audio tracks (agent joined / published / mute
#      state) and expose window.__emhRtcSnapshot() which reads
#      live inbound-rtp audio stats via getStats().
# ============================================================

INIT_SCRIPT = """
(() => {
    if (window.__emh) return;

    const state = {
        events: [],
        mic: { lastEnergyTs: 0, speechMs: 0, playing: false },
        bot: {
            lastSpeechTs: 0, speechMs: 0, monitoredElements: 0,
            // Split detection channels so pipeline stages can be
            // isolated: trackSpeechMs = decoded audio on the
            // remote WebRTC track (agent RTP arrived),
            // elementSpeechMs = playback energy on the app's
            // <audio>/<video> elements (candidate can hear it).
            elementSpeechMs: 0, lastElementSpeechTs: 0,
            trackSpeechMs: 0, lastTrackSpeechTs: 0,
        },
        rtc: { pcCount: 0, remoteAudioTracks: [] },
    };
    window.__emh = state;

    const ev = (type, detail) => {
        state.events.push({ ts: Date.now(), type, detail: detail || null });
        if (state.events.length > 2000) state.events.shift();
    };

    // ------------------------------------------------------
    // 1. Candidate microphone injection
    // ------------------------------------------------------

    let micCtx = null;
    let micDest = null;
    let micAnalyser = null;

    function ensureMic() {
        if (micCtx) return;
        micCtx = new (window.AudioContext || window.webkitAudioContext)();
        micDest = micCtx.createMediaStreamDestination();
        micAnalyser = micCtx.createAnalyser();
        micAnalyser.fftSize = 2048;
        ev('mic_created');
    }

    // Play a base64 WAV into the fake microphone. Resolves with
    // the clip duration (ms) once playback finishes.
    window.__emhSpeak = async (base64Wav) => {
        ensureMic();
        if (micCtx.state === 'suspended') await micCtx.resume();

        const raw = atob(base64Wav);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

        const buffer = await micCtx.decodeAudioData(bytes.buffer);

        return new Promise((resolve) => {
            const source = micCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(micDest);
            source.connect(micAnalyser);
            state.mic.playing = true;
            ev('candidate_audio_start', {
                durationMs: Math.round(buffer.duration * 1000),
            });
            source.onended = () => {
                state.mic.playing = false;
                ev('candidate_audio_end');
                resolve(Math.round(buffer.duration * 1000));
            };
            source.start();
        });
    };

    const realGetUserMedia =
        navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

    // Every getUserMedia call gets its OWN clone of the fake
    // mic track. The app opens the microphone more than once
    // (LiveKit publish + its report recorder) and stops tracks
    // during cleanup - with a single shared track instance one
    // consumer's track.stop() would silence the published
    // LiveKit track and the bot would never hear the candidate.
    const handedMicTracks = [];

    navigator.mediaDevices.getUserMedia = async (constraints) => {
        if (!constraints || !constraints.audio) {
            return realGetUserMedia(constraints);
        }

        ensureMic();
        const micTrack = micDest.stream.getAudioTracks()[0].clone();
        handedMicTracks.push(micTrack);
        ev('getUserMedia_intercepted', {
            video: !!constraints.video,
            cloneIndex: handedMicTracks.length - 1,
        });

        if (constraints.video) {
            const videoStream =
                await realGetUserMedia({ video: constraints.video });
            return new MediaStream([
                ...videoStream.getVideoTracks(),
                micTrack,
            ]);
        }

        return new MediaStream([micTrack]);
    };

    window.__emhLocalMicTracks = () =>
        handedMicTracks.map((track, index) => ({
            index,
            id: track.id,
            readyState: track.readyState,
            enabled: track.enabled,
            muted: track.muted,
        }));

    // ------------------------------------------------------
    // 2. Bot playback monitoring
    //
    // LiveKit attaches the remote agent's audio to an <audio>
    // (or <video>) element. Measure actual output energy via an
    // analyser: a live-but-silent remote track keeps the element
    // "playing", so paused/currentTime checks cannot detect a
    // bot that went mute.
    // ------------------------------------------------------

    let outCtx = null;
    const elementAnalysers = [];
    const trackAnalysers = [];
    const monitored = new WeakSet();

    function monitorElement(el) {
        if (monitored.has(el)) return;
        try {
            const stream =
                el.srcObject instanceof MediaStream
                    ? el.srcObject
                    : (el.captureStream ? el.captureStream() : null);
            if (!stream || stream.getAudioTracks().length === 0) return;

            if (!outCtx) {
                outCtx =
                    new (window.AudioContext || window.webkitAudioContext)();
            }
            const source = outCtx.createMediaStreamSource(stream);
            const analyser = outCtx.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);
            elementAnalysers.push(analyser);
            monitored.add(el);
            state.bot.monitoredElements += 1;
            ev('bot_element_monitored', { tag: el.tagName });
        } catch (error) {
            ev('bot_monitor_error', { message: String(error) });
        }
    }

    // Also measure decoded receive-side audio straight from a
    // remote WebRTC track (headless Chrome sometimes yields
    // silence when capturing from the app's <audio> element).
    function monitorRemoteTrack(track) {
        try {
            if (!outCtx) {
                outCtx =
                    new (window.AudioContext || window.webkitAudioContext)();
            }
            const source = outCtx.createMediaStreamSource(
                new MediaStream([track])
            );
            const analyser = outCtx.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);
            trackAnalysers.push(analyser);
            ev('remote_track_monitored', { id: track.id });
        } catch (error) {
            ev('remote_track_monitor_error', { message: String(error) });
        }
    }

    setInterval(() => {
        document.querySelectorAll('audio, video').forEach(monitorElement);
        if (outCtx && outCtx.state === 'suspended') {
            outCtx.resume().catch(() => {});
        }
        if (micCtx && micCtx.state === 'suspended') {
            micCtx.resume().catch(() => {});
        }
    }, 1000);

    const timeData = new Float32Array(2048);
    const ENERGY_THRESHOLD = 0.01;
    const PUMP_MS = 100;

    function rmsOf(analyser) {
        analyser.getFloatTimeDomainData(timeData);
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) {
            sum += timeData[i] * timeData[i];
        }
        return Math.sqrt(sum / timeData.length);
    }

    setInterval(() => {
        const now = Date.now();
        let heard = false;
        for (const analyser of elementAnalysers) {
            if (rmsOf(analyser) > ENERGY_THRESHOLD) {
                state.bot.lastElementSpeechTs = now;
                state.bot.elementSpeechMs += PUMP_MS;
                heard = true;
                break;
            }
        }
        for (const analyser of trackAnalysers) {
            if (rmsOf(analyser) > ENERGY_THRESHOLD) {
                state.bot.lastTrackSpeechTs = now;
                state.bot.trackSpeechMs += PUMP_MS;
                heard = true;
                break;
            }
        }
        if (heard) {
            state.bot.lastSpeechTs = now;
            state.bot.speechMs += PUMP_MS;
        }
        if (micAnalyser && state.mic.playing
                && rmsOf(micAnalyser) > ENERGY_THRESHOLD) {
            state.mic.lastEnergyTs = now;
            state.mic.speechMs += PUMP_MS;
        }
    }, PUMP_MS);

    // Inventory of the app's audio/video elements and the
    // tracks attached to them - lets the test verify the agent
    // track actually reached an element that should play it.
    window.__emhAudioElements = () =>
        Array.from(document.querySelectorAll('audio, video')).map((el) => {
            const stream =
                el.srcObject instanceof MediaStream ? el.srcObject : null;
            return {
                tag: el.tagName,
                paused: el.paused,
                elementMuted: el.muted,
                volume: el.volume,
                currentTime: el.currentTime,
                audioTracks: stream
                    ? stream.getAudioTracks().map((t) => ({
                          id: t.id,
                          muted: t.muted,
                          enabled: t.enabled,
                          readyState: t.readyState,
                      }))
                    : [],
            };
        });

    // ------------------------------------------------------
    // 3. WebRTC instrumentation
    //
    // A remote audio track on a peer connection is the ground
    // truth that the agent participant joined AND published
    // audio; getStats() inbound-rtp is the ground truth that
    // audio bytes are actually being received.
    // ------------------------------------------------------

    const pcs = [];
    const NativePC = window.RTCPeerConnection;

    window.RTCPeerConnection = function (...args) {
        const pc = new NativePC(...args);
        pcs.push(pc);
        state.rtc.pcCount = pcs.length;
        ev('pc_created');

        pc.addEventListener('track', (event) => {
            const track = event.track;
            if (track.kind !== 'audio') return;

            const info = {
                id: track.id,
                streamIds: event.streams.map((s) => s.id),
                muted: track.muted,
                readyState: track.readyState,
                firstSeenTs: Date.now(),
                everUnmuted: !track.muted,
            };
            state.rtc.remoteAudioTracks.push(info);
            ev('remote_audio_track', {
                id: track.id,
                streams: info.streamIds,
                muted: track.muted,
            });
            track.addEventListener('unmute', () => {
                info.muted = false;
                info.everUnmuted = true;
                ev('remote_track_unmute', { id: track.id });
            });
            track.addEventListener('mute', () => {
                info.muted = true;
                ev('remote_track_mute', { id: track.id });
            });
            track.addEventListener('ended', () => {
                info.readyState = 'ended';
                ev('remote_track_ended', { id: track.id });
            });

            monitorRemoteTrack(track);
        });

        return pc;
    };
    window.RTCPeerConnection.prototype = NativePC.prototype;
    Object.setPrototypeOf(window.RTCPeerConnection, NativePC);

    window.__emhRtcSnapshot = async () => {
        const inboundAudio = [];
        const outboundAudio = [];
        const mediaSources = [];
        const remoteInboundAudio = [];
        const connectionStates = [];

        for (const pc of pcs) {
            connectionStates.push({
                connection: pc.connectionState,
                ice: pc.iceConnectionState,
                signaling: pc.signalingState,
            });
            let stats;
            try {
                stats = await pc.getStats();
            } catch (error) {
                continue;
            }
            stats.forEach((report) => {
                if (
                    report.type === 'inbound-rtp'
                    && (report.kind === 'audio'
                        || report.mediaType === 'audio')
                ) {
                    inboundAudio.push({
                        trackIdentifier: report.trackIdentifier || null,
                        bytesReceived: report.bytesReceived || 0,
                        packetsReceived: report.packetsReceived || 0,
                        packetsLost: report.packetsLost || 0,
                        audioLevel:
                            typeof report.audioLevel === 'number'
                                ? report.audioLevel
                                : null,
                        totalAudioEnergy:
                            typeof report.totalAudioEnergy === 'number'
                                ? report.totalAudioEnergy
                                : null,
                    });
                }
                if (
                    report.type === 'outbound-rtp'
                    && (report.kind === 'audio'
                        || report.mediaType === 'audio')
                ) {
                    outboundAudio.push({
                        trackIdentifier: report.trackIdentifier || null,
                        mediaSourceId: report.mediaSourceId || null,
                        bytesSent: report.bytesSent || 0,
                        packetsSent: report.packetsSent || 0,
                    });
                }
                if (
                    report.type === 'remote-inbound-rtp'
                    && (report.kind === 'audio'
                        || report.mediaType === 'audio')
                ) {
                    // RTCP receiver report from the LiveKit SFU
                    // about OUR outbound audio - its existence
                    // proves the server is receiving the
                    // candidate's stream.
                    remoteInboundAudio.push({
                        packetsLost: report.packetsLost || 0,
                        jitter:
                            typeof report.jitter === 'number'
                                ? report.jitter
                                : null,
                        roundTripTime:
                            typeof report.roundTripTime === 'number'
                                ? report.roundTripTime
                                : null,
                    });
                }
                if (
                    report.type === 'media-source'
                    && report.kind === 'audio'
                ) {
                    mediaSources.push({
                        trackIdentifier: report.trackIdentifier || null,
                        audioLevel:
                            typeof report.audioLevel === 'number'
                                ? report.audioLevel
                                : null,
                        totalAudioEnergy:
                            typeof report.totalAudioEnergy === 'number'
                                ? report.totalAudioEnergy
                                : null,
                    });
                }
            });
        }

        return {
            ts: Date.now(),
            pcCount: pcs.length,
            connectionStates,
            remoteAudioTracks: state.rtc.remoteAudioTracks,
            inboundAudio,
            outboundAudio,
            mediaSources,
            remoteInboundAudio,
            localMicTracks: window.__emhLocalMicTracks(),
            audioElements: window.__emhAudioElements(),
        };
    };
})();
"""


# ============================================================
# Log collection
# ============================================================

def mask_tokens(text: str) -> str:
    # The interview URL itself embeds a session JWT - never let
    # it into logs or failure output.
    if INTERVIEW_URL:
        text = text.replace(INTERVIEW_URL, "<INTERVIEW_URL>")
    return re.sub(
        r"(token|access_token|authorization)=[^&\s\"']+",
        r"\1=***REDACTED***",
        text,
        flags=re.IGNORECASE,
    )


def decode_livekit_claims(websocket_url: str) -> dict:
    """
    The LiveKit signalling URL carries the access token as a JWT
    query parameter. Its payload (room name, participant
    identity, video grants) is plain base64 - decode it for
    diagnostics. The token itself is never stored or printed.
    """

    try:
        query = urllib.parse.urlsplit(websocket_url).query
        params = urllib.parse.parse_qs(query)
        token = (
            params.get("access_token", [])
            or params.get("token", [])
            or [None]
        )[0]
        if not token or token.count(".") != 2:
            return {}
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        video = claims.get("video", {}) or {}
        return {
            "room": video.get("room"),
            "identity": claims.get("sub"),
            "name": claims.get("name"),
            "can_publish": video.get("canPublish"),
            "can_subscribe": video.get("canSubscribe"),
        }
    except Exception:
        return {}


class PipelineRecorder:
    """
    Collects console messages, page errors and WebSocket frame
    activity with wall-clock timestamps, plus LiveKit room /
    identity / grant claims decoded from the signalling URL.

    chrome-extension:// noise goes to a separate bucket and is
    never used as pipeline evidence.
    """

    def __init__(self):
        self.console: list[dict] = []
        self.extension_noise: list[dict] = []
        self.page_errors: list[dict] = []
        self.websocket_frames: list[dict] = []
        self.websockets: list[str] = []
        self.livekit_claims: list[dict] = []

    def attach(self, page):
        page.on("console", self._on_console)
        page.on("pageerror", self._on_page_error)
        page.on("websocket", self._on_websocket)

    def _on_console(self, message):
        location = message.location or {}
        entry = {
            "ts": time.time() * 1000,
            "type": message.type,
            "text": mask_tokens(message.text)[:1000],
            "url": location.get("url", ""),
        }
        if "chrome-extension://" in entry["url"] \
                or "chrome-extension://" in entry["text"]:
            self.extension_noise.append(entry)
        else:
            self.console.append(entry)

    def _on_page_error(self, error):
        self.page_errors.append(
            {"ts": time.time() * 1000, "error": str(error)[:2000]}
        )

    def _on_websocket(self, websocket):
        safe_url = mask_tokens(websocket.url.split("?")[0])
        self.websockets.append(safe_url)

        if "livekit" in websocket.url:
            claims = decode_livekit_claims(websocket.url)
            if claims:
                self.livekit_claims.append(claims)

        def frame(direction):
            def handler(payload):
                if isinstance(payload, bytes):
                    body = f"<binary {len(payload)} bytes>"
                else:
                    body = mask_tokens(str(payload))[:300]
                self.websocket_frames.append(
                    {
                        "ts": time.time() * 1000,
                        "direction": direction,
                        "url": safe_url,
                        "payload": body,
                    }
                )
            return handler

        websocket.on("framesent", frame("sent"))
        websocket.on("framereceived", frame("received"))

    # --------------------------------------------------------

    @property
    def room_name(self) -> str:
        for claims in self.livekit_claims:
            if claims.get("room"):
                return claims["room"]
        return "<unknown - no LiveKit token seen>"

    @property
    def local_identity(self) -> str:
        for claims in self.livekit_claims:
            if claims.get("identity"):
                return claims["identity"]
        return "<unknown>"

    def livekit_frame_count(self) -> int:
        return sum(
            1 for frame in self.websocket_frames
            if "livekit" in frame["url"]
        )

    def no_active_stream_warnings(self) -> list[dict]:
        return [
            entry for entry in self.console
            if "no active audio stream" in entry["text"].lower()
        ]

    def fatal_app_warnings_since(self, start_ms: float) -> list[dict]:
        """
        App warnings that prove the pipeline is dead - waiting
        out the full response timeout after one of these appears
        only wastes time and blurs the diagnosis.
        """

        fatal = ("no agent response after silence",
                 "no active audio stream")
        return [
            entry for entry in self.console
            if entry["ts"] >= start_ms
            and any(marker in entry["text"].lower() for marker in fatal)
        ]

    def agent_error_lines(self, limit: int = 15) -> list[dict]:
        lines = [
            entry for entry in self.console
            if entry["type"] in ("error", "warning")
            and AGENT_ERROR_PATTERN.search(entry["text"])
        ]
        return lines[-limit:]

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)

        def dump(name, rows):
            (directory / name).write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False) for row in rows
                )
            )

        dump("console.jsonl", self.console)
        dump("extension_noise.jsonl", self.extension_noise)
        dump("page_errors.jsonl", self.page_errors)
        dump("websocket_frames.jsonl", self.websocket_frames)
        dump("livekit_claims.jsonl", self.livekit_claims)
        (directory / "websockets.txt").write_text(
            "\n".join(self.websockets)
        )


# ============================================================
# Page-state polling helpers
# ============================================================

async def page_now(page) -> float:
    return await page.evaluate("Date.now()")


async def bot_state(page) -> dict:
    return await page.evaluate("window.__emh.bot")


async def mic_state(page) -> dict:
    return await page.evaluate("window.__emh.mic")


async def rtc_snapshot(page) -> dict:
    return await page.evaluate("window.__emhRtcSnapshot()")


def inbound_totals(snapshot: dict) -> dict:
    """Aggregate inbound-rtp audio stats across peer connections."""

    reports = snapshot.get("inboundAudio", [])
    levels = [
        r["audioLevel"] for r in reports if r["audioLevel"] is not None
    ]
    energies = [
        r["totalAudioEnergy"] for r in reports
        if r["totalAudioEnergy"] is not None
    ]
    return {
        "bytesReceived": sum(r["bytesReceived"] for r in reports),
        "packetsReceived": sum(r["packetsReceived"] for r in reports),
        "packetsLost": sum(r["packetsLost"] for r in reports),
        "audioLevel": max(levels) if levels else None,
        "totalAudioEnergy": sum(energies) if energies else None,
        "reportCount": len(reports),
    }


def livekit_participants(snapshot: dict) -> dict:
    """
    Derive LiveKit participants and their published audio tracks
    from the WebRTC stream IDs (LiveKit encodes them as
    "PA_<participantSid>|TR_<trackSid>"). The per-track muted
    flag mirrors TrackPublication.isMuted as observed by the
    subscriber. The livekit-client Room object is not exposed on
    window by the app, so this derived view is the browser-side
    ground truth.
    """

    participants: dict[str, list] = {}
    for track in snapshot.get("remoteAudioTracks", []):
        for stream_id in track["streamIds"]:
            if stream_id.startswith("PA_") and "|" in stream_id:
                participant_sid, track_sid = stream_id.split("|", 1)
                participants.setdefault(participant_sid, []).append(
                    {
                        "trackSid": track_sid,
                        "isMuted": track["muted"],
                        "everUnmuted": track["everUnmuted"],
                        "readyState": track["readyState"],
                    }
                )
    return participants


def outbound_totals(snapshot: dict) -> dict:
    """
    Aggregate outbound (candidate -> LiveKit) audio stats:
    outbound-rtp proves packets left the browser, media-source
    audioLevel proves the PUBLISHED track carried real energy
    (not just that some local fake-mic node had audio).
    """

    reports = snapshot.get("outboundAudio", [])
    sources = snapshot.get("mediaSources", [])
    levels = [
        s["audioLevel"] for s in sources if s["audioLevel"] is not None
    ]
    return {
        "bytesSent": sum(r["bytesSent"] for r in reports),
        "packetsSent": sum(r["packetsSent"] for r in reports),
        "sourceAudioLevel": max(levels) if levels else None,
    }


# ============================================================
# Per-turn pipeline stage matrix
#
#   Candidate Mic -> Outbound RTP -> LiveKit receives audio
#   -> STT transcript -> AI response -> TTS audio generated
#   -> Agent publishes audio track -> Browser inbound RTP
#   -> Correct audio element plays
#
# Every stage gets an independent PASS/FAIL/WARN/UNKNOWN verdict
# with a timestamp, so a failure names the FIRST stage where
# data stops moving instead of a generic timeout.
# ============================================================

PIPELINE_STAGES = [
    ("candidate_mic", "Candidate mic (fake-mic energy)"),
    ("outbound_rtp", "Outbound RTP (bytesSent/packetsSent)"),
    ("livekit_receive", "LiveKit/agent receives audio (RTCP)"),
    ("stt", "STT transcript"),
    ("ai", "AI response"),
    ("tts", "TTS audio generated"),
    ("agent_publish", "Agent publishes audio track"),
    ("inbound_rtp", "Browser inbound RTP (bytesReceived)"),
    ("audio_element", "Browser playback (decoded agent audio)"),
]

STAGE_TITLES = dict(PIPELINE_STAGES)

# STT/AI/TTS run inside the agent worker; the browser can only
# observe them indirectly (rendered transcript text, console/ws
# events). UNKNOWN means "no observable signal", never a proven
# failure - but if a LATER stage passed, earlier UNKNOWN stages
# are inferred PASS (audio out implies STT/AI/TTS happened).
class TurnPipeline:
    def __init__(self, turn: int):
        self.turn = turn
        self.stages: dict[str, dict] = {}

    def record(self, key: str, status: str, detail: str = "") -> None:
        self.stages[key] = {
            "status": status,
            "ts": now_stamp(),
            "detail": detail,
        }

    def verdict(self, key: str, passed: bool, detail: str = "") -> None:
        self.record(key, "PASS" if passed else "FAIL", detail)

    def finalize(self) -> None:
        order = [key for key, _ in PIPELINE_STAGES]
        # Only stages that prove PER-TURN data movement may anchor
        # inference. agent_publish is standing state (the track has
        # been published since turn 0), so it proves nothing about
        # this turn's STT/AI/TTS.
        anchors = {
            "candidate_mic", "outbound_rtp", "livekit_receive",
            "stt", "ai", "inbound_rtp", "audio_element",
        }
        pass_indexes = [
            index for index, key in enumerate(order)
            if key in anchors
            and self.stages.get(key, {}).get("status") == "PASS"
        ]
        last_pass = max(pass_indexes, default=-1)
        for index, key in enumerate(order):
            stage = self.stages.get(key)
            if (
                index < last_pass
                and stage
                and stage["status"] == "UNKNOWN"
            ):
                stage["status"] = "PASS (inferred)"
                stage["detail"] = (
                    "a later pipeline stage carried data, so this "
                    "stage must have worked. " + stage["detail"]
                ).strip()

    def first_failure(self) -> str | None:
        """Key of the first stage with a proven FAIL."""

        for key, _ in PIPELINE_STAGES:
            if self.stages.get(key, {}).get("status") == "FAIL":
                return key
        return None

    def first_unconfirmed(self) -> str | None:
        """
        Key of the first stage where data movement can no longer
        be CONFIRMED (FAIL, UNKNOWN or WARN). When agent-internal
        stages (STT/AI/TTS) are unobservable and downstream audio
        is dead, this points at where the break must lie.
        """

        for key, _ in PIPELINE_STAGES:
            if self.stages.get(key, {}).get("status") in (
                "FAIL", "UNKNOWN", "WARN"
            ):
                return key
        return None

    def lines(self) -> list[str]:
        rows = [f"Pipeline stage matrix (turn {self.turn}):"]
        for key, title in PIPELINE_STAGES:
            stage = self.stages.get(
                key, {"status": "N/A", "ts": "", "detail": ""}
            )
            rows.append(
                f"  {title:<40} {stage['status']:<15} "
                f"[{stage['ts']}] {stage['detail']}"
            )
        return rows

    def as_dict(self) -> dict:
        return {"turn": self.turn, "stages": self.stages}


# --- Transcript evidence (STT / AI stages) -------------------
#
# The interview room renders conversation text (verified: the
# greeting text appears in the page body). Diffing body text
# around a turn gives browser-observable evidence that STT
# transcribed the answer and that the AI produced a response.

def significant_words(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z']+", text.lower())
            if len(word) > 3]


async def body_text_lines(page) -> set[str]:
    try:
        text = await page.locator("body").inner_text()
    except Exception:
        return set()
    return {
        line.strip() for line in text.splitlines() if line.strip()
    }


async def body_new_lines(page, before: set[str]) -> list[str]:
    """New page lines since `before`, in DOM order, deduplicated."""

    try:
        text = await page.locator("body").inner_text()
    except Exception:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and line not in before and line not in seen:
            seen.add(line)
            ordered.append(line)
    return ordered


async def detect_interview_complete(page) -> bool:
    """
    True when the interviewer has signalled the interview is
    over (a specific closing phrase in the rendered text, or an
    interview-complete UI state). Used to drive the interview to
    REAL completion instead of a fixed question count.
    """

    try:
        body = await page.locator("body").inner_text()
    except Exception:
        return False

    if INTERVIEW_COMPLETE_RE.search(body):
        return True

    # Common end-of-interview UI copy (kept specific).
    lowered = body.lower()
    return any(
        marker in lowered
        for marker in (
            "interview complete",
            "interview completed",
            "your interview is complete",
            "you may now close this window",
            "the interview has ended",
        )
    )


def transcript_evidence(
    new_lines: list[str], answer_text: str
) -> dict:
    new_text = " ".join(new_lines).lower()

    answer_vocabulary = significant_words(answer_text)
    matched = [
        word for word in answer_vocabulary if word in new_text
    ]
    stt_seen = (
        len(answer_vocabulary) > 0
        and len(matched) >= max(3, len(answer_vocabulary) * 0.3)
    )

    # AI evidence: substantial new text that is NOT just the
    # echoed candidate answer.
    ai_lines = [
        line for line in new_lines
        if len(line) >= 40
        and (
            not significant_words(line)
            or sum(
                1 for word in significant_words(line)
                if word in answer_vocabulary
            ) / len(significant_words(line)) < 0.5
        )
    ]
    # Lines dominated by the candidate's answer vocabulary are
    # the app's own rendered STT of the candidate's speech.
    stt_lines = [
        line for line in new_lines
        if significant_words(line)
        and sum(
            1 for word in significant_words(line)
            if word in answer_vocabulary
        ) / len(significant_words(line)) >= 0.5
    ]
    return {
        "stt_seen": stt_seen,
        "stt_matched_words": len(matched),
        "stt_vocabulary": len(answer_vocabulary),
        "ai_seen": len(ai_lines) > 0,
        "ai_sample": ai_lines[0][:120] if ai_lines else None,
        "ai_lines": ai_lines,
        "stt_lines": stt_lines,
        "new_line_count": len(new_lines),
    }


class BotAudioWatcher:
    """
    Detects bot speech through two independent channels:

      analyser - WebAudio energy on the app's <audio>/<video>
                 elements (what the candidate actually hears),
      stats    - WebRTC inbound-rtp deltas (bytes/packets flowing
                 with a non-silent audioLevel/totalAudioEnergy).

    Stats are the ground truth that audio was RECEIVED; the
    analyser is the ground truth that it was PLAYED. Keeping both
    lets a failure distinguish "bot silent" from "bot spoke but
    the Playwright-side detector failed".
    """

    def __init__(self, page):
        self.page = page
        self.base_speech_ms = 0
        self.base_element_ms = 0
        self.base_track_ms = 0
        self.base_bytes = 0
        self.base_packets = 0
        self.base_energy = None
        self.max_level = 0.0
        self.last_stats_activity_ts = 0.0

    async def rebase(self) -> None:
        bot = await bot_state(self.page)
        totals = inbound_totals(await rtc_snapshot(self.page))
        self.base_speech_ms = bot["speechMs"]
        self.base_element_ms = bot["elementSpeechMs"]
        self.base_track_ms = bot["trackSpeechMs"]
        self.base_bytes = totals["bytesReceived"]
        self.base_packets = totals["packetsReceived"]
        self.base_energy = totals["totalAudioEnergy"]
        self.max_level = 0.0
        self.last_stats_activity_ts = 0.0

    async def poll(self) -> dict:
        bot = await bot_state(self.page)
        snapshot = await rtc_snapshot(self.page)
        totals = inbound_totals(snapshot)

        if totals["audioLevel"] is not None:
            self.max_level = max(self.max_level, totals["audioLevel"])

        bytes_delta = totals["bytesReceived"] - self.base_bytes
        packets_delta = totals["packetsReceived"] - self.base_packets

        # inbound-rtp reports may not exist yet at rebase time;
        # anchor the energy baseline on first sight so deltas are
        # measured from the start of the observation window.
        if totals["totalAudioEnergy"] is not None and self.base_energy is None:
            self.base_energy = totals["totalAudioEnergy"]
        energy_delta = None
        if (
            totals["totalAudioEnergy"] is not None
            and self.base_energy is not None
        ):
            energy_delta = totals["totalAudioEnergy"] - self.base_energy

        speech_ms = bot["speechMs"] - self.base_speech_ms
        element_ms = bot["elementSpeechMs"] - self.base_element_ms
        track_ms = bot["trackSpeechMs"] - self.base_track_ms
        analyser_heard = speech_ms >= MIN_BOT_SPEECH_MS

        packets_flowing = (
            bytes_delta >= STATS_MIN_BYTES
            and packets_delta >= STATS_MIN_PACKETS
        )
        # Non-silent evidence: an observed audioLevel above the
        # silence floor, or accumulated audio energy. If Chrome
        # exposes neither field, fall back to bytes alone.
        level_known = (
            self.max_level > 0 or energy_delta is not None
        )
        non_silent = (
            self.max_level >= STATS_AUDIO_LEVEL
            or (energy_delta is not None
                and energy_delta >= STATS_ENERGY_DELTA)
        )
        stats_heard = packets_flowing and (non_silent or not level_known)

        # Instantaneous activity: inbound-rtp audioLevel is the
        # level of the most recent receive window, so it tells us
        # whether the bot is speaking RIGHT NOW (used for
        # stats-based end-of-utterance / silence detection).
        active_now = (
            totals["audioLevel"] is not None
            and totals["audioLevel"] >= STATS_AUDIO_LEVEL
        ) or (
            energy_delta is not None
            and energy_delta >= STATS_ENERGY_DELTA
        )
        if active_now:
            self.last_stats_activity_ts = time.monotonic()
        if energy_delta is not None and energy_delta >= STATS_ENERGY_DELTA:
            # Rebase energy so the next poll measures fresh
            # activity, but keep max_level accumulating.
            self.base_energy = totals["totalAudioEnergy"]

        return {
            "bot": bot,
            "snapshot": snapshot,
            "totals": totals,
            "speech_ms": speech_ms,
            "bytes_delta": bytes_delta,
            "packets_delta": packets_delta,
            "energy_delta": energy_delta,
            "max_level": self.max_level,
            "analyser_heard": analyser_heard,
            "stats_heard": stats_heard,
            "active_now": active_now,
            "element_ms": element_ms,
            "track_ms": track_ms,
            "element_heard": element_ms >= MIN_BOT_SPEECH_MS,
            "track_heard": track_ms >= MIN_BOT_SPEECH_MS,
        }


async def wait_for_bot_silence(
    page, watcher: BotAudioWatcher
) -> None:
    """
    Wait until the bot has been silent for BOT_SILENCE_MS, i.e.
    it finished its current utterance, so the candidate never
    barges in over the bot mid-question (which can break the
    agent's turn-taking). Activity is measured on BOTH channels:
    WebAudio analyser recency AND live inbound-rtp audioLevel -
    either one indicating speech keeps the wait going. Bounded by
    BOT_UTTERANCE_MAX_S; if the cap is hit the test proceeds to
    answer (speaking over an endless utterance).
    """

    deadline = time.monotonic() + BOT_UTTERANCE_MAX_S

    # "Silent" means no bot activity for BOT_SILENCE_MS measured
    # from the start of this wait - a bot that is ALREADY silent
    # must release the turn quickly instead of stalling until
    # BOT_UTTERANCE_MAX_S.
    silence_since = time.monotonic()

    while time.monotonic() < deadline:
        result = await watcher.poll()
        bot = result["bot"]
        page_ts = result["snapshot"]["ts"]
        analyser_active = (
            bot["lastSpeechTs"]
            and page_ts - bot["lastSpeechTs"] < 1_000
        )
        if result["active_now"] or analyser_active:
            silence_since = time.monotonic()
        if (time.monotonic() - silence_since) * 1000 >= BOT_SILENCE_MS:
            return
        await asyncio.sleep(0.5)

    print(
        f"[{now_stamp()}] [WARNING] Bot did not go silent within "
        f"{BOT_UTTERANCE_MAX_S}s - answering anyway."
    )


# ============================================================
# Interview launch flow (same steps as the other E2E tests)
# ============================================================

async def launch_into_interview_room(page, stages: StageLog) -> None:
    """
    Delegate to the shared page-object launch flow so this
    test and every other E2E test drive the interview
    identically (Start/Continue, joyride, one-session lock,
    fresh-session guard). The fake-mic check inside the shared
    helper runs automatically because this test injects
    window.__emhSpeak.
    """
    await _launch_shared(page, log=stages.stamp)


# ============================================================
# Diagnostics dump / failure capture
# ============================================================

def format_rtc_diagnostics(
    snapshot: dict, recorder: PipelineRecorder
) -> list[str]:
    totals = inbound_totals(snapshot)
    tracks = snapshot.get("remoteAudioTracks", [])

    lines = [
        f"Room name:            {recorder.room_name}",
        f"Local identity:       {recorder.local_identity}",
        (
            "LiveKit token grants: "
            + ", ".join(
                f"{c.get('identity')}"
                f" (canPublish={c.get('can_publish')},"
                f" canSubscribe={c.get('can_subscribe')})"
                for c in recorder.livekit_claims
            )
            if recorder.livekit_claims
            else "LiveKit token grants: <no LiveKit signalling URL seen>"
        ),
        f"Peer connections:     {snapshot.get('pcCount', 0)} "
        f"{snapshot.get('connectionStates', [])}",
        f"Remote audio tracks:  {len(tracks)}",
    ]
    for track in tracks:
        lines.append(
            f"  - track {track['id']}"
            f" streams={track['streamIds']}"
            f" muted={track['muted']}"
            f" everUnmuted={track['everUnmuted']}"
            f" readyState={track['readyState']}"
            f" firstSeen={datetime.fromtimestamp(track['firstSeenTs'] / 1000):%H:%M:%S}"
        )
    participants = livekit_participants(snapshot)
    lines.append(
        "LiveKit participants and audio publications (derived "
        "from WebRTC stream SIDs; isMuted = TrackPublication "
        "mute state as observed by the subscriber):"
    )
    if participants:
        for participant_sid, tracks in participants.items():
            lines.append(f"  - {participant_sid}:")
            for publication in tracks:
                lines.append(
                    f"      {publication['trackSid']}"
                    f" isMuted={publication['isMuted']}"
                    f" everUnmuted={publication['everUnmuted']}"
                    f" readyState={publication['readyState']}"
                )
    else:
        lines.append("  <no remote participants observed>")
    lines += [
        "Inbound audio stats (all peer connections):",
        f"  bytesReceived:    {totals['bytesReceived']}",
        f"  packetsReceived:  {totals['packetsReceived']}",
        f"  packetsLost:      {totals['packetsLost']}",
        f"  audioLevel:       {totals['audioLevel']}",
        f"  totalAudioEnergy: {totals['totalAudioEnergy']}",
        "Per-track inbound RTP reports:",
        *[
            f"  {report}"
            for report in snapshot.get("inboundAudio", [])
        ],
        "Per-track outbound RTP reports:",
        *[
            f"  {report}"
            for report in snapshot.get("outboundAudio", [])
        ],
    ]
    out = outbound_totals(snapshot)
    lines += [
        "Outbound audio stats (candidate -> LiveKit):",
        f"  bytesSent:        {out['bytesSent']}",
        f"  packetsSent:      {out['packetsSent']}",
        f"  sourceAudioLevel: {out['sourceAudioLevel']}",
        "LiveKit RTCP receiver reports for our outbound audio "
        "(presence proves the server receives the candidate):",
        f"  {snapshot.get('remoteInboundAudio', [])}",
        "Audio elements and attached tracks:",
    ]
    for element in snapshot.get("audioElements", []):
        lines.append(
            f"  - <{element['tag'].lower()}> paused={element['paused']}"
            f" elementMuted={element['elementMuted']}"
            f" volume={element['volume']}"
            f" currentTime={round(element['currentTime'], 1)}"
            f" tracks={element['audioTracks']}"
        )
    lines += [
        "Local mic tracks handed to the app "
        "(clones of the fake microphone):",
    ]
    for track in snapshot.get("localMicTracks", []):
        lines.append(
            f"  - clone {track['index']} id={track['id']}"
            f" readyState={track['readyState']}"
            f" enabled={track['enabled']} muted={track['muted']}"
        )
    lines += [
        f"LiveKit ws frames:    {recorder.livekit_frame_count()}",
        f"All websockets:       {recorder.websockets}",
        "'no active audio stream' warnings from the app: "
        f"{len(recorder.no_active_stream_warnings())}",
    ]
    return lines


async def capture_failure(
    page,
    context,
    recorder: PipelineRecorder,
    stages: StageLog,
    run_dir: Path,
    label: str,
    reason: str,
    likely_causes: list[str],
    turn_reports: list[dict] | None = None,
    pipeline: TurnPipeline | None = None,
) -> str:
    """
    Save screenshot, trace and all collected logs, and build a
    failure message with the classification label first and full
    diagnostics after it.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    screenshot = SCREENSHOT_DIR / "bot_responsiveness_failed.png"
    try:
        await page.screenshot(path=str(screenshot), full_page=True)
    except Exception:
        screenshot = None

    trace_path = run_dir / "trace.zip"
    try:
        await context.tracing.stop(path=str(trace_path))
    except Exception:
        trace_path = None

    recorder.save(run_dir)

    try:
        snapshot = await rtc_snapshot(page)
    except Exception:
        snapshot = {}

    try:
        events = await page.evaluate("window.__emh.events")
        (run_dir / "page_events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events)
        )
    except Exception:
        pass

    (run_dir / "turn_report.json").write_text(
        json.dumps(turn_reports or [], indent=2)
    )
    (run_dir / "stages.txt").write_text("\n".join(stages.entries))

    if pipeline is not None:
        pipeline.finalize()
        (run_dir / "pipeline_matrix.json").write_text(
            json.dumps(pipeline.as_dict(), indent=2)
        )

    lines = [
        f"{label}: {reason}",
        "",
        *(
            [
                *pipeline.lines(),
                "",
                "FIRST PIPELINE STAGE WITH A PROVEN FAILURE: "
                + STAGE_TITLES.get(
                    pipeline.first_failure() or "",
                    "<none identified>",
                ),
                "FIRST STAGE WHERE DATA FLOW IS NO LONGER "
                "CONFIRMED: "
                + STAGE_TITLES.get(
                    pipeline.first_unconfirmed() or "",
                    "<none identified>",
                ),
                "",
            ]
            if pipeline is not None else []
        ),
        "Likely causes:",
        *[f"  - {cause}" for cause in likely_causes],
        "",
        "--- Room / participant / audio diagnostics ---",
        *format_rtc_diagnostics(snapshot, recorder),
        "",
        "--- Stage timeline ---",
        *[f"  {entry}" for entry in stages.entries],
        "",
        f"Artifacts: {run_dir}",
        f"Screenshot: {screenshot}",
        f"Trace: {trace_path} (open with: playwright show-trace)",
    ]

    agent_errors = recorder.agent_error_lines()
    if agent_errors:
        lines += ["", "Agent/worker-looking console errors:"]
        for entry in agent_errors:
            when = datetime.fromtimestamp(entry["ts"] / 1000)
            lines.append(
                f"  [{when:%H:%M:%S}] [{entry['type']}] "
                f"{entry['text'][:250]}"
            )

    pipeline_lines = [
        entry for entry in recorder.console
        if PIPELINE_LOG_PATTERN.search(entry["text"])
    ]
    lines += ["", "Last pipeline-related console lines:"]
    for entry in pipeline_lines[-15:]:
        lines.append(f"  [{entry['type']}] {entry['text'][:200]}")

    lines += ["", "Last WebSocket frames:"]
    for frame in recorder.websocket_frames[-10:]:
        lines.append(
            f"  {frame['direction']:>8} {frame['url']} "
            f"{frame['payload'][:120]}"
        )

    if recorder.page_errors:
        lines += ["", "Page errors:"]
        for error in recorder.page_errors[-5:]:
            lines.append(f"  {error['error'][:300]}")

    lines += [
        "",
        f"(Ignored {len(recorder.extension_noise)} "
        "chrome-extension:// console entries.)",
    ]

    return "\n".join(lines)


# ============================================================
# Phase 1: Turn 0 - greeting validation
#
# The bot must speak FIRST. No candidate audio is injected here.
# ============================================================

async def validate_greeting(
    page,
    context,
    recorder: PipelineRecorder,
    stages: StageLog,
    run_dir: Path,
    watcher: BotAudioWatcher,
    collector: TranscriptCollector,
) -> bool:
    """
    Verify Jamie's automatic greeting. Returns True if the
    analyser heard the greeting, False if only WebRTC stats
    confirmed it (detector failed - later turns fall back to
    stats-based detection). Calls pytest.fail with a classified
    message if the greeting never arrived.
    """

    stages.stamp(
        "[Turn 0] Expecting the bot to speak FIRST - no candidate "
        "audio will be injected before the greeting."
    )
    stages.stamp(f"[Turn 0] Expected opening: \"{EXPECTED_GREETING}\"")

    body_before_greeting = await body_text_lines(page)
    await watcher.rebase()

    pipe = TurnPipeline(0)
    for key in ("candidate_mic", "outbound_rtp", "livekit_receive", "stt"):
        pipe.record(
            key, "N/A",
            "bot speaks first - no candidate audio required",
        )
    pipe.record("ai", "UNKNOWN", "scripted greeting")
    pipe.record(
        "tts", "UNKNOWN", "not directly observable from the browser"
    )

    # ----------------------------------------------------
    # Stage A: the agent participant must join (a remote
    # audio track appears on the peer connection).
    # ----------------------------------------------------

    stages.stamp(
        f"[Turn 0] Waiting up to {AGENT_JOIN_TIMEOUT_S}s for the "
        "agent participant to join (remote audio track)..."
    )

    join_deadline = time.monotonic() + AGENT_JOIN_TIMEOUT_S
    agent_joined = False
    while time.monotonic() < join_deadline:
        snapshot = await rtc_snapshot(page)
        if snapshot.get("remoteAudioTracks"):
            agent_joined = True
            break
        await asyncio.sleep(1.0)

    if not agent_joined:
        livekit_alive = recorder.livekit_frame_count() > 0
        pipe.verdict(
            "agent_publish", False, "no remote audio track appeared"
        )
        pipe.verdict("inbound_rtp", False, "no agent track to receive")
        pipe.verdict("audio_element", False, "no agent audio to play")
        message = await capture_failure(
            page, context, recorder, stages, run_dir,
            label=AGENT_NEVER_JOINED,
            reason=(
                "No remote audio track appeared within "
                f"{AGENT_JOIN_TIMEOUT_S}s of entering the interview "
                "room - no agent participant published anything. "
                + (
                    "The LiveKit room connection itself is live "
                    "(signalling frames observed), so the candidate "
                    "side is fine."
                    if livekit_alive else
                    "No LiveKit signalling frames were observed "
                    "either - the room connection itself may have "
                    "failed before the agent question even arises."
                )
            ),
            likely_causes=[
                "The agent worker is not running or not registered "
                "with the LiveKit server.",
                "Agent dispatch does not match this room "
                f"(room: {recorder.room_name}) - check the dispatch "
                "rule / agent name against the room the frontend "
                "created.",
                "The agent process crashed on startup - check the "
                "agent worker logs for this room name.",
            ],
            pipeline=pipe,
        )
        pytest.fail(message)

    snapshot = await rtc_snapshot(page)
    stages.stamp(
        "[Turn 0] Agent participant joined - remote audio track(s): "
        + json.dumps(snapshot["remoteAudioTracks"])
    )

    # ----------------------------------------------------
    # Stage B: the published track must carry real audio.
    # Watch analyser energy AND inbound-rtp stats.
    # ----------------------------------------------------

    stages.stamp(
        f"[Turn 0] Waiting up to {GREETING_TIMEOUT_S}s for the "
        "greeting audio (analyser + inbound-rtp stats)..."
    )

    greeting_deadline = time.monotonic() + GREETING_TIMEOUT_S
    result = None
    first_stats_heard = None
    while time.monotonic() < greeting_deadline:
        result = await watcher.poll()
        # Analyser alone is enough (stats fields can lag/vary),
        # once real playback energy accumulated.
        if result["analyser_heard"]:
            break
        # Stats already prove bot audio; give the analyser a
        # short grace period to confirm playback, then stop
        # waiting instead of burning the whole timeout.
        if result["stats_heard"]:
            if first_stats_heard is None:
                first_stats_heard = time.monotonic()
            elif time.monotonic() - first_stats_heard >= 10:
                break
        await asyncio.sleep(1.0)

    stages.stamp(
        "[Turn 0] Observation window closed: "
        f"analyser speech {result['speech_ms']}ms, "
        f"bytesReceived +{result['bytes_delta']}, "
        f"packetsReceived +{result['packets_delta']}, "
        f"max audioLevel {result['max_level']}, "
        f"totalAudioEnergy delta {result['energy_delta']}."
    )

    # Stage verdicts for the greeting window.
    greeting_tracks = result["snapshot"].get("remoteAudioTracks", [])
    greeting_publishing = [
        track for track in greeting_tracks
        if track["readyState"] == "live" and track["everUnmuted"]
    ]
    pipe.verdict(
        "agent_publish", len(greeting_publishing) > 0,
        f"{len(greeting_tracks)} remote audio track(s), "
        f"{len(greeting_publishing)} live+unmuted",
    )
    greeting_packets_ok = (
        result["bytes_delta"] >= STATS_MIN_BYTES
        and result["packets_delta"] >= STATS_MIN_PACKETS
    )
    greeting_non_silent = result["max_level"] >= STATS_AUDIO_LEVEL
    pipe.verdict(
        "inbound_rtp", greeting_packets_ok and greeting_non_silent,
        f"bytesReceived +{result['bytes_delta']}, packetsReceived "
        f"+{result['packets_delta']}, max audioLevel "
        f"{result['max_level']}",
    )
    if result["track_heard"] or result["element_heard"]:
        pipe.verdict(
            "audio_element", True,
            f"decoded-track energy {result['track_ms']} ms, "
            f"element energy {result['element_ms']} ms",
        )
    else:
        pipe.verdict(
            "audio_element", False,
            "no decoded agent audio reached the browser's "
            "WebAudio graph",
        )

    # Transcript evidence + capture: if the app renders the
    # greeting text, record it as the interview's first
    # assistant turn. Never a pass/fail criterion on its own
    # (the app may not render captions).
    try:
        body_text = await page.locator("body").inner_text()
        greeting_on_screen = (
            "jamie" in body_text.lower()
            and "interviewer" in body_text.lower()
        )
        stages.stamp(
            "[Turn 0] Greeting text rendered on screen: "
            f"{greeting_on_screen}"
        )
        if greeting_on_screen:
            pipe.verdict("ai", True, "greeting text rendered on screen")
        greeting_lines = await body_new_lines(
            page, body_before_greeting
        )
        if collector.record_assistant_lines(
            greeting_lines, source="dom-greeting", turn=0
        ):
            stages.stamp(
                "[Turn 0] Captured the rendered greeting as the "
                "first assistant transcript turn."
            )
    except Exception:
        pass

    if result["analyser_heard"]:
        pipe.finalize()
        stages.stamp(
            "[Turn 0] PASSED - bot spoke first "
            f"(audible greeting, {result['speech_ms']}ms of speech; "
            f"stats confirm: {result['stats_heard']})."
        )
        for line in pipe.lines():
            print(f"  {line}")
        return True

    # Analyser heard nothing. Before declaring the greeting
    # failed, confirm via WebRTC stats whether bot audio was
    # actually received.
    if result["stats_heard"]:
        pipe.finalize()
        for line in pipe.lines():
            print(f"  {line}")
        stages.stamp(
            f"[Turn 0] {DETECTOR_FAILED} - inbound-rtp stats show "
            f"real bot audio (bytes +{result['bytes_delta']}, "
            f"packets +{result['packets_delta']}, "
            f"max audioLevel {result['max_level']}) but the WebAudio "
            "analyser measured no playback energy. The bot DID "
            "greet; the Playwright-side audio detector is at fault "
            "(e.g. the app's <audio> element was not monitorable). "
            "Continuing with stats-based detection for the "
            "remaining turns."
        )
        return False

    # No audio by any measure - classify.
    tracks = result["snapshot"].get("remoteAudioTracks", [])
    packets_flowed = result["packets_delta"] >= STATS_MIN_PACKETS
    all_stayed_muted = tracks and all(
        not track["everUnmuted"] for track in tracks
    )

    if not packets_flowed:
        if all_stayed_muted:
            label = AGENT_JOINED_BUT_NO_AUDIO
            reason = (
                "The agent joined and announced an audio track, but "
                "the track stayed muted and no audio packets were "
                f"received within {GREETING_TIMEOUT_S}s. The agent "
                "is connected but never started speaking."
            )
            causes = [
                "The agent is waiting for candidate audio / VAD "
                "before its first utterance - but the required flow "
                "is that the bot greets FIRST, so this is an agent "
                "configuration bug.",
                "TTS failed or produced no audio, so the published "
                "track never carried frames - check the agent "
                "worker's TTS logs for this room.",
            ]
        else:
            label = AGENT_JOINED_BUT_NO_AUDIO
            reason = (
                "The agent joined and its audio track unmuted at "
                "least once, but essentially no audio packets "
                f"arrived within {GREETING_TIMEOUT_S}s "
                f"(bytes +{result['bytes_delta']}, "
                f"packets +{result['packets_delta']})."
            )
            causes = [
                "The agent token/API permissions prevent publishing "
                "audio (canPublish grant) - see the token grants in "
                "the diagnostics below.",
                "TTS failed mid-start or the agent's audio source "
                "produced no frames - check agent worker logs.",
            ]
    else:
        label = TTS_AUDIO_PUBLISH_FAILED
        reason = (
            "Audio packets flowed from the agent "
            f"(bytes +{result['bytes_delta']}, packets "
            f"+{result['packets_delta']}) but the stream was silent "
            f"(max audioLevel {result['max_level']}, "
            f"totalAudioEnergy delta {result['energy_delta']}). The "
            "publish pipeline works; the audio content is empty."
        )
        causes = [
            "TTS produced no audio (empty synthesis result) while "
            "the agent still published a silent track.",
            "The agent's audio pipeline is wired to a silent "
            "source - check the TTS provider response in the agent "
            "worker logs.",
        ]

    message = await capture_failure(
        page, context, recorder, stages, run_dir,
        label=label, reason=reason, likely_causes=causes,
        pipeline=pipe,
    )
    pytest.fail(message)


# ============================================================
# Phase 2: multi-turn responsiveness (after the greeting)
# ============================================================

async def run_multi_turn(
    page,
    context,
    recorder: PipelineRecorder,
    stages: StageLog,
    run_dir: Path,
    watcher: BotAudioWatcher,
    fixtures: list[Path],
    detector_ok: bool,
    turn_reports: list[dict],
    collector: TranscriptCollector,
    audio_records: list[dict],
) -> dict:
    use_stats = not detector_ok

    stages.stamp(
        "Greeting (scripted) validated separately - turn 1 below "
        "is the FIRST REAL INTERACTIVE TURN: the first candidate "
        "response that exercises the full STT -> AI -> TTS loop."
    )

    turn = 0
    interview_complete = False
    while turn < MAX_TURNS:
        turn += 1

        # Wait for the interviewer to finish its current question.
        await wait_for_bot_silence(page, watcher)

        # Drive to REAL completion: stop once the interviewer has
        # signalled the interview is over (unless a fixed turn
        # count was forced for debugging).
        if FORCED_TURNS is None:
            if await detect_interview_complete(page):
                turn -= 1
                interview_complete = True
                stages.stamp(
                    "Interview-complete signal detected after "
                    f"{turn} candidate answer(s) - ending the "
                    "drive-to-completion loop."
                )
                break
        elif turn > FORCED_TURNS:
            turn -= 1
            break

        answer_wav = fixtures[(turn - 1) % len(fixtures)]
        stages.stamp(
            f"[Turn {turn}/max {MAX_TURNS}] Speaking candidate "
            f"answer ({answer_wav.name})..."
        )

        pipe = TurnPipeline(turn)
        mic_before = await mic_state(page)
        snapshot_before = await rtc_snapshot(page)
        out_before = outbound_totals(snapshot_before)
        body_before = await body_text_lines(page)
        await watcher.rebase()
        answer_start_wall = time.time() * 1000

        clip_ms = await page.evaluate(
            "b => window.__emhSpeak(b)",
            fixture_base64(answer_wav),
        )

        # Stage 1: candidate mic - the injected clip must carry
        # energy on the fake microphone track (harness sanity).
        mic_after = await mic_state(page)
        mic_energy = mic_after["speechMs"] - mic_before["speechMs"]
        pipe.verdict(
            "candidate_mic", mic_energy > 0,
            f"{mic_energy} ms of mic energy from a {clip_ms} ms clip",
        )
        assert mic_energy > 0, (
            f"Turn {turn}: candidate audio was injected "
            f"({clip_ms} ms clip) but produced no energy on the "
            "fake microphone track - the input side of the test "
            "harness is broken, this is NOT a bot failure."
        )

        # Stage 2: outbound RTP - the answer must actually LEAVE
        # the browser with real (non-silent) source audio.
        snapshot_after_clip = await rtc_snapshot(page)
        out_after = outbound_totals(snapshot_after_clip)
        sent_delta = out_after["bytesSent"] - out_before["bytesSent"]
        packets_sent_delta = (
            out_after["packetsSent"] - out_before["packetsSent"]
        )
        pipe.verdict(
            "outbound_rtp", sent_delta > 0,
            f"bytesSent +{sent_delta}, packetsSent "
            f"+{packets_sent_delta}, sourceAudioLevel "
            f"{out_after['sourceAudioLevel']}",
        )

        # Stage 3: LiveKit received it - the SFU's RTCP receiver
        # reports for our outbound stream are the only browser-
        # observable proof of server-side receipt.
        remote_reports = snapshot_after_clip.get(
            "remoteInboundAudio", []
        )
        if remote_reports:
            pipe.verdict(
                "livekit_receive", sent_delta > 0,
                f"RTCP receiver reports: {remote_reports}",
            )
        else:
            pipe.record(
                "livekit_receive", "UNKNOWN",
                "no remote-inbound-rtp reports exposed yet",
            )

        stages.stamp(
            f"[Turn {turn}] Candidate audio delivered ({clip_ms} ms, "
            f"mic energy {mic_energy} ms, "
            f"outbound bytesSent +{sent_delta}, published "
            f"sourceAudioLevel {out_after['sourceAudioLevel']})."
        )
        if sent_delta <= 0:
            message = await capture_failure(
                page, context, recorder, stages, run_dir,
                label=f"CANDIDATE AUDIO NOT PUBLISHED AT TURN {turn}",
                reason=(
                    "The fake microphone carried energy locally but "
                    "no outbound audio bytes were sent to LiveKit "
                    f"during the {clip_ms} ms answer - the bot never "
                    "had a chance to hear it. This is a publish "
                    "problem (app or harness), NOT a bot-response "
                    "failure."
                ),
                likely_causes=[
                    "The app stopped/replaced its published mic "
                    "track (see local mic track states above).",
                    "The LiveKit publication was closed mid-"
                    "interview.",
                ],
                turn_reports=turn_reports,
                pipeline=pipe,
            )
            pytest.fail(message)

        # Stages 4-9: the bot must reply with audible speech.
        # Fail fast if the app's own watchdog declares the agent
        # dead - waiting out the rest of the timeout after that
        # only wastes time.
        deadline = time.monotonic() + BOT_RESPONSE_TIMEOUT_S
        responded = False
        via = None
        result = None
        watchdog_hits: list[dict] = []
        while time.monotonic() < deadline:
            result = await watcher.poll()
            if result["analyser_heard"]:
                responded, via = True, "analyser"
                break
            if use_stats and result["stats_heard"]:
                responded, via = True, "stats"
                break
            warnings = recorder.fatal_app_warnings_since(
                answer_start_wall
            )
            agent_gone = [
                w for w in warnings
                if "no agent response" in w["text"].lower()
            ]
            stream_gone = [
                w for w in warnings
                if "no active audio stream" in w["text"].lower()
            ]
            if agent_gone or len(stream_gone) >= 3:
                watchdog_hits = warnings
                stages.stamp(
                    f"[Turn {turn}] FAIL-FAST: app watchdog "
                    f"warning \"{warnings[-1]['text'][:100]}\" - "
                    "not waiting out the remaining response "
                    "timeout."
                )
                break
            await asyncio.sleep(0.5)
        response_wall = time.time() * 1000

        # STT / AI evidence from the rendered conversation text.
        answer_text = CANDIDATE_ANSWERS[
            (turn - 1) % len(CANDIDATE_ANSWERS)
        ]
        new_lines = await body_new_lines(page, body_before)
        evidence = transcript_evidence(new_lines, answer_text)

        # Real transcript capture: the candidate turn is what the
        # app's STT rendered for the injected audio (falling back
        # to the exact text that was spoken into the mic), the
        # assistant turn is the newly rendered interviewer text.
        app_stt_text = " ".join(evidence["stt_lines"]).strip()
        collector.record_turn(
            "user",
            app_stt_text or answer_text,
            source="app-stt" if app_stt_text else "injected-audio",
            turn=turn,
        )
        collector.record_assistant_lines(
            evidence["ai_lines"], source="dom-diff", turn=turn
        )
        audio_records.append(
            {
                "turn": turn,
                "interviewer_prompt": None,
                "reference_transcript": answer_text,
                "stt_transcript": app_stt_text or None,
                "reference_segments": None,
                "detected_segments": None,
                "audio_path": str(answer_wav),
            }
        )
        if evidence["stt_seen"]:
            pipe.verdict(
                "stt", True,
                f"answer text rendered on screen "
                f"({evidence['stt_matched_words']}/"
                f"{evidence['stt_vocabulary']} words)",
            )
        else:
            pipe.record(
                "stt", "UNKNOWN",
                "answer transcript not visible in page text "
                f"({evidence['new_line_count']} new lines) - STT "
                "runs inside the agent worker",
            )
        if evidence["ai_seen"]:
            pipe.verdict(
                "ai", True,
                f"new agent text: \"{evidence['ai_sample']}\"",
            )
        else:
            pipe.record(
                "ai", "UNKNOWN",
                "no new agent text visible in page body",
            )
        # TTS itself runs inside the agent worker; audio arriving
        # on the wire (later stages) is the only proof.
        pipe.record(
            "tts", "UNKNOWN",
            "not directly observable from the browser",
        )

        # Agent publish state during the response window.
        final_snapshot = result["snapshot"] if result else {}
        live_tracks = [
            track for track in
            final_snapshot.get("remoteAudioTracks", [])
            if track["readyState"] == "live"
        ]
        publishing = [
            track for track in live_tracks if track["everUnmuted"]
        ]
        pipe.verdict(
            "agent_publish", len(publishing) > 0,
            f"{len(live_tracks)} live remote audio track(s), "
            f"{len(publishing)} ever unmuted; mute states: "
            + str([
                (track["id"][:8], track["muted"])
                for track in final_snapshot.get(
                    "remoteAudioTracks", []
                )
            ]),
        )

        # Browser inbound RTP: packets AND non-silent content.
        if result:
            packets_ok = (
                result["bytes_delta"] >= STATS_MIN_BYTES
                and result["packets_delta"] >= STATS_MIN_PACKETS
            )
            non_silent = result["max_level"] >= STATS_AUDIO_LEVEL
            pipe.verdict(
                "inbound_rtp", packets_ok and non_silent,
                f"bytesReceived +{result['bytes_delta']}, "
                f"packetsReceived +{result['packets_delta']}, "
                f"max audioLevel {result['max_level']}"
                + (
                    " (packets flowed but content is SILENCE)"
                    if packets_ok and not non_silent else ""
                ),
            )
        else:
            pipe.verdict("inbound_rtp", False, "no stats polled")

        # Browser playback: decoded agent audio reaching WebAudio.
        # The app attaches no dedicated <audio> element (its only
        # media element is a muted <video> with no tracks), so
        # element-level capture is NOT authoritative here -
        # decoded remote-track energy is the playback signal, and
        # element energy is recorded as extra detail only.
        if result and (result["track_heard"] or result["element_heard"]):
            pipe.verdict(
                "audio_element", True,
                f"decoded-track energy {result['track_ms']} ms, "
                f"element energy {result['element_ms']} ms",
            )
        else:
            pipe.verdict(
                "audio_element", False,
                "no decoded agent audio reached the browser's "
                "WebAudio graph",
            )

        turn_reports.append(
            {
                "turn": turn,
                "answer_fixture": answer_wav.name,
                "answer_clip_ms": clip_ms,
                "candidate_mic_energy_ms": (
                    mic_after["speechMs"] - mic_before["speechMs"]
                ),
                "bot_responded": responded,
                "detected_via": via,
                "response_latency_s": (
                    round(
                        (response_wall - answer_start_wall) / 1000
                        - clip_ms / 1000,
                        1,
                    )
                    if responded else None
                ),
                "bytes_delta": result["bytes_delta"] if result else 0,
                "packets_delta": result["packets_delta"] if result else 0,
                "max_audio_level": result["max_level"] if result else 0,
                "pipeline": pipe.as_dict(),
            }
        )

        if not responded:
            # Distinguish "no reply at all" from "reply received
            # but not played" so a mid-interview freeze is not
            # blamed on the detector (and vice versa).
            stats_received = bool(result and result["stats_heard"])
            pipe.finalize()
            first_fail = pipe.first_failure()
            label = (
                DETECTOR_FAILED if stats_received
                else (
                    f"BOT STOPPED RESPONDING AT TURN {turn} "
                    "- PIPELINE BROKEN AT: "
                    + STAGE_TITLES.get(
                        first_fail or "", "<none identified>"
                    )
                )
            )
            reason = (
                (
                    "Inbound-rtp stats show bot audio arrived for "
                    f"turn {turn} (bytes +{result['bytes_delta']}, "
                    f"packets +{result['packets_delta']}, max "
                    f"audioLevel {result['max_level']}) but the "
                    "analyser heard no playback."
                )
                if stats_received else
                (
                    f"Candidate audio was delivered ({clip_ms} ms "
                    "with microphone energy) but the bot produced "
                    "no audio by ANY measure within "
                    f"{BOT_RESPONSE_TIMEOUT_S}s "
                    f"(analyser {result['speech_ms']}ms, bytes "
                    f"+{result['bytes_delta']}, packets "
                    f"+{result['packets_delta']}). The greeting and "
                    f"{turn - 1} earlier turn(s) worked, so this is "
                    "the mid-interview freeze, not a join/dispatch "
                    "problem."
                    + (
                        " FAILED FAST on app watchdog warning(s): "
                        + "; ".join(
                            f"[{datetime.fromtimestamp(w['ts'] / 1000):%H:%M:%S}] "
                            f"{w['text'][:120]}"
                            for w in watchdog_hits[-3:]
                        )
                        if watchdog_hits else ""
                    )
                )
            )
            message = await capture_failure(
                page, context, recorder, stages, run_dir,
                label=label, reason=reason,
                likely_causes=(
                    ["The app's <audio> element for the agent track "
                     "could not be monitored by the test harness."]
                    if stats_received else
                    [
                        "The agent stopped hearing the candidate "
                        "(subscription/VAD died) and is waiting "
                        "forever for input.",
                        "The agent's LLM/TTS call for this turn "
                        "failed or hung - check agent worker logs "
                        f"around the turn-{turn} timestamps in the "
                        "stage timeline.",
                        "The agent process crashed mid-interview "
                        "(its track would also show ended/muted in "
                        "the diagnostics above).",
                    ]
                ),
                turn_reports=turn_reports,
                pipeline=pipe,
            )
            pytest.fail(message)

        pipe.finalize()
        report = turn_reports[-1]
        stages.stamp(
            f"[Turn {turn}] Bot replied in "
            f"~{report['response_latency_s']}s (via {via}, bytes "
            f"+{report['bytes_delta']}, packets "
            f"+{report['packets_delta']})."
        )
        for line in pipe.lines():
            print(f"  {line}")

    reached_cap = not interview_complete and turn >= MAX_TURNS
    if reached_cap:
        stages.stamp(
            f"[WARNING] Reached MAX_TURNS={MAX_TURNS} without an "
            "interview-complete signal - the transcript will be "
            "marked INCOMPLETE (over-long interview or looping "
            "agent). Whole-interview evaluation will reject it."
        )
    else:
        stages.stamp(
            f"Interview drive finished: {turn} candidate turn(s), "
            f"complete={interview_complete}."
        )

    return {
        "complete": interview_complete,
        "turns_completed": turn,
        "reached_cap": reached_cap,
    }


# ============================================================
# Main test
# ============================================================

@pytest.mark.asyncio
async def test_bot_greets_first_then_stays_responsive():
    # Enforce a FRESH, non-stale, non-expired interview session so
    # a reused dead room is not misread as a bot/audio failure.
    try:
        _url, claims = require_fresh_interview_url()
        # Full-interview evaluation needs a room nobody joined
        # before: the greeting fires on room join. A consumed
        # session must fail as SESSION ALREADY CONSUMED
        # (session/environment), never as a bot failure.
        require_unconsumed_session(claims)
    except InterviewSessionError as error:
        pytest.fail(str(error))

    # How many distinct candidate answers to synthesize (they
    # cycle for longer interviews). Driven to completion up to
    # MAX_TURNS, so pre-generate up to the answer-set size.
    fixture_count = FORCED_TURNS or min(MAX_TURNS, len(CANDIDATE_ANSWERS))
    fixtures = ensure_answer_fixtures(fixture_count)

    run_dir = (
        DEBUG_ROOT
        / f"bot_responsiveness_{datetime.now():%Y%m%d_%H%M%S}"
    )

    print(
        f"Fresh interview session: candidate={claims.candidate_id} "
        f"job={claims.job_id} company={claims.company_id}"
    )

    recorder = PipelineRecorder()
    stages = StageLog()
    turn_reports: list[dict] = []
    collector = TranscriptCollector()
    audio_records: list[dict] = []
    interview_status = {
        "complete": False,
        "turns_completed": 0,
        "reached_cap": False,
    }

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                # Fake camera; audio comes from the injected
                # WebAudio microphone in INIT_SCRIPT.
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = await browser.new_context(
            permissions=["camera", "microphone"],
        )
        await context.add_init_script(INIT_SCRIPT)
        await context.tracing.start(screenshots=True, snapshots=True)

        page = await context.new_page()
        recorder.attach(page)

        try:
            print()
            print("=" * 70)
            print(
                "BOT RESPONSIVENESS TEST (turn 0 greeting + "
                f"drive-to-completion, cap {MAX_TURNS})"
            )
            print("=" * 70)

            await launch_into_interview_room(page, stages)

            # Room joined: the greeting fires now, so this
            # session can never again serve as a fresh full
            # interview - record it in the used-session ledger.
            mark_session_consumed(
                claims, "bot_responsiveness full interview"
            )

            watcher = BotAudioWatcher(page)

            # Phase 1: the bot speaks first. pytest.fail inside
            # carries one of the classified turn-0 labels.
            detector_ok = await validate_greeting(
                page, context, recorder, stages, run_dir, watcher,
                collector,
            )

            # Phase 2: drive the FULL interview to completion.
            # Failures here are labelled BOT STOPPED RESPONDING AT
            # TURN N, never confused with a greeting failure.
            interview_status = await run_multi_turn(
                page, context, recorder, stages, run_dir, watcher,
                fixtures, detector_ok, turn_reports,
                collector, audio_records,
            )

            # A whole interview that never reached completion is a
            # truncated interview - fail rather than pass a partial
            # transcript as success.
            if not interview_status["complete"]:
                message = await capture_failure(
                    page, context, recorder, stages, run_dir,
                    label="INTERVIEW DID NOT COMPLETE",
                    reason=(
                        "The interview ran "
                        f"{interview_status['turns_completed']} "
                        "candidate turn(s) but never reached an "
                        "interview-complete signal "
                        f"(reached_cap={interview_status['reached_cap']}). "
                        "The captured transcript is truncated and "
                        "must not be used for whole-interview "
                        "evaluation."
                    ),
                    likely_causes=[
                        "The agent stalled mid-interview (server-"
                        "side STT/AI/TTS) and never produced its "
                        "closing statement.",
                        "The interview genuinely exceeds MAX_TURNS "
                        f"({MAX_TURNS}) - raise EMH_MAX_TURNS if the "
                        "real interview is longer.",
                    ],
                    turn_reports=turn_reports,
                )
                pytest.fail(message)

            # Success: keep the per-turn report, drop the trace.
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / "bot_responsiveness_report.json").write_text(
                json.dumps(
                    {
                        "room": recorder.room_name,
                        "greeting_detector_ok": detector_ok,
                        "interview_complete": interview_status["complete"],
                        "turns_completed": (
                            interview_status["turns_completed"]
                        ),
                        "turns": turn_reports,
                    },
                    indent=2,
                )
            )
            await context.tracing.stop()

            print()
            print("=" * 70)
            print(
                "BOT RESPONSIVENESS TEST PASSED - greeting verified "
                "(bot spoke first) and the bot stayed responsive "
                "through the FULL interview "
                f"({interview_status['turns_completed']} turns, "
                "completed)."
            )
            print("=" * 70)

        except pytest.fail.Exception:
            # Already captured + classified by capture_failure.
            raise

        except Exception:
            # Unexpected failure (launch flow, navigation,
            # harness): still capture full diagnostics.
            message = await capture_failure(
                page, context, recorder, stages, run_dir,
                label="UNEXPECTED TEST ERROR",
                reason=(
                    "The test aborted before a turn verdict - see "
                    "the raised exception and the stage timeline."
                ),
                likely_causes=[
                    "Launch-flow / navigation / harness problem, "
                    "not necessarily a bot failure.",
                ],
                turn_reports=turn_reports,
            )
            print(message)
            raise

        finally:
            # Persist the REAL captured transcript and per-turn
            # audio records for the evaluation layers, whether
            # the run passed or failed - a partial transcript
            # from a failed run is still real product output.
            transcript_path = collector.save()
            records_path = save_audio_turn_records(audio_records)
            status_path = save_transcript_status(
                complete=interview_status["complete"],
                turn_count=interview_status["turns_completed"],
                reached_cap=interview_status["reached_cap"],
                captured_at=time.time(),
            )
            print(
                f"Real transcript artifact: {transcript_path} "
                f"({len(collector.turns)} turns captured, "
                f"complete={interview_status['complete']})"
            )
            print(f"Audio turn records: {records_path}")
            print(f"Transcript status: {status_path}")

            await context.close()
            await browser.close()
