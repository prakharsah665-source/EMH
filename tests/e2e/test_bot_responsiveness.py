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

Phase 2 - FULL INTERVIEW, DRIVEN UNTIL THE BOT CONCLUDES IT
    Only after the greeting is verified does the candidate start
    answering. Per turn: inject a synthesized spoken answer into
    the fake microphone, then expect an audible bot reply within
    BOT_RESPONSE_TIMEOUT_S. There is NO fixed turn count: the
    evaluator stays in the interview room until the AI
    interviewer itself concludes the interview (socket.io exit
    signal / end-of-interview UI). A missing reply is recorded as
    a DEFERRED failure ("BOT STOPPED RESPONDING AT TURN N" with
    the full pipeline diagnostics) - it never ends the interview:
    the evaluator waits BOT_RECOVERY_TIMEOUT_S for a late reply,
    then re-speaks the answer, and repeats until the bot answers
    or concludes (bounded only by the INTERVIEW_MAX_S wall-clock
    safety cap). Deferred failures are reported AFTER the
    interview concludes, with the transcript marked complete.

Candidate audio never comes from a human microphone: an init
script replaces navigator.mediaDevices.getUserMedia with a
WebAudio MediaStreamDestination the test feeds on demand. The
candidate's words come from the LLM CandidateSimulator
(simulator/candidate_simulator.py): every turn the ACTUAL
interviewer question is read from the live caption stream
(simulator/live_answers.py), the simulator answers it in its
hidden persona (EMH_TEST_MODE=competency, default) or executes
the adversarial spec (EMH_TEST_MODE=robustness), and the answer
is synthesized with macOS `say` into artifacts/audio/candidate/.
There is NO scripted answer bank on this path: if a valid live
question cannot be read the drive fails with the reason.

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
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from collectors.transcript_capture import (
    AUDIO_RECORD_JS,
    TRANSCRIPT_HOOK_JS,
    clear_previous_capture_artifacts,
    drain_transcript_events,
    start_bot_audio_recording,
    stop_bot_audio_recording,
    transcribe_captured_audio,
    write_capture_evidence,
)
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
from pages.interview_launch import (
    launch_into_interview_room as _launch_shared,
)
from simulator.candidate_simulator import CandidateSimulator
from simulator.live_answers import (
    LiveQuestionUnavailable,
    LiveSimulatorAnswerSource,
    ScriptedAnswerSource,
)
from simulator.personas import PERSONAS
from simulator.role_context import role_context_from_frame_rows


# ============================================================
# Configuration
# ============================================================

# The interview is driven to REAL completion: the evaluator stays
# in the room until the AI interviewer itself concludes the
# interview (socket.io exit signal / end-of-interview UI). There
# is NO fixed question/turn count - the only bound is a wall-
# clock safety cap so a dead or looping agent cannot hang CI
# forever. EMH_MAX_TURNS (default 0 = unlimited) remains an
# optional hard cap; EMH_INTERVIEW_TURNS still forces an exact
# turn count for single-question debugging.
INTERVIEW_MAX_S = int(os.getenv("EMH_INTERVIEW_MAX_S", str(45 * 60)))
MAX_TURNS = max(0, int(os.getenv("EMH_MAX_TURNS", "0")))  # 0 = no cap

# After the interviewer's exit signal, keep the page alive for a
# bounded grace period so the app can finish its completion flow
# (end-interview / interview_exit exchange, final media-chunk
# flush, completion UI / status update) before teardown. 0
# disables the wait.
POST_EXIT_GRACE_S = max(
    0.0, float(os.getenv("EMH_POST_EXIT_GRACE_S", "25"))
)
_FORCED_TURNS = os.getenv("EMH_INTERVIEW_TURNS")
FORCED_TURNS = max(1, int(_FORCED_TURNS)) if _FORCED_TURNS else None

# Mid-interview stall recovery. A bot that does not answer within
# BOT_RESPONSE_TIMEOUT_S is NOT terminal: the evaluator records
# the stall (full diagnostics), keeps waiting in the room for
# BOT_RECOVERY_TIMEOUT_S for a late reply, then re-speaks the
# candidate answer (the app's own watchdog returns to
# USER_WAITING, so a re-prompt is the natural recovery) and
# repeats until the bot answers, concludes, or INTERVIEW_MAX_S
# is exhausted. Recorded stalls are reported AFTER the interview
# concludes, never by ending it early.
BOT_RECOVERY_TIMEOUT_S = int(os.getenv("EMH_BOT_RECOVERY_TIMEOUT", "120"))

# The ONE non-bot terminal condition besides the safety caps: the
# server tears the room down (LiveKit peer connections closed /
# socket.io "transport close") WITHOUT a conclusion signal and
# never reconnects within this grace period. Re-prompting into a
# dead page cannot recover; the drive exits and the transcript is
# marked incomplete with an ENVIRONMENT classification. Set
# EMH_ROOM_DISCONNECT_GRACE=0 to disable (stay until the wall-
# clock cap even in a dead room).
ROOM_DISCONNECT_GRACE_S = int(os.getenv("EMH_ROOM_DISCONNECT_GRACE", "180"))
ROOM_DISCONNECTED = "ROOM DISCONNECTED BY SERVER (no conclusion signal)"

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

# Candidate answers: generated live per turn by the CandidateSimulator
# from the ACTUAL interviewer question (see simulator/live_answers.py).
# Mode/persona selection for the live drive:
TEST_MODE = os.getenv("EMH_TEST_MODE", "competency").lower()
SIMULATOR_PERSONA = os.getenv("EMH_SIMULATOR_PERSONA", "average").lower()
SIMULATOR_TURNS_PATH = Path("artifacts/transcripts/simulator_turns.json")


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
# Audio injection (no human microphone)
# ============================================================

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

    // The currently playing fake-mic source, so setup-screen
    // audio can be explicitly stopped before the room is joined.
    let activeMicSource = null;

    // Stop any clip still playing on the fake microphone.
    // Returns true when something was actually stopped.
    // stop() fires the source's onended, so the pending
    // __emhSpeak promise resolves and mic.playing clears.
    window.__emhStopSpeak = () => {
        if (!activeMicSource) return false;
        try { activeMicSource.stop(); } catch (e) {}
        activeMicSource = null;
        return true;
    };

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
                if (activeMicSource === source) activeMicSource = null;
                state.mic.playing = false;
                ev('candidate_audio_end');
                resolve(Math.round(buffer.duration * 1000));
            };
            activeMicSource = source;
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
    for url in {os.getenv("EMH_INTERVIEW_URL"), os.getenv("INTERVIEW_URL")}:
        if url:
            text = text.replace(url, "<INTERVIEW_URL>")
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
                    # 2000 chars so text frames (socket.io
                    # events) are recorded whole - the
                    # socket.io transcript audit must be able
                    # to rule text in/out conclusively.
                    body = mask_tokens(str(payload))[:2000]
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


## Socket.io conclusion signals. The bot's closing sentence is
## AUDIO-ONLY (no DOM text), so DOM detection alone misses the
## concluding state; the room-api socket.io events are the
## authoritative signal ("bot-speech-ended" carries isExit, and
## the app broadcasts terminal interview states).
_SOCKET_EXIT_MARKERS = (
    '"isExit":true',
    '"isExit": true',
    '"isMoveToCodingAssessment":true',
    '"isMoveToCodingAssessment": true',
    "INTERVIEW_COMPLETED",
    "INTERVIEW_ENDED",
    "INTERVIEW_END",
)


def socket_exit_signalled(recorder: "PipelineRecorder") -> str | None:
    """
    Return the matching frame payload when the room-api
    socket.io channel has signalled the interview concluded,
    else None.
    """

    for frame in reversed(recorder.websocket_frames):
        if "socket.io" not in (frame.get("url") or ""):
            continue
        payload = frame.get("payload") or ""
        for marker in _SOCKET_EXIT_MARKERS:
            if marker in payload:
                return payload[:200]
    return None


async def interview_concluded(
    page, recorder: "PipelineRecorder"
) -> str | None:
    """
    Combined conclusion check used BEFORE injecting any
    candidate turn and inside every failure path: returns a
    human-readable reason when the interview has entered its
    final/concluding state (socket.io exit signal or
    end-of-interview UI), else None. After conclusion no
    candidate audio may be injected and missing outbound RTP
    is expected teardown, never an audio/bot failure.
    """

    exit_frame = socket_exit_signalled(recorder)
    if exit_frame:
        return f"socket.io exit signal: {exit_frame}"

    if await detect_interview_complete(page):
        return "end-of-interview text/UI rendered on the page"

    return None


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
    keep_trace: bool = False,
    screenshot_name: str = "bot_responsiveness_failed.png",
) -> str:
    """
    Save screenshot, trace and all collected logs, and build a
    failure message with the classification label first and full
    diagnostics after it.

    keep_trace=True captures the diagnostics for a NON-terminal
    (deferred) failure while the interview keeps running: the
    Playwright trace stays open so the rest of the interview is
    still traced, and the screenshot gets its own name.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    screenshot = SCREENSHOT_DIR / screenshot_name
    try:
        await page.screenshot(path=str(screenshot), full_page=True)
    except Exception:
        screenshot = None

    trace_path = run_dir / "trace.zip"
    if keep_trace:
        trace_path = f"{trace_path} (still recording - saved at exit)"
    else:
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
# Deferred failures
#
# The evaluator never leaves the interview room because of a
# bot-response failure: every classified failure that used to
# call pytest.fail() mid-interview is recorded here with its
# full diagnostics and reported AFTER the AI interviewer has
# concluded the interview (or the wall-clock cap is hit).
# ============================================================

def defer_failure(
    deferred: list[dict],
    stages: StageLog,
    *,
    turn: int,
    label: str,
    message: str,
    attempt: int = 1,
) -> None:
    deferred.append(
        {
            "turn": turn,
            "attempt": attempt,
            "label": label,
            "message": message,
            "recorded_at": time.time(),
            "recovered": False,
        }
    )
    stages.stamp(
        f"[Turn {turn}] DEFERRED FAILURE #{len(deferred)}: {label} "
        "- diagnostics captured; STAYING IN THE ROOM (the "
        "interview ends only when the AI interviewer concludes "
        "it)."
    )
    print()
    print("-" * 70)
    print(f"DEFERRED (non-terminal) FAILURE at turn {turn}: {label}")
    print(message)
    print("-" * 70)


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
    deferred: list[dict] | None = None,
) -> bool:
    """
    Verify Jamie's automatic greeting. Returns True if the
    analyser heard the greeting, False if only WebRTC stats
    confirmed it (detector failed - later turns fall back to
    stats-based detection) OR if the greeting never arrived - in
    that case the classified failure is DEFERRED (recorded with
    full diagnostics) and the evaluator stays in the room so the
    multi-turn loop can prompt the bot and let it recover.
    """

    if deferred is None:
        deferred = []

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
            keep_trace=True,
            screenshot_name="bot_responsiveness_turn00_stall.png",
        )
        defer_failure(
            deferred, stages, turn=0, label=AGENT_NEVER_JOINED,
            message=message,
        )
        return False

    snapshot = await rtc_snapshot(page)
    stages.stamp(
        "[Turn 0] Agent participant joined - remote audio track(s): "
        + json.dumps(snapshot["remoteAudioTracks"])
    )

    # Record the bot's remote audio from the very first
    # utterance (greeting) for the stt-local fallback.
    if await start_bot_audio_recording(page):
        stages.stamp(
            "[Turn 0] Remote-audio recording started (stt-local "
            "capture)."
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
        keep_trace=True,
        screenshot_name="bot_responsiveness_turn00_stall.png",
    )
    defer_failure(
        deferred, stages, turn=0, label=label, message=message
    )
    return False


# ============================================================
# Phase 2: multi-turn responsiveness (after the greeting)
# ============================================================

def room_gone(snapshot: dict) -> bool:
    """
    True when the browser's LiveKit peer connection(s) are all
    closed/failed/disconnected and no remote audio track is
    live - i.e. the server tore the room down.
    """

    states = snapshot.get("connectionStates") or []
    if not states:
        return False
    dead = {"closed", "failed", "disconnected"}
    if any(
        (st.get("connection") or "") not in dead for st in states
    ):
        return False
    return not any(
        track.get("readyState") == "live"
        for track in snapshot.get("remoteAudioTracks", [])
    )


async def wait_for_late_reply(
    page,
    watcher: BotAudioWatcher,
    recorder: PipelineRecorder,
    use_stats: bool,
    timeout_s: float,
    room_state: dict | None = None,
) -> tuple[bool, str | None]:
    """
    Recovery wait after a stall: stay in the room and keep
    polling for a LATE bot reply or an interview-conclusion
    signal for up to timeout_s. Returns (heard, concluded).

    room_state (shared across attempts) tracks how long the room
    has been torn down; once ROOM_DISCONNECT_GRACE_S is exceeded
    room_state["disconnected"] is set and the wait returns early
    so the caller can leave with the ENVIRONMENT classification.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = await watcher.poll()
        if result["analyser_heard"] or (
            use_stats and result["stats_heard"]
        ):
            if room_state is not None:
                room_state.pop("gone_since", None)
            return True, None
        concluded = await interview_concluded(page, recorder)
        if concluded:
            return False, concluded
        if room_state is not None and ROOM_DISCONNECT_GRACE_S > 0:
            if room_gone(result["snapshot"]):
                room_state.setdefault("gone_since", time.monotonic())
                if (
                    time.monotonic() - room_state["gone_since"]
                    >= ROOM_DISCONNECT_GRACE_S
                ):
                    room_state["disconnected"] = True
                    return False, None
            else:
                room_state.pop("gone_since", None)
        await asyncio.sleep(1.0)
    return False, None


async def post_exit_grace(
    page,
    recorder: "PipelineRecorder",
    stages: StageLog,
    run_dir: Path,
) -> dict:
    """
    Bounded wait AFTER the interviewer's exit signal, BEFORE
    teardown: the interview is over (no further turns are
    driven), but the app still needs the page to finish its
    completion flow. Observes the post-exit socket.io traffic
    (end-interview / interview_exit, final media chunk with
    isSegmentComplete:true) and the completion UI, ends early
    once completion is confirmed, and always ends at the
    POST_EXIT_GRACE_S bound. The post-exit frames are dumped to
    run_dir/post_exit_frames.jsonl as completion evidence.
    """

    summary = {
        "grace_s": POST_EXIT_GRACE_S,
        "events": [],
        "completion_ui": False,
        "post_exit_frames": 0,
        "ended_early": False,
    }
    if POST_EXIT_GRACE_S <= 0:
        return summary

    start_ms = time.time() * 1000
    deadline = time.monotonic() + POST_EXIT_GRACE_S

    def post_exit_frames() -> list[dict]:
        return [
            frame
            for frame in recorder.websocket_frames
            if frame.get("ts", 0) >= start_ms
            and "socket.io" in (frame.get("url") or "")
        ]

    seen: set[tuple[str, str]] = set()
    while time.monotonic() < deadline:
        for frame in post_exit_frames():
            payload = frame.get("payload") or ""
            for name in ("end-interview", "interview_exit"):
                if name in payload:
                    seen.add((name, frame.get("direction") or "?"))
            if '"isSegmentComplete":true' in payload:
                seen.add(
                    ("final-chunk-flush", frame.get("direction") or "?")
                )
        summary["completion_ui"] = await detect_interview_complete(page)
        names = {name for name, _direction in seen}
        if summary["completion_ui"] and (
            "end-interview" in names or "interview_exit" in names
        ):
            summary["ended_early"] = True
            break
        await asyncio.sleep(1.0)

    frames = post_exit_frames()
    summary["events"] = sorted(f"{n}({d})" for n, d in seen)
    summary["post_exit_frames"] = len(frames)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "post_exit_frames.jsonl").write_text(
            "\n".join(json.dumps(frame) for frame in frames)
        )
    except Exception:
        pass
    stages.stamp(
        f"Post-exit grace ({POST_EXIT_GRACE_S:.0f}s cap, ended "
        f"{'early on confirmed completion' if summary['ended_early'] else 'at the bound'}): "
        f"completion events {summary['events'] or 'NONE'}, "
        f"completion UI {summary['completion_ui']}, "
        f"{summary['post_exit_frames']} post-exit socket frame(s) "
        f"-> {run_dir / 'post_exit_frames.jsonl'}"
    )
    return summary


async def run_multi_turn(
    page,
    context,
    recorder: PipelineRecorder,
    stages: StageLog,
    run_dir: Path,
    watcher: BotAudioWatcher,
    answers,
    detector_ok: bool,
    turn_reports: list[dict],
    collector: TranscriptCollector,
    audio_records: list[dict],
    interview_status: dict | None = None,
    audio_manifest: list[dict] | None = None,
    deferred: list[dict] | None = None,
) -> dict:
    """
    Drive the interview until the AI interviewer CONCLUDES it.

    The evaluator stays in the room for the whole interview:
    a bot that does not answer is never a reason to leave.
    Each non-response is recorded as a deferred failure (full
    diagnostics), followed by a recovery wait for a late reply
    and a re-prompt (the candidate answer is spoken again). The
    loop exits ONLY on the bot's conclusion signal, or on the
    INTERVIEW_MAX_S wall-clock safety cap / optional EMH_MAX_TURNS.
    """

    # `answers` is the per-turn answer source: the live drive
    # passes a LiveSimulatorAnswerSource (actual interviewer
    # question -> CandidateSimulator -> speech). The offline
    # FakeBot harness passes a list of WAV paths, wrapped here -
    # that wrapper is never used by the live test.
    if isinstance(answers, (list, tuple)):
        answers = ScriptedAnswerSource([Path(p) for p in answers])

    # interview_status is mutated IN PLACE as turns complete so
    # the finally-block status sidecar is truthful whatever way
    # this function exits.
    if interview_status is None:
        interview_status = {}
    if audio_manifest is None:
        audio_manifest = []
    if deferred is None:
        deferred = []

    use_stats = not detector_ok
    drive_started = time.monotonic()
    drive_deadline = drive_started + INTERVIEW_MAX_S

    def time_left() -> float:
        return drive_deadline - time.monotonic()

    stages.stamp(
        "Greeting (scripted) validated separately - turn 1 below "
        "is the FIRST REAL INTERACTIVE TURN: the first candidate "
        "response that exercises the full STT -> AI -> TTS loop. "
        "The evaluator stays in the room until the interviewer "
        f"concludes (wall-clock cap {INTERVIEW_MAX_S}s"
        + (f", hard turn cap {MAX_TURNS}" if MAX_TURNS else "")
        + ")."
    )

    turn = 0
    interview_complete = False
    conclusion_reason: str | None = None
    reached_cap = False
    room_state: dict = {}
    interview_status.setdefault("stalls", 0)
    interview_status.setdefault("recoveries", 0)
    interview_status.setdefault("room_disconnected", False)

    while True:
        # ---- Safety caps (never the normal exit path) ----
        if time_left() <= 0:
            reached_cap = True
            stages.stamp(
                f"[WARNING] Wall-clock cap INTERVIEW_MAX_S="
                f"{INTERVIEW_MAX_S}s exhausted after {turn} "
                "candidate turn(s) without an interview-complete "
                "signal - leaving the room."
            )
            break
        if MAX_TURNS and turn >= MAX_TURNS:
            reached_cap = True
            stages.stamp(
                f"[WARNING] Hard turn cap EMH_MAX_TURNS={MAX_TURNS} "
                "reached without an interview-complete signal - "
                "leaving the room."
            )
            break

        turn += 1

        # Wait for the interviewer to finish its current question.
        await wait_for_bot_silence(page, watcher)

        # The bot utterance that just ended is fully recorded -
        # persist it for the stt-local fallback (turn 1's
        # predecessor is the greeting).
        clip_label = (
            "bot_turn_00_greeting"
            if turn == 1
            else f"bot_turn_{turn - 1:02d}"
        )
        saved_clip = await stop_bot_audio_recording(
            page, clip_label
        )
        if saved_clip:
            audio_manifest.append(
                {
                    "role": "assistant",
                    "turn": turn - 1,
                    "audio_path": str(saved_clip),
                }
            )

        # Drive to REAL completion: stop once the interviewer
        # has signalled the interview is over (unless a fixed
        # turn count was forced for debugging). Checked BEFORE
        # every injection - a candidate answer must never be
        # spoken into a concluded interview.
        if FORCED_TURNS is None:
            concluded = await interview_concluded(page, recorder)
            if concluded:
                turn -= 1
                interview_complete = True
                conclusion_reason = concluded
                stages.stamp(
                    "Interview-complete signal detected after "
                    f"{turn} candidate answer(s) ({concluded}) - "
                    "the AI interviewer concluded the interview; "
                    "ending the drive loop cleanly, no further "
                    "answers injected."
                )
                break
        elif turn > FORCED_TURNS:
            # Debug mode: a forced turn count is NOT a completed
            # interview - the transcript stays marked incomplete.
            turn -= 1
            conclusion_reason = (
                f"forced turn count EMH_INTERVIEW_TURNS={FORCED_TURNS} "
                "(debug; not a bot conclusion)"
            )
            break

        # The answer for THIS turn: read the interviewer's actual
        # completed question from the live caption stream and let
        # the simulator answer it. No scripted fallback - an
        # unreadable question fails the drive with the reason.
        try:
            answer_wav, answer_text, live_question, _ = (
                await answers.next_answer(turn, page)
            )
        except LiveQuestionUnavailable as error:
            reason = f"LIVE QUESTION UNAVAILABLE at turn {turn}: {error}"
            stages.stamp(f"[Turn {turn}] {reason}")
            interview_status["conclusion_reason"] = reason
            raise AssertionError(
                f"{reason}\n"
                "The candidate simulator can only answer the ACTUAL "
                "interviewer question; it never substitutes scripted "
                "text. Either the interviewer produced no new "
                "utterance after the previous answer (bot failure) or "
                "its caption stream was not captured (capture "
                "failure) - see the deferred failures / stage log."
            ) from error

        # ------------------------------------------------------
        # One candidate turn = as many ATTEMPTS as it takes:
        # inject the answer, wait for the bot; on a stall
        # record the failure, wait for a late reply, re-prompt.
        # ------------------------------------------------------
        attempt = 0
        responded = False
        via = None
        turn_done = False
        while not turn_done:
            attempt += 1
            if time_left() <= 0:
                break  # outer loop records the wall-clock cap

            retry_note = f" (re-prompt #{attempt - 1})" if attempt > 1 else ""
            stages.stamp(
                f"[Turn {turn}] Speaking candidate answer "
                f"({answer_wav.name}){retry_note}..."
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

            # Start recording the bot's REPLY to this answer (the
            # remote track carries only bot audio, so starting now
            # captures the full upcoming utterance).
            await start_bot_audio_recording(page)

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
                # If the interview concluded while (or just before)
                # this answer played, the app legitimately tears
                # down its published track: missing outbound RTP is
                # expected teardown here, NOT an audio/bot failure.
                concluded = await interview_concluded(page, recorder)
                if concluded:
                    interview_complete = True
                    conclusion_reason = concluded
                    stages.stamp(
                        f"[Turn {turn}] Interview concluded during "
                        f"this answer ({concluded}) - outbound "
                        "publish teardown is expected; ending the "
                        "loop cleanly and marking the interview "
                        "complete."
                    )
                    turn -= 1
                    turn_done = True
                    break
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
                    keep_trace=True,
                    screenshot_name=(
                        f"bot_responsiveness_turn{turn:02d}_"
                        f"attempt{attempt}_publish.png"
                    ),
                )
                defer_failure(
                    deferred, stages, turn=turn, attempt=attempt,
                    label=f"CANDIDATE AUDIO NOT PUBLISHED AT TURN {turn}",
                    message=message,
                )
                interview_status["stalls"] += 1
                # Give the app a moment to re-publish, then
                # re-prompt (stay in the room).
                heard, concluded = await wait_for_late_reply(
                    page, watcher, recorder, use_stats,
                    min(BOT_RECOVERY_TIMEOUT_S, max(0, time_left())),
                    room_state=room_state,
                )
                if concluded:
                    interview_complete = True
                    conclusion_reason = concluded
                    turn -= 1
                    turn_done = True
                    break
                if room_state.get("disconnected"):
                    break
                continue

            # Stages 4-9: the bot must reply with audible speech.
            # The app's own watchdog declaring the agent silent
            # ends THIS WAIT early (no point waiting out the rest
            # of the timeout) - it never ends the interview.
            deadline = time.monotonic() + min(
                BOT_RESPONSE_TIMEOUT_S, max(1, time_left())
            )
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
                        f"[Turn {turn}] App watchdog warning "
                        f"\"{warnings[-1]['text'][:100]}\" - ending "
                        "this response wait early (the interview "
                        "continues: recovery wait + re-prompt)."
                    )
                    break
                await asyncio.sleep(0.5)
            response_wall = time.time() * 1000

            # STT / AI evidence from the rendered conversation text.
            new_lines = await body_new_lines(page, body_before)
            evidence = transcript_evidence(new_lines, answer_text)

            # Real transcript capture: the candidate turn is what the
            # app's STT rendered for the injected audio (falling back
            # to the exact text that was spoken into the mic), the
            # assistant turn is the newly rendered interviewer text.
            # Re-prompts are recorded too - the bot heard them.
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
                    "attempt": attempt,
                    "interviewer_prompt": None,
                    "reference_transcript": answer_text,
                    "stt_transcript": app_stt_text or None,
                    "reference_segments": None,
                    "detected_segments": None,
                    "audio_path": str(answer_wav),
                }
            )
            # The candidate clip actually injected this attempt also
            # feeds the stt-local transcript (genuine STT output of
            # the audio the bot heard).
            audio_manifest.append(
                {
                    "role": "user",
                    "turn": turn,
                    "attempt": attempt,
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
                    "attempt": attempt,
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

            if responded:
                turn_done = True
                break

            # ---- Bot did not reply within the response window ----

            # A bot that just DELIVERED its closing statement
            # does not reply again: if the interview concluded
            # during this turn, this is natural completion, not
            # a freeze.
            concluded = await interview_concluded(page, recorder)
            if concluded:
                interview_complete = True
                conclusion_reason = concluded
                stages.stamp(
                    f"[Turn {turn}] No further bot reply and the "
                    f"interview has concluded ({concluded}) - "
                    "treating this as natural completion, not a "
                    "response failure."
                )
                turn_done = True
                break

            # Agent-side diagnostics streamed over the LiveKit
            # data channel often carry the ROOT CAUSE (e.g. the
            # agent's STT provider erroring) - surface them in
            # the failure instead of leaving them buried in the
            # artifacts.
            try:
                dc_agent_errors = await page.evaluate(
                    "() => (window.__emhTranscriptEvents || [])"
                    ".filter(e => e.text && "
                    "/error=|APIStatusError|Exception/.test(e.text))"
                    ".slice(-3).map(e => e.text.slice(0, 300))"
                )
            except Exception:
                dc_agent_errors = []

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
                        " App watchdog warning(s): "
                        + "; ".join(
                            f"[{datetime.fromtimestamp(w['ts'] / 1000):%H:%M:%S}] "
                            f"{w['text'][:120]}"
                            for w in watchdog_hits[-3:]
                        )
                        if watchdog_hits else ""
                    )
                    + (
                        " AGENT-SIDE ERRORS streamed over the "
                        "LiveKit data channel (likely ROOT "
                        "CAUSE): "
                        + " | ".join(dc_agent_errors)
                        if dc_agent_errors else ""
                    )
                    + f" (attempt {attempt} of this turn)"
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
                keep_trace=True,
                screenshot_name=(
                    f"bot_responsiveness_turn{turn:02d}_"
                    f"attempt{attempt}_stall.png"
                ),
            )
            defer_failure(
                deferred, stages, turn=turn, attempt=attempt,
                label=label, message=message,
            )
            interview_status["stalls"] += 1

            if stats_received:
                # Bot audio DID arrive (detector problem only):
                # the interview is progressing - move on with
                # stats-based detection rather than re-prompting
                # into a bot that already answered.
                use_stats = True
                responded, via = True, "stats"
                turn_reports[-1]["bot_responded"] = True
                turn_reports[-1]["detected_via"] = "stats"
                turn_done = True
                break

            # ---- Recovery: stay in the room, wait for a late
            # reply, then re-prompt with the same answer. ----
            wait_s = min(BOT_RECOVERY_TIMEOUT_S, max(0, time_left()))
            stages.stamp(
                f"[Turn {turn}] RECOVERY: staying in the room for "
                f"up to {wait_s:.0f}s for a late bot reply or a "
                "conclusion signal before re-prompting."
            )
            heard, concluded = await wait_for_late_reply(
                page, watcher, recorder, use_stats, wait_s,
                room_state=room_state,
            )
            if room_state.get("disconnected"):
                stages.stamp(
                    f"[Turn {turn}] {ROOM_DISCONNECTED}: LiveKit peer "
                    "connections closed and no remote track for "
                    f">= {ROOM_DISCONNECT_GRACE_S}s during the "
                    "recovery wait - the room is gone, nothing to "
                    "stay in. Leaving with an ENVIRONMENT "
                    "classification (transcript incomplete)."
                )
                break
            if concluded:
                interview_complete = True
                conclusion_reason = concluded
                stages.stamp(
                    f"[Turn {turn}] Interview concluded during the "
                    f"recovery wait ({concluded})."
                )
                turn_done = True
                break
            if heard:
                # The bot recovered on its own (late reply).
                deferred[-1]["recovered"] = True
                deferred[-1]["recovery"] = "late reply"
                interview_status["recoveries"] += 1
                responded, via = True, "late-reply"
                turn_reports[-1]["bot_responded"] = True
                turn_reports[-1]["detected_via"] = "late-reply"
                turn_reports[-1]["response_latency_s"] = round(
                    (time.time() * 1000 - answer_start_wall) / 1000
                    - clip_ms / 1000, 1,
                )
                stages.stamp(
                    f"[Turn {turn}] RECOVERED: late bot reply heard "
                    "during the recovery wait - continuing the "
                    "interview."
                )
                turn_done = True
                break
            # Still silent: re-prompt (next attempt). The loop
            # is bounded only by the wall-clock cap.
            stages.stamp(
                f"[Turn {turn}] Still no bot reply after the "
                "recovery wait - re-speaking the candidate answer."
            )
            # Wait for the bot to be silent again (it may have
            # started talking right at the deadline).
            await wait_for_bot_silence(page, watcher)

        if interview_complete:
            break

        if room_state.get("disconnected"):
            interview_status["room_disconnected"] = True
            break

        if not responded:
            # Only reachable when the wall-clock cap expired
            # mid-turn; the outer loop records the cap.
            continue

        pipe.finalize()
        report = turn_reports[-1]
        if attempt > 1:
            deferred_for_turn = [
                d for d in deferred if d["turn"] == turn
            ]
            for entry in deferred_for_turn:
                if not entry["recovered"]:
                    entry["recovered"] = True
                    entry["recovery"] = f"answered after re-prompt #{attempt - 1}"
            interview_status["recoveries"] += 1
        stages.stamp(
            f"[Turn {turn}] Bot replied in "
            f"~{report['response_latency_s']}s (via {via}, bytes "
            f"+{report['bytes_delta']}, packets "
            f"+{report['packets_delta']}"
            + (f", after {attempt - 1} re-prompt(s)" if attempt > 1 else "")
            + ")."
        )
        for line in pipe.lines():
            print(f"  {line}")

        # Turn finished end-to-end: record it immediately so the
        # sidecar is truthful whatever happens next.
        interview_status["turns_completed"] = turn

    if interview_status.get("room_disconnected"):
        stages.stamp(
            f"[WARNING] {ROOM_DISCONNECTED} after "
            f"{interview_status.get('turns_completed', 0)} completed "
            "candidate turn(s) - the transcript is marked "
            "INCOMPLETE. This is a room/environment condition, "
            "not a bot-response failure."
        )
    elif reached_cap:
        stages.stamp(
            f"[WARNING] Safety cap hit after {turn} candidate "
            "turn(s) without an interview-complete signal - the "
            "transcript will be marked INCOMPLETE (agent dead, "
            "over-long or looping interview). Whole-interview "
            "evaluation will reject it."
        )
    else:
        stages.stamp(
            f"Interview drive finished: {turn} candidate turn(s), "
            f"complete={interview_complete} "
            f"({conclusion_reason}), stalls="
            f"{interview_status['stalls']}, recoveries="
            f"{interview_status['recoveries']}, elapsed "
            f"{time.monotonic() - drive_started:.0f}s."
        )

    # The interviewer concluded: give the app a bounded window to
    # finish its completion flow before the caller tears the page
    # down. Never runs on cap/disconnect exits (nothing to
    # complete), never hangs (hard POST_EXIT_GRACE_S bound).
    if interview_complete:
        interview_status["post_exit"] = await post_exit_grace(
            page, recorder, stages, run_dir
        )

    interview_status["complete"] = interview_complete
    if interview_complete or FORCED_TURNS is not None:
        interview_status["turns_completed"] = turn
    interview_status["reached_cap"] = reached_cap
    interview_status["conclusion_reason"] = conclusion_reason
    return interview_status


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

    # Session guards passed: clear the PREVIOUS session's
    # capture artifacts so this run can never mix old and new
    # transcripts/audio/status (a guard failure above leaves
    # the last good run's evidence untouched).
    removed = clear_previous_capture_artifacts()
    if removed:
        print(
            f"Cleared {len(removed)} artifact(s) from the "
            "previous capture run."
        )

    if TEST_MODE not in ("competency", "robustness"):
        pytest.fail(f"EMH_TEST_MODE={TEST_MODE!r} must be competency|robustness")
    if TEST_MODE == "competency" and SIMULATOR_PERSONA not in PERSONAS:
        pytest.fail(
            f"EMH_SIMULATOR_PERSONA={SIMULATOR_PERSONA!r} must be one of "
            f"{sorted(PERSONAS)}"
        )

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
    audio_manifest: list[dict] = []
    interview_status = {
        "complete": False,
        "turns_completed": 0,
        "reached_cap": False,
        "stalls": 0,
        "recoveries": 0,
        "conclusion_reason": None,
    }
    # Classified failures recorded DURING the interview (greeting
    # / per-turn stalls). They never end the interview; they are
    # reported after the AI interviewer concludes it.
    deferred: list[dict] = []

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
        # Record LiveKit data-channel traffic (transcription/
        # chat text, if the agent publishes any) and the bot's
        # remote audio (per-turn, for the stt-local fallback) -
        # see collectors/transcript_capture.py.
        await context.add_init_script(TRANSCRIPT_HOOK_JS)
        await context.add_init_script(AUDIO_RECORD_JS)
        await context.tracing.start(screenshots=True, snapshots=True)

        page = await context.new_page()
        recorder.attach(page)

        try:
            print()
            print("=" * 70)
            print(
                "BOT RESPONSIVENESS TEST (turn 0 greeting + "
                "drive until the AI interviewer concludes; "
                f"wall-clock cap {INTERVIEW_MAX_S}s)"
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

            # Candidate simulator for THIS interview: role/skills/
            # seniority from the job-candidate-details frame the
            # recorder just captured; persona or adversarial mode
            # from the environment. One simulator, one LLM.
            role = role_context_from_frame_rows(recorder.websocket_frames)
            simulator = CandidateSimulator(
                mode=TEST_MODE,
                role=role,
                persona=(
                    PERSONAS[SIMULATOR_PERSONA]
                    if TEST_MODE == "competency" else None
                ),
            )
            answers = LiveSimulatorAnswerSource(
                simulator,
                save_path=SIMULATOR_TURNS_PATH,
                log=stages.stamp,
            )
            stages.stamp(
                f"Candidate simulator: mode={TEST_MODE}, "
                f"model={simulator.model}, "
                + (
                    f"persona={simulator.persona.id} "
                    if simulator.persona else "adversarial specs "
                )
                + f"role={role.role if role else 'UNKNOWN (no job-candidate-details frame)'}"
            )

            # Phase 1: the bot speaks first. A missing greeting
            # is recorded as a DEFERRED classified turn-0
            # failure - the evaluator stays in the room and lets
            # the interview loop prompt the bot.
            detector_ok = await validate_greeting(
                page, context, recorder, stages, run_dir, watcher,
                collector, deferred=deferred,
            )

            # Phase 2: drive the FULL interview until the AI
            # interviewer concludes it. Stalls are recorded as
            # deferred failures (BOT STOPPED RESPONDING AT TURN N)
            # and recovered from in-room; they never end the
            # interview. interview_status is mutated IN PLACE.
            await run_multi_turn(
                page, context, recorder, stages, run_dir, watcher,
                answers, detector_ok, turn_reports,
                collector, audio_records,
                interview_status=interview_status,
                audio_manifest=audio_manifest,
                deferred=deferred,
            )

            # A whole interview that never reached the bot's
            # conclusion signal (wall-clock / hard cap hit) is a
            # truncated interview - fail rather than pass a
            # partial transcript as success.
            if not interview_status["complete"]:
                disconnected = interview_status.get("room_disconnected")
                message = await capture_failure(
                    page, context, recorder, stages, run_dir,
                    label=(
                        ROOM_DISCONNECTED if disconnected
                        else "INTERVIEW DID NOT COMPLETE"
                    ),
                    reason=(
                        "The interview ran "
                        f"{interview_status['turns_completed']} "
                        "candidate turn(s) but never reached an "
                        "interview-complete signal "
                        + (
                            "- the server tore the room down "
                            "(LiveKit peer connections closed / "
                            "socket.io transport close) and never "
                            f"reconnected within "
                            f"{ROOM_DISCONNECT_GRACE_S}s. ROOM/"
                            "ENVIRONMENT condition, not a bot-"
                            "response failure. "
                            if disconnected else
                            f"(reached_cap={interview_status['reached_cap']}). "
                        )
                        + "The captured transcript is truncated and "
                        "must not be used for whole-interview "
                        "evaluation."
                    ),
                    likely_causes=[
                        "The agent stalled/died mid-interview "
                        "(server-side STT/AI/TTS) and never "
                        "produced its closing statement despite "
                        f"{interview_status['stalls']} recorded "
                        "stall(s) with in-room recovery attempts.",
                        "The interview genuinely exceeds the wall-"
                        f"clock cap ({INTERVIEW_MAX_S}s) - raise "
                        "EMH_INTERVIEW_MAX_S if the real interview "
                        "is longer.",
                    ],
                    turn_reports=turn_reports,
                )
                if deferred:
                    message += (
                        f"\n\n{len(deferred)} deferred failure(s) "
                        "were recorded during the interview:\n"
                        + "\n".join(
                            f"  - turn {d['turn']} attempt "
                            f"{d['attempt']}: {d['label']}"
                            + (
                                f" (recovered: {d.get('recovery')})"
                                if d["recovered"] else ""
                            )
                            for d in deferred
                        )
                    )
                pytest.fail(message)

            # The AI interviewer concluded the interview: keep the
            # per-turn report and the full trace.
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / "bot_responsiveness_report.json").write_text(
                json.dumps(
                    {
                        "room": recorder.room_name,
                        "greeting_detector_ok": detector_ok,
                        "interview_complete": interview_status["complete"],
                        "conclusion_reason": (
                            interview_status["conclusion_reason"]
                        ),
                        "turns_completed": (
                            interview_status["turns_completed"]
                        ),
                        "stalls": interview_status["stalls"],
                        "recoveries": interview_status["recoveries"],
                        "deferred_failures": [
                            {k: v for k, v in d.items() if k != "message"}
                            for d in deferred
                        ],
                        "turns": turn_reports,
                    },
                    indent=2,
                )
            )
            try:
                await context.tracing.stop(
                    path=str(run_dir / "trace.zip")
                )
            except Exception:
                pass

            print()
            print("=" * 70)
            if deferred:
                unrecovered = [d for d in deferred if not d["recovered"]]
                print(
                    "INTERVIEW COMPLETED (AI interviewer concluded it "
                    f"after {interview_status['turns_completed']} "
                    f"turns) but the bot stalled {len(deferred)} "
                    f"time(s) ({len(unrecovered)} never recovered) - "
                    "reporting the deferred failures now."
                )
                print("=" * 70)
                summary = "\n".join(
                    f"  - turn {d['turn']} attempt {d['attempt']}: "
                    f"{d['label']}"
                    + (
                        f" (recovered: {d.get('recovery')})"
                        if d["recovered"] else " (NOT recovered)"
                    )
                    for d in deferred
                )
                pytest.fail(
                    "BOT RESPONSIVENESS DEGRADED - interview "
                    "completed (transcript is complete and "
                    f"scorable) but {len(deferred)} stall(s) were "
                    "recorded during the interview:\n"
                    f"{summary}\n\nFirst stall diagnostics:\n"
                    f"{deferred[0]['message']}"
                )
            print(
                "BOT RESPONSIVENESS TEST PASSED - greeting verified "
                "(bot spoke first) and the bot stayed responsive "
                "through the FULL interview until the AI "
                "interviewer concluded it "
                f"({interview_status['turns_completed']} turns, "
                f"{interview_status['conclusion_reason']})."
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
            #
            # Drain the LiveKit data-channel transcript hook
            # FIRST: it both answers whether the agent publishes
            # text (transcriptions/chat ride the WebRTC data
            # channel, invisible to DOM and WebSocket capture)
            # and, when it does, provides the preferred
            # transcript source (collectors.transcript_capture).
            try:
                dc_events = await drain_transcript_events(page)
                dc_text = [
                    e for e in dc_events
                    if e.get("ev") == "message" and e.get("text")
                ]
                print(
                    "LiveKit data-channel transcript hook: "
                    f"{len(dc_events)} events, "
                    f"{len(dc_text)} with text -> "
                    "artifacts/transcripts/"
                    "livekit_transcript_events.json"
                )
            except Exception as error:
                dc_events = []
                print(
                    "LiveKit transcript hook drain failed "
                    f"(harness issue, not a bot failure): {error}"
                )

            # Persist the final bot utterance still being
            # recorded (closing statement / last reply).
            final_clip = await stop_bot_audio_recording(
                page,
                f"bot_turn_{interview_status['turns_completed']:02d}_final",
            )
            if final_clip:
                audio_manifest.append(
                    {
                        "role": "assistant",
                        "turn": interview_status["turns_completed"],
                        "audio_path": str(final_clip),
                    }
                )

            # Local Whisper over ALL captured audio (bot turns +
            # the candidate clips actually injected) -> the
            # stt-local capture backend. Environment-limited,
            # never a test error.
            try:
                stt_summary = transcribe_captured_audio(
                    audio_manifest
                )
                print(
                    "stt-local transcription: "
                    f"{stt_summary['transcribed']}/"
                    f"{stt_summary['requested']} clips"
                    + (
                        f" (skipped: {stt_summary['skipped_reason']})"
                        if stt_summary["skipped_reason"]
                        else f" -> {stt_summary['path']}"
                    )
                )
            except Exception as error:
                stt_summary = {
                    "requested": len(audio_manifest),
                    "transcribed": 0,
                    "skipped_reason": f"transcription error: {error}",
                }
                print(
                    "stt-local transcription failed "
                    f"(environment issue, not a bot failure): {error}"
                )

            # One evidence manifest per run: which sources
            # produced usable bot text, which were empty.
            try:
                evidence = write_capture_evidence(
                    dom_metadata=[
                        {
                            "role": captured.role,
                            "content": captured.content,
                        }
                        for captured in collector.turns
                    ],
                    datachannel_events=dc_events,
                    socketio_frames=[
                        frame
                        for frame in recorder.websocket_frames
                        if "socket.io" in (frame.get("url") or "")
                    ],
                    audio_files=[
                        entry["audio_path"]
                        for entry in audio_manifest
                        if entry["role"] == "assistant"
                    ],
                    stt_summary=stt_summary,
                )
                usable = [
                    name
                    for name, data in evidence["sources"].items()
                    if data.get("usable_bot_text")
                ]
                print(
                    "Capture evidence -> artifacts/"
                    "capture_evidence.json | sources with usable "
                    f"bot text: {usable or 'NONE'}"
                )
            except Exception as error:
                print(
                    "capture_evidence.json write failed "
                    f"(harness issue): {error}"
                )

            # Deferred (non-terminal) failure log for the report
            # layer - full diagnostics per stall.
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "deferred_failures.json").write_text(
                    json.dumps(deferred, indent=2)
                )
            except Exception:
                pass

            transcript_path = collector.save()
            records_path = save_audio_turn_records(audio_records)
            status_path = save_transcript_status(
                complete=interview_status["complete"],
                turn_count=interview_status["turns_completed"],
                reached_cap=interview_status["reached_cap"],
                captured_at=time.time(),
                room_disconnected=bool(
                    interview_status.get("room_disconnected")
                ),
                conclusion_reason=interview_status.get(
                    "conclusion_reason"
                ),
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
