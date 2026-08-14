"""
Offline unit tests for the TranscriptCapture backends and PII
redaction. No browser, no network.
"""

import json

import pytest

from collectors.transcript_capture import (
    AppApiCapture,
    DomScrapeCapture,
    LiveKitEventsCapture,
    SttCapture,
    select_capture,
)
from evaluation.redaction import redact_pii


# ============================================================
# Redaction
# ============================================================

def test_redaction_masks_pii_and_secrets():
    text = (
        "Contact alex@example.com or +91 74772 21171. Token "
        "eyJhbGciOiJIUzI1NiJ9.eyJjYW5kaWRhdGVfaWQiOjF9.abc12345 "
        "and card 4111111111111111."
    )
    out = redact_pii(text)
    assert "alex@example.com" not in out
    assert "74772" not in out
    assert "eyJhbGciOiJIUzI1NiJ9" not in out
    assert "4111111111111111" not in out
    assert "[REDACTED-EMAIL]" in out
    assert "[REDACTED-JWT]" in out


def test_redaction_preserves_technical_content():
    text = (
        "p95 latency 0.25s, version 1.0.25, scored 0.75 on "
        "3 of 11 turns at 44100 Hz"
    )
    assert redact_pii(text) == text


# ============================================================
# LiveKitEventsCapture
# ============================================================

def test_livekit_capture_parses_string_transcription(tmp_path):
    events = [
        {"ev": "channel", "label": "_reliable", "origin": "local"},
        {
            "ev": "message", "label": "lk.transcription",
            "kind": "string", "ts": 1000,
            "text": json.dumps(
                {"text": "Tell me about your last project.",
                 "participantIdentity": "agent-jamie"}
            ),
        },
        {
            "ev": "message", "label": "lk.transcription",
            "kind": "string", "ts": 2000,
            "text": json.dumps(
                {"text": "I built a data pipeline.",
                 "participantIdentity": "candidate-8671"}
            ),
        },
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events))

    capture = LiveKitEventsCapture(path)
    assert capture.available()
    turns = capture.get_turns()

    assert [t["role"] for t in turns] == ["assistant", "user"]
    assert turns[0]["text"] == "Tell me about your last project."
    assert turns[0]["confidence"] == "high"
    assert turns[0]["source"].startswith("livekit-")
    assert set(turns[0]) == {
        "role", "text", "ts", "source", "confidence"
    }


def test_livekit_capture_filters_binary_framing_noise(tmp_path):
    events = [
        {
            "ev": "message", "label": "_reliable",
            "kind": "binary", "ts": 1000,
            # framing-only printable runs must not become turns
            "text": "PA_abc123 TR_def456 lk.transcription",
        },
        {
            "ev": "message", "label": "_reliable",
            "kind": "binary", "ts": 2000,
            "text": "PA_abc123 What draws you to backend work?",
        },
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events))

    turns = LiveKitEventsCapture(path).get_turns()
    assert len(turns) == 1
    assert turns[0]["text"] == "What draws you to backend work?"
    assert turns[0]["confidence"] == "medium"


def test_livekit_capture_unavailable_without_file(tmp_path):
    capture = LiveKitEventsCapture(tmp_path / "missing.json")
    assert not capture.available()
    assert capture.get_turns() == []


# ============================================================
# DomScrapeCapture
# ============================================================

def test_dom_capture_maps_source_to_confidence(tmp_path):
    metadata = [
        {"turn": 0, "role": "assistant", "source": "dom-greeting",
         "content": "Welcome to Your AI Interview tour text"},
        {"turn": 1, "role": "user", "source": "injected-audio",
         "content": "Hello, I am Alex."},
        {"turn": 2, "role": "user", "source": "app-stt",
         "content": "hello i am alex"},
        {"turn": 3, "role": "assistant", "source": "dom-diff",
         "content": "What did you build most recently?"},
    ]
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(metadata))

    capture = DomScrapeCapture(path)
    assert capture.available()
    turns = capture.get_turns()

    by_source = {t["source"]: t["confidence"] for t in turns}
    assert by_source == {
        "dom-greeting": "low",
        "injected-audio": "low",
        "app-stt": "high",
        "dom-diff": "medium",
    }


# ============================================================
# Selection
# ============================================================

def test_select_capture_explicit_and_unknown(monkeypatch):
    monkeypatch.setenv("EMH_TRANSCRIPT_CAPTURE", "stt")
    assert isinstance(select_capture(), SttCapture)

    monkeypatch.setenv("EMH_TRANSCRIPT_CAPTURE", "nope")
    with pytest.raises(RuntimeError, match="Unknown"):
        select_capture()


def test_select_capture_auto_reports_capture_layer(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EMH_TRANSCRIPT_CAPTURE", "auto")
    monkeypatch.delenv("EMH_TRANSCRIPT_API_URL", raising=False)
    # Point both file-backed backends at empty locations.
    monkeypatch.setattr(
        "collectors.transcript_capture.LIVEKIT_EVENTS_PATH",
        tmp_path / "lk.json",
    )
    monkeypatch.setattr(
        "collectors.transcript_capture.DOM_METADATA_PATH",
        tmp_path / "dom.json",
    )
    monkeypatch.setattr(
        "collectors.transcript_capture.STT_LOCAL_PATH",
        tmp_path / "stt.json",
    )
    with pytest.raises(RuntimeError) as excinfo:
        select_capture()
    # The failure names the capture layer, never the bot.
    assert "not a bot failure" in str(excinfo.value)


def test_app_api_capture_unavailable_without_env(monkeypatch):
    monkeypatch.delenv("EMH_TRANSCRIPT_API_URL", raising=False)
    assert not AppApiCapture().available()


# ============================================================
# SttCapture (stt-local fallback)
# ============================================================

def test_stt_capture_serves_transcribed_turns(tmp_path):
    manifest = [
        {"turn": 0, "role": "assistant",
         "audio_path": "artifacts/audio/bot_turn_00_greeting.webm",
         "text": "Hi, I'm Jamie. Let's start with your introduction."},
        {"turn": 1, "role": "user",
         "audio_path": "data/audio_fixtures/answer_01.wav",
         "text": "Hello Jamie, I'm Alex, a backend engineer."},
        {"turn": 1, "role": "assistant",
         "audio_path": "artifacts/audio/bot_turn_01.webm",
         "text": "Great. What drew you to backend work?"},
        {"turn": 2, "role": "user",
         "audio_path": "x.wav", "text": "   "},  # empty -> dropped
    ]
    path = tmp_path / "stt_local_transcript.json"
    path.write_text(json.dumps(manifest))

    capture = SttCapture(path)
    assert capture.available()
    turns = capture.get_turns()

    assert [t["role"] for t in turns] == [
        "assistant", "user", "assistant"
    ]
    assert all(t["source"] == "stt-local" for t in turns)
    assert all(t["confidence"] == "medium" for t in turns)
    # Conversational order: greeting, answer 1, reply 1.
    assert "introduction" in turns[0]["text"]
    assert "backend engineer" in turns[1]["text"]


def test_stt_capture_unavailable_without_text(tmp_path):
    path = tmp_path / "stt.json"
    path.write_text(json.dumps(
        [{"turn": 1, "role": "assistant", "text": ""}]
    ))
    assert not SttCapture(path).available()
    assert not SttCapture(tmp_path / "missing.json").available()


def test_auto_order_prefers_stt_over_dom_never_over_livekit():
    from collectors.transcript_capture import AUTO_ORDER

    assert AUTO_ORDER.index("livekit") < AUTO_ORDER.index("stt")
    assert AUTO_ORDER.index("app-api") < AUTO_ORDER.index("stt")
    assert AUTO_ORDER.index("stt") < AUTO_ORDER.index("dom")


def test_capture_evidence_manifest(tmp_path, monkeypatch):
    from collectors import transcript_capture as tc

    monkeypatch.setattr(
        tc, "CAPTURE_EVIDENCE_PATH",
        tmp_path / "capture_evidence.json",
    )
    stt_path = tmp_path / "stt.json"
    stt_path.write_text(json.dumps(
        [{"turn": 0, "role": "assistant", "text": "Hello there."}]
    ))
    monkeypatch.setattr(tc, "STT_LOCAL_PATH", stt_path)

    evidence = tc.write_capture_evidence(
        dom_metadata=[{"role": "user", "content": "hi"}],
        datachannel_events=[{"ev": "channel", "label": "x"}],
        socketio_frames=[
            {"payload": '42["bot-speech-ended",{"isExit":true}]'}
        ],
        audio_files=["artifacts/audio/bot_turn_00.webm"],
        stt_summary={"requested": 2, "transcribed": 2,
                     "skipped_reason": None},
    )

    sources = evidence["sources"]
    assert sources["dom"]["usable_bot_text"] is False
    assert sources["livekit_datachannel"]["usable_bot_text"] is False
    assert sources["socketio"]["event_names"] == ["bot-speech-ended"]
    assert sources["stt_local"]["usable_bot_text"] is True
    assert sources["remote_audio"]["count"] == 1
    assert (tmp_path / "capture_evidence.json").exists()


def test_livekit_capture_rejects_agent_session_framing(tmp_path):
    # Real payloads from a live 'lk.agent.session' data-stream
    # capture (2026-08-14): ids/topics/mimetypes only. They
    # must neither make the backend "available" nor become
    # judged bot turns.
    events = [
        {"ev": "message", "label": "_reliable", "kind": "binary",
         "text": ("agent-AJ_uTHzMTVLZ7Mb PA_7wQqvUykrYKsjl "
                  "$eaa5aec7-7d41-4e5c-a0af-76b6dc4b49a3 "
                  "lk.agent.session\" application/octet-streamR "
                  "AS_76b4ca20461e")},
        {"ev": "message", "label": "_reliable", "kind": "binary",
         "text": ("agent-AJ_uTHzMTVLZ7Mb PA_7wQqvUykrYKsre "
                  "$eaa5aec7 item_39e6acd0aade "
                  "emh_interview_agent")},
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events))

    capture = LiveKitEventsCapture(path)
    assert not capture.available()
    assert capture.get_turns() == []

    # A genuine spoken sentence inside a binary run still
    # passes.
    events.append(
        {"ev": "message", "label": "_reliable", "kind": "binary",
         "text": ("PA_7wQqvUykrYKsjl Can you tell me about the "
                  "most difficult bug you have fixed recently")}
    )
    path.write_text(json.dumps(events))
    capture = LiveKitEventsCapture(path)
    assert capture.available()
    turns = capture.get_turns()
    assert len(turns) == 1
    assert "difficult bug" in turns[0]["text"]
