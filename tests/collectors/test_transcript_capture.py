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
    with pytest.raises(RuntimeError) as excinfo:
        select_capture()
    # The failure names the capture layer, never the bot.
    assert "not a bot failure" in str(excinfo.value)


def test_app_api_capture_unavailable_without_env(monkeypatch):
    monkeypatch.delenv("EMH_TRANSCRIPT_API_URL", raising=False)
    assert not AppApiCapture().available()
