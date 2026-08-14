"""
Unit tests for the transcript collector. Fully local and
deterministic - no browser, no network.
"""

import json

from collectors.transcript_collector import (
    TranscriptCollector,
    looks_like_conversation,
    save_audio_turn_records,
)
from evaluation.transcript import load_real_transcript


def test_ui_noise_is_filtered_out():
    assert not looks_like_conversation("Start interview")
    assert not looks_like_conversation("Mute")
    assert not looks_like_conversation("12:34")
    assert not looks_like_conversation("   ")
    assert not looks_like_conversation("...")
    assert looks_like_conversation(
        "Hi, my name is Jamie and I am your interviewer today."
    )


def test_roles_order_and_duplicates():
    collector = TranscriptCollector()

    assert collector.record_turn(
        "assistant", "Tell me about yourself.", source="dom", turn=0
    )
    assert collector.record_turn(
        "user", "I am a software engineer.", source="injected", turn=1
    )
    # Exact duplicate (case/whitespace-insensitive) is dropped.
    assert not collector.record_turn(
        "assistant", "tell me about  yourself.", source="dom", turn=2
    )
    # Invalid role and empty content are rejected.
    assert not collector.record_turn(
        "system", "hidden", source="dom", turn=2
    )
    assert not collector.record_turn(
        "user", "   ", source="injected", turn=2
    )

    transcript = collector.as_transcript()
    assert transcript == [
        {"role": "assistant", "content": "Tell me about yourself."},
        {"role": "user", "content": "I am a software engineer."},
    ]


def test_assistant_lines_join_and_filter():
    collector = TranscriptCollector()

    added = collector.record_assistant_lines(
        [
            "Mute",
            "That sounds interesting, can you go deeper?",
            "What trade-offs did you consider there?",
            "00:42",
        ],
        source="dom-diff",
        turn=2,
    )

    assert added == 1
    assert collector.as_transcript() == [
        {
            "role": "assistant",
            "content": (
                "That sounds interesting, can you go deeper? "
                "What trade-offs did you consider there?"
            ),
        }
    ]
    # Pure-noise diff adds nothing.
    assert collector.record_assistant_lines(
        ["Mute", "12:00"], source="dom-diff", turn=3
    ) == 0


def test_save_roundtrips_through_existing_loader(tmp_path):
    collector = TranscriptCollector()
    collector.record_turn(
        "assistant", "Let's start with your introduction.",
        source="dom", turn=0,
    )
    collector.record_turn(
        "user", "I am Alex, a backend engineer.",
        source="injected", turn=1,
    )

    path = tmp_path / "actual_transcript.json"
    saved = collector.save(path)

    assert saved == path
    # The existing evaluation loader must accept the artifact.
    # validate=False: this unit test checks loader MECHANICS on
    # a temp file; freshness/completeness validation needs the
    # status sidecar a real capture run writes.
    text = load_real_transcript(path, validate=False)
    assert "AI Interviewer:" in text
    assert "Candidate:" in text
    # Capture metadata is stored alongside.
    metadata_path = tmp_path / "actual_transcript_capture_metadata.json"
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text())[0]["source"] == "dom"


def test_empty_capture_saves_nothing(tmp_path):
    collector = TranscriptCollector()

    assert collector.save(tmp_path / "t.json") is None
    assert not (tmp_path / "t.json").exists()


def test_audio_turn_records_roundtrip(tmp_path):
    path = tmp_path / "audio_turn_records.json"
    records = [
        {
            "turn": 1,
            "reference_transcript": "hello there",
            "stt_transcript": "hello there",
            "reference_segments": None,
            "detected_segments": None,
        }
    ]

    assert save_audio_turn_records(records, path) == path
    assert json.loads(path.read_text()) == records
    assert save_audio_turn_records([], tmp_path / "e.json") is None


def test_room_and_tour_chrome_is_never_bot_speech():
    # The interview-room status bar + guided-tour copy is long
    # and prose-like, so it used to pass the generic filter and
    # get recorded as an assistant turn ("dom-greeting" chrome).
    chrome = (
        "Frontend developer Elapsed time : 01 secs AI is "
        "speaking Jamie (Interviewer) Welcome to Your AI "
        "Interview Let's take a quick tour of the interview "
        "interface."
    )
    assert not looks_like_conversation(chrome)
    assert not looks_like_conversation(
        "You'll learn about the status indicators, controls, "
        "and how to interact with the AI interviewer."
    )
    # Real interviewer speech still passes.
    assert looks_like_conversation(
        "Could you walk me through the most challenging bug "
        "you have debugged recently?"
    )

    collector = TranscriptCollector()
    assert collector.record_assistant_lines(
        [chrome], source="dom-greeting", turn=0
    ) == 0
