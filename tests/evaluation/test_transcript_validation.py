"""
Offline tests for the whole-interview evaluation gates
(evaluation.transcript_validation + save_transcript_status).

Covers the scenarios the evaluation model must classify
correctly WITHOUT any live interview or judge call:

  - complete interview            -> scorable
  - incomplete interview          -> skip-upstream (agent-side)
  - room/server disconnect        -> skip-upstream (environment,
                                     named as a disconnect)
  - missing / corrupt sidecar     -> fail-missing (provenance
                                     unknown, never a crash)
  - stale capture                 -> fail-stale
  - transcript/media mismatch     -> coverage rejection (a
    (complete drive, fragmentary      fragment is never judged
    capture)                          as the whole interview)
  - missing STT hypothesis        -> "cannot evaluate", never
                                     WER 1.0 (stt_accuracy)
"""

import time

import pytest

from collectors.transcript_collector import save_transcript_status
from evaluation.stt_accuracy import (
    MissingHypothesisError,
    MissingReferenceError,
    evaluate_stt,
)
from evaluation.transcript_validation import (
    TranscriptStatus,
    TranscriptValidationError,
    capture_coverage_problem,
    classify_status_for_scoring,
    load_status,
    validate_transcript_turns,
)


def _turns(assistant: int, user: int) -> list[dict]:
    turns = []
    for index in range(max(assistant, user)):
        if index < assistant:
            turns.append(
                {"role": "assistant", "content": f"Question {index}?"}
            )
        if index < user:
            turns.append(
                {"role": "user", "content": f"Answer {index}."}
            )
    return turns


def _status(
    *,
    complete=True,
    turn_count=3,
    age=60.0,
    reached_cap=False,
    room_disconnected=None,
):
    return TranscriptStatus(
        complete=complete,
        turn_count=turn_count,
        reached_cap=reached_cap,
        captured_at=time.time() - age if age is not None else None,
        age_seconds=age,
        room_disconnected=room_disconnected,
    )


# ============================================================
# Complete interview
# ============================================================

def test_complete_fresh_interview_is_scorable():
    validate_transcript_turns(
        _turns(assistant=3, user=3),
        status=_status(complete=True, turn_count=3),
    )
    verdict, _ = classify_status_for_scoring(
        _status(complete=True, turn_count=3)
    )
    assert verdict == "ok"


# ============================================================
# Incomplete / disconnected interviews are never scored
# ============================================================

def test_incomplete_interview_is_never_scored():
    with pytest.raises(TranscriptValidationError) as excinfo:
        validate_transcript_turns(
            _turns(assistant=3, user=3),
            status=_status(complete=False),
        )
    assert "INCOMPLETE" in str(excinfo.value)


def test_room_disconnect_classified_as_environment():
    verdict, reason = classify_status_for_scoring(
        _status(
            complete=False,
            room_disconnected=True,
        )
    )
    assert verdict == "skip-upstream"
    assert "DISCONNECT" in reason.upper()
    assert "environment" in reason.lower()


def test_cap_hit_classified_as_agent_side():
    verdict, reason = classify_status_for_scoring(
        _status(complete=False, reached_cap=True)
    )
    assert verdict == "skip-upstream"
    assert "cap" in reason.lower()


# ============================================================
# Missing / corrupt sidecar: provenance unknown, no crash
# ============================================================

def test_missing_sidecar_fails_at_capture_layer(tmp_path):
    status = load_status(tmp_path / "does_not_exist.json")
    assert status.complete is None
    verdict, _ = classify_status_for_scoring(status)
    assert verdict == "fail-missing"


def test_corrupt_sidecar_is_unknown_not_a_crash(tmp_path):
    path = tmp_path / "status.json"
    path.write_text("{not valid json", encoding="utf-8")
    status = load_status(path)
    assert status.complete is None
    verdict, _ = classify_status_for_scoring(status)
    assert verdict == "fail-missing"


def test_non_object_sidecar_is_unknown(tmp_path):
    path = tmp_path / "status.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_status(path).complete is None


# ============================================================
# Staleness
# ============================================================

def test_stale_capture_is_rejected():
    with pytest.raises(TranscriptValidationError) as excinfo:
        validate_transcript_turns(
            _turns(assistant=3, user=3),
            status=_status(age=8 * 3600.0),
        )
    assert "STALE" in str(excinfo.value)


# ============================================================
# Transcript/media mismatch: complete drive, fragmentary capture
# ============================================================

def test_fragmentary_capture_of_complete_drive_is_rejected():
    # The drive completed 10 candidate turns but the capture
    # channel caught only 2 - judging that fragment as the whole
    # interview would penalize uncaptured turns.
    with pytest.raises(TranscriptValidationError) as excinfo:
        validate_transcript_turns(
            _turns(assistant=3, user=2),
            status=_status(complete=True, turn_count=10),
        )
    assert "coverage" in str(excinfo.value).lower()


def test_deduplicated_long_interview_passes_coverage():
    # Cycled candidate answers legitimately collapse under
    # de-duplication: 8 captured of 12 driven is fine.
    assert (
        capture_coverage_problem(
            8, _status(complete=True, turn_count=12)
        )
        is None
    )


def test_unknown_turn_count_skips_coverage_check():
    assert (
        capture_coverage_problem(
            1, _status(complete=True, turn_count=None)
        )
        is None
    )


# ============================================================
# Sidecar round-trip (writer -> loader)
# ============================================================

def test_status_roundtrip_preserves_disconnect_context(tmp_path):
    path = tmp_path / "status.json"
    save_transcript_status(
        complete=False,
        turn_count=4,
        reached_cap=False,
        captured_at=time.time(),
        room_disconnected=True,
        conclusion_reason=None,
        path=path,
    )
    status = load_status(path)
    assert status.complete is False
    assert status.turn_count == 4
    assert status.room_disconnected is True


def test_status_roundtrip_without_new_fields(tmp_path):
    # Older callers that never pass the new kwargs still work
    # and load as "unknown cause".
    path = tmp_path / "status.json"
    save_transcript_status(
        complete=True,
        turn_count=6,
        reached_cap=False,
        captured_at=time.time(),
        path=path,
    )
    status = load_status(path)
    assert status.complete is True
    assert status.room_disconnected is None


# ============================================================
# STT failure semantics: missing media is never scored as WER
# ============================================================

def test_missing_stt_hypothesis_is_cannot_evaluate():
    # None hypothesis = the pipeline never produced STT at all
    # (capture/UI gap). It must be "cannot evaluate", never a
    # WER-1.0 "STT failed" verdict.
    with pytest.raises(MissingHypothesisError):
        evaluate_stt("what the candidate said", None)


def test_missing_reference_is_cannot_evaluate():
    with pytest.raises(MissingReferenceError):
        evaluate_stt(None, "some stt output")


def test_empty_stt_output_is_a_real_all_deletion_result():
    # "" is different from None: STT ran and emitted nothing.
    result = evaluate_stt("hello there candidate", "")
    assert result.wer == 1.0
    assert result.deletions == 3
