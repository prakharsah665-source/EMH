"""
Unit tests for the pyannote.metrics-based diarization layer.

Fully local and deterministic: fixture JSONs in
data/audio_eval_fixtures/, no network, no NVIDIA, no browser.
"""

import pytest

from evaluation.audio_pipeline import (
    AUDIO_FIXTURE_DIR,
    load_audio_turn_record,
)
from evaluation.diarization_accuracy import (
    CANDIDATE_SPEAKER,
    SpeakerSegment,
    evaluate_diarization,
)


def load_fixture(name: str):
    return load_audio_turn_record(AUDIO_FIXTURE_DIR / name)


def test_perfect_diarization_scores_zero_der():
    record = load_fixture("perfect_transcription.json")

    result = evaluate_diarization(
        record.reference_segments, record.detected_segments
    )

    assert result.available
    assert result.der == pytest.approx(0.0)
    assert result.confusion_s == pytest.approx(0.0)
    assert result.missed_detection_s == pytest.approx(0.0)
    assert result.false_alarm_s == pytest.approx(0.0)
    assert result.speaker_mapping == {
        "SPEAKER_00": CANDIDATE_SPEAKER
    }


def test_correct_attribution_separates_background_speaker():
    record = load_fixture("correct_speaker_attribution.json")

    result = evaluate_diarization(
        record.reference_segments, record.detected_segments
    )

    assert result.available
    assert result.der == pytest.approx(0.0)
    assert result.speaker_mapping == {
        "SPEAKER_00": CANDIDATE_SPEAKER,
        "SPEAKER_01": "background",
    }
    assert result.candidate_talk_time_s == pytest.approx(5.0)
    assert result.background_talk_time_s == pytest.approx(2.0)
    # Non-overlapping background, correctly attributed: no
    # contamination of the candidate stream.
    assert result.background_overlap_candidate_s == pytest.approx(0.0)
    assert result.background_attributed_to_candidate_s == \
        pytest.approx(0.0)


def test_contamination_fixture_attributes_background_to_candidate():
    record = load_fixture("background_speech_contamination.json")

    result = evaluate_diarization(
        record.reference_segments, record.detected_segments
    )

    assert result.available
    # candidate 6s + background 2s of reference speech; the
    # single detected speaker maps to the candidate, so the
    # overlapped background 2s is missed by diarization AND
    # attributed to the candidate stream.
    assert result.total_reference_speech_s == pytest.approx(8.0)
    assert result.speaker_mapping == {
        "SPEAKER_00": CANDIDATE_SPEAKER
    }
    assert result.missed_detection_s == pytest.approx(2.0)
    assert result.der == pytest.approx(0.25)
    assert result.background_overlap_candidate_s == pytest.approx(2.0)
    assert result.background_attributed_to_candidate_s == \
        pytest.approx(2.0)


def test_diarization_errors_break_down_into_der_components():
    record = load_fixture("diarization_errors.json")

    result = evaluate_diarization(
        record.reference_segments, record.detected_segments
    )

    assert result.available
    # Reference: candidate 0-5, background 6-8 (total 7s).
    # Detected: one speaker 0-7.5 mapped to the candidate.
    assert result.confusion_s == pytest.approx(1.5)
    assert result.missed_detection_s == pytest.approx(0.5)
    assert result.false_alarm_s == pytest.approx(1.0)
    assert result.der == pytest.approx(3.0 / 7.0)
    assert result.jer > 0.0
    assert result.background_attributed_to_candidate_s == \
        pytest.approx(1.5)


def test_missing_reference_yields_unavailable_not_perfect():
    record = load_fixture("missing_reference_data.json")

    result = evaluate_diarization(
        record.reference_segments, record.detected_segments
    )

    assert not result.available
    assert "reference" in (result.reason or "").lower()


def test_missing_detected_segments_yields_unavailable():
    reference = [SpeakerSegment(0.0, 5.0, CANDIDATE_SPEAKER)]

    result = evaluate_diarization(reference, None)

    assert not result.available
    assert "detected" in (result.reason or "").lower()


def test_malformed_segment_raises():
    with pytest.raises(ValueError):
        SpeakerSegment(start=5.0, end=5.0, speaker="candidate")


def test_result_serializes_for_reports():
    record = load_fixture("correct_speaker_attribution.json")

    payload = evaluate_diarization(
        record.reference_segments, record.detected_segments
    ).as_dict()

    assert payload["available"] is True
    assert payload["der"] == pytest.approx(0.0)
    assert payload["speaker_mapping"]["SPEAKER_01"] == "background"
