"""
Unit tests for the jiwer-based STT accuracy layer.

Fully local and deterministic: fixture JSONs in
data/audio_eval_fixtures/, no network, no NVIDIA, no browser.
"""

from pathlib import Path

import pytest

from evaluation.audio_pipeline import (
    AUDIO_FIXTURE_DIR,
    load_audio_turn_record,
)
from evaluation.stt_accuracy import (
    MissingHypothesisError,
    MissingReferenceError,
    evaluate_stt,
)


def load_fixture(name: str):
    return load_audio_turn_record(AUDIO_FIXTURE_DIR / name)


def test_perfect_transcription_scores_zero_wer():
    record = load_fixture("perfect_transcription.json")

    result = evaluate_stt(
        record.reference_transcript, record.stt_transcript
    )

    assert result.wer == 0.0
    assert result.substitutions == 0
    assert result.deletions == 0
    assert result.insertions == 0
    assert result.hits == result.reference_word_count == 15


def test_known_errors_are_counted_exactly():
    record = load_fixture("transcription_errors.json")

    result = evaluate_stt(
        record.reference_transcript, record.stt_transcript
    )

    # 'years'->'ears' substitution, 'in' deletion, 'now'
    # insertion over an 11-word reference.
    assert result.substitutions == 1
    assert result.deletions == 1
    assert result.insertions == 1
    assert result.reference_word_count == 11
    assert result.wer == pytest.approx(3 / 11)
    assert result.total_errors == 3


def test_background_contamination_shows_up_as_insertions():
    record = load_fixture("background_speech_contamination.json")

    result = evaluate_stt(
        record.reference_transcript, record.stt_transcript
    )

    # The background phrase "order a large pizza" leaks in as
    # 4 inserted words over an 8-word reference.
    assert result.insertions == 4
    assert result.substitutions == 0
    assert result.deletions == 0
    assert result.wer == pytest.approx(0.5)


def test_missing_reference_raises_not_scores():
    record = load_fixture("missing_reference_data.json")

    with pytest.raises(MissingReferenceError):
        evaluate_stt(
            record.reference_transcript, record.stt_transcript
        )


def test_none_hypothesis_raises_not_scores():
    # None = the pipeline never rendered an STT transcript.
    # It must be "cannot evaluate", never a scored WER 1.0
    # that misattributes a capture gap as STT degradation.
    with pytest.raises(MissingHypothesisError):
        evaluate_stt("one two three", None)


def test_empty_hypothesis_scores_as_all_deletions():
    result = evaluate_stt("one two three", "")

    assert result.wer == 1.0
    assert result.deletions == 3
    assert result.hits == 0
    assert result.hypothesis_word_count == 0


def test_normalization_ignores_case_and_punctuation():
    result = evaluate_stt("Hello, World!", "hello world")

    assert result.wer == 0.0


def test_result_serializes_for_reports():
    record = load_fixture("transcription_errors.json")

    result = evaluate_stt(
        record.reference_transcript, record.stt_transcript
    )
    payload = result.as_dict()

    assert payload["wer"] == pytest.approx(3 / 11)
    assert payload["substitutions"] == 1
    assert payload["deletions"] == 1
    assert payload["insertions"] == 1
    assert payload["total_errors"] == 3


def test_fixture_directory_exists_and_is_complete():
    expected = {
        "perfect_transcription.json",
        "transcription_errors.json",
        "background_speech_contamination.json",
        "correct_speaker_attribution.json",
        "diarization_errors.json",
        "missing_reference_data.json",
    }
    present = {
        path.name for path in Path(AUDIO_FIXTURE_DIR).glob("*.json")
    }
    assert expected <= present
