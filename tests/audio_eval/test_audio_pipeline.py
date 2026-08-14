"""
Unit tests for the audio evaluation layer that connects
audio -> jiwer/pyannote metrics -> transcript -> the existing
Nemotron rubric evaluator.

Fully local and deterministic. run_full_evaluation is always
called with run_nemotron=False here, so the NVIDIA client is
never imported and no credits are consumed.
"""

import pytest

from evaluation.audio_pipeline import (
    AUDIO_FIXTURE_DIR,
    build_transcript_turns,
    evaluate_audio_turn,
    load_audio_turn_record,
    run_full_evaluation,
    to_evaluator_transcript,
)


FIXTURES = [
    "perfect_transcription.json",
    "transcription_errors.json",
    "background_speech_contamination.json",
    "correct_speaker_attribution.json",
    "diarization_errors.json",
    "missing_reference_data.json",
]


def load_all_records():
    return [
        load_audio_turn_record(AUDIO_FIXTURE_DIR / name)
        for name in FIXTURES
    ]


def test_clean_turn_raises_no_interference_suspicion():
    record = load_audio_turn_record(
        AUDIO_FIXTURE_DIR / "perfect_transcription.json"
    )

    evaluation = evaluate_audio_turn(record)

    assert evaluation.stt is not None
    assert evaluation.stt.wer == 0.0
    assert evaluation.diarization.available
    assert not evaluation.background_interference_suspected
    assert evaluation.interference_reasons == []


def test_contaminated_turn_is_flagged_with_reasons():
    record = load_audio_turn_record(
        AUDIO_FIXTURE_DIR / "background_speech_contamination.json"
    )

    evaluation = evaluate_audio_turn(record)

    # Contamination (overlap + misattribution) AND degraded STT
    # together are the freeze-hypothesis signature.
    assert evaluation.background_interference_suspected
    reasons = " ".join(evaluation.interference_reasons).lower()
    assert "overlaps candidate" in reasons
    assert "background speech to the candidate" in reasons
    assert "wer" in reasons


def test_correctly_attributed_background_is_not_flagged():
    record = load_audio_turn_record(
        AUDIO_FIXTURE_DIR / "correct_speaker_attribution.json"
    )

    evaluation = evaluate_audio_turn(record)

    assert evaluation.stt is not None
    assert evaluation.stt.wer == 0.0
    assert evaluation.diarization.available
    assert not evaluation.background_interference_suspected


def test_high_wer_alone_is_not_background_interference():
    record = load_audio_turn_record(
        AUDIO_FIXTURE_DIR / "transcription_errors.json"
    )

    evaluation = evaluate_audio_turn(record)

    # WER 3/11 exceeds the alert threshold, but with clean
    # single-speaker diarization it is an STT problem, not
    # background interference.
    assert evaluation.stt is not None
    assert evaluation.stt.wer >= 0.25
    assert not evaluation.background_interference_suspected
    assert any(
        "wer" in reason.lower()
        for reason in evaluation.interference_reasons
    )


def test_missing_reference_turn_reports_unavailable_cleanly():
    record = load_audio_turn_record(
        AUDIO_FIXTURE_DIR / "missing_reference_data.json"
    )

    evaluation = evaluate_audio_turn(record)

    assert evaluation.stt is None
    assert evaluation.stt_unavailable_reason
    assert not evaluation.diarization.available
    assert not evaluation.background_interference_suspected


def test_transcript_turns_use_stt_text_in_project_shape():
    records = load_all_records()

    turns = build_transcript_turns(records)

    # Same role/content shape as the existing transcript
    # artifacts consumed by evaluators/metrics.py and
    # evaluation/transcript.py.
    assert all(set(turn) == {"role", "content"} for turn in turns)
    assert any(turn["role"] == "assistant" for turn in turns)
    user_text = " ".join(
        turn["content"] for turn in turns if turn["role"] == "user"
    )
    # The AI heard the STT output, contamination included.
    assert "order a large pizza" in user_text


def test_evaluator_transcript_matches_existing_formatter():
    records = load_all_records()

    transcript = to_evaluator_transcript(records)

    assert "AI Interviewer:" in transcript
    assert "Candidate:" in transcript


def test_empty_records_cannot_reach_the_rubric_evaluator():
    with pytest.raises(ValueError):
        to_evaluator_transcript([])


def test_full_evaluation_without_nemotron_is_local_only():
    records = load_all_records()

    report = run_full_evaluation(records, run_nemotron=False)

    assert report["nemotron"] is None
    assert len(report["turns"]) == len(records)
    # Only the contamination fixture should be flagged.
    assert report["turns_with_suspected_interference"] == [3]
