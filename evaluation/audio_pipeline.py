"""
Audio evaluation layer for the EMH interviewer.

Connects the chain:

    audio (per-turn records)
      -> diarization / STT metrics   (pyannote.metrics / jiwer)
      -> transcript                  (evaluation.transcript format)
      -> existing Nemotron/DeepEval  (evaluation.evaluator - only
                                      when explicitly requested)

Purpose: determine whether background speech/noise could be the
reason the AI interviewer stops responding. Per turn it answers:
was the candidate's audio transcribed faithfully (WER), was
background speech present/attributed to the candidate (DER +
attribution), and does the combination make agent-side
interference plausible for that turn.

The jiwer / pyannote.metrics layer is fully local. The Nemotron
judge is NEVER imported or called unless run_full_evaluation is
invoked with run_nemotron=True, so importing this module and
running its unit tests costs no API credits.

Interference thresholds below are new to this layer and do not
alter any existing quality thresholds.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.diarization_accuracy import (
    DiarizationResult,
    SpeakerSegment,
    evaluate_diarization,
)
from evaluation.stt_accuracy import (
    MissingReferenceError,
    STTAccuracyResult,
    evaluate_stt,
)
from evaluation.transcript import format_transcript


# New, audio-layer-specific alert thresholds (independent of the
# Nemotron rubric and of every existing test threshold).
STT_WER_ALERT = 0.25
BACKGROUND_CONTAMINATION_ALERT_S = 1.0

AUDIO_FIXTURE_DIR = Path("data/audio_eval_fixtures")

# Real per-turn records written by the bot-responsiveness E2E
# run (collectors.transcript_collector.save_audio_turn_records).
REAL_AUDIO_RECORDS_PATH = Path(
    "artifacts/transcripts/audio_turn_records.json"
)

# Optional diarization sidecar. The EMH app does not expose
# speaker segments, so pyannote diarization stays "unavailable"
# on a normal run. When a real diarization pass (any tool) writes
# this file, its segments are merged into the records by turn so
# the pyannote metrics run against real data.
#   {"turns": [{"turn": 1,
#               "reference_segments": [{start,end,speaker}, ...],
#               "detected_segments":  [{start,end,speaker}, ...]}]}
DIARIZATION_SIDECAR_PATH = Path(
    "artifacts/transcripts/diarization_segments.json"
)


@dataclass
class AudioTurnRecord:
    """
    One interview turn's audio ground truth and pipeline output.

    reference_transcript - what the candidate actually said
    stt_transcript       - what the pipeline's STT produced
    reference_segments   - who really spoke when (candidate /
                           background), or None if unavailable
    detected_segments    - the pipeline's diarization output,
                           or None if unavailable
    """

    turn: int
    reference_transcript: str | None = None
    stt_transcript: str | None = None
    reference_segments: list[SpeakerSegment] | None = None
    detected_segments: list[SpeakerSegment] | None = None
    interviewer_prompt: str | None = None
    audio_path: str | None = None


@dataclass
class AudioTurnEvaluation:
    """Structured per-turn audio verdict suitable for pytest."""

    turn: int
    stt: STTAccuracyResult | None
    stt_unavailable_reason: str | None
    diarization: DiarizationResult
    background_interference_suspected: bool
    interference_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "stt": self.stt.as_dict() if self.stt else None,
            "stt_unavailable_reason": self.stt_unavailable_reason,
            "diarization": self.diarization.as_dict(),
            "background_interference_suspected":
                self.background_interference_suspected,
            "interference_reasons": self.interference_reasons,
        }


def _record_from_dict(data: dict) -> AudioTurnRecord:
    def parse_segments(key: str) -> list[SpeakerSegment] | None:
        raw = data.get(key)
        if raw is None:
            return None
        return [
            SpeakerSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker=str(item["speaker"]),
            )
            for item in raw
        ]

    return AudioTurnRecord(
        turn=int(data.get("turn", 0)),
        reference_transcript=data.get("reference_transcript"),
        stt_transcript=data.get("stt_transcript"),
        reference_segments=parse_segments("reference_segments"),
        detected_segments=parse_segments("detected_segments"),
        interviewer_prompt=data.get("interviewer_prompt"),
        audio_path=data.get("audio_path"),
    )


def load_audio_turn_record(path: Path) -> AudioTurnRecord:
    """Load a single fixture / captured-turn JSON into a record."""

    with Path(path).open(encoding="utf-8") as file:
        return _record_from_dict(json.load(file))


def load_audio_turn_records(
    path: Path = REAL_AUDIO_RECORDS_PATH,
) -> list[AudioTurnRecord]:
    """
    Load the real per-turn records captured by the E2E run
    (a JSON list). Raises FileNotFoundError when no capture
    exists - callers decide whether that means skip or fail.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No captured audio turn records at {path}. Run:\n"
            "    pytest tests/e2e/test_bot_responsiveness.py -s"
        )

    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of records.")

    records = [_record_from_dict(item) for item in data]
    return merge_diarization_sidecar(records)


def _parse_segment_list(raw) -> list[SpeakerSegment] | None:
    if raw is None:
        return None
    return [
        SpeakerSegment(
            start=float(item["start"]),
            end=float(item["end"]),
            speaker=str(item["speaker"]),
        )
        for item in raw
    ]


def merge_diarization_sidecar(
    records: list[AudioTurnRecord],
    path: Path = DIARIZATION_SIDECAR_PATH,
) -> list[AudioTurnRecord]:
    """
    Merge real speaker segments from the diarization sidecar into
    matching turns (by turn number). Absent sidecar -> records
    returned unchanged (diarization stays honestly unavailable).
    """

    path = Path(path)
    if not path.exists():
        return records

    with path.open(encoding="utf-8") as file:
        sidecar = json.load(file)

    by_turn = {
        int(entry["turn"]): entry
        for entry in sidecar.get("turns", [])
    }
    for record in records:
        entry = by_turn.get(record.turn)
        if not entry:
            continue
        reference = _parse_segment_list(entry.get("reference_segments"))
        detected = _parse_segment_list(entry.get("detected_segments"))
        if reference is not None:
            record.reference_segments = reference
        if detected is not None:
            record.detected_segments = detected
    return records


def evaluate_audio_turn(record: AudioTurnRecord) -> AudioTurnEvaluation:
    """
    Run the local audio metrics for one turn and decide whether
    background interference plausibly degraded it.
    """

    stt_result: STTAccuracyResult | None = None
    stt_unavailable: str | None = None
    try:
        stt_result = evaluate_stt(
            record.reference_transcript, record.stt_transcript
        )
    except MissingReferenceError as error:
        stt_unavailable = str(error)

    diarization = evaluate_diarization(
        record.reference_segments, record.detected_segments
    )

    reasons: list[str] = []

    if diarization.available:
        if (
            diarization.background_overlap_candidate_s
            >= BACKGROUND_CONTAMINATION_ALERT_S
        ):
            reasons.append(
                "background speech overlaps candidate speech for "
                f"{diarization.background_overlap_candidate_s:.1f}s"
            )
        if (
            diarization.background_attributed_to_candidate_s
            >= BACKGROUND_CONTAMINATION_ALERT_S
        ):
            reasons.append(
                "diarization attributes "
                f"{diarization.background_attributed_to_candidate_s:.1f}s "
                "of background speech to the candidate"
            )

    if stt_result is not None and stt_result.wer >= STT_WER_ALERT:
        reasons.append(
            f"STT WER {stt_result.wer:.2f} is at or above the "
            f"{STT_WER_ALERT:.2f} alert threshold"
        )

    # Interference needs BOTH an audio-side cause (contamination
    # or misattribution) and a degraded-transcription effect;
    # a high WER alone is an STT problem, not proof of background
    # interference.
    contamination = any(
        "background" in reason for reason in reasons
    )
    degraded = stt_result is not None and stt_result.wer >= STT_WER_ALERT

    return AudioTurnEvaluation(
        turn=record.turn,
        stt=stt_result,
        stt_unavailable_reason=stt_unavailable,
        diarization=diarization,
        background_interference_suspected=contamination and degraded,
        interference_reasons=reasons,
    )


def evaluate_audio_turns(
    records: list[AudioTurnRecord],
) -> list[AudioTurnEvaluation]:
    return [evaluate_audio_turn(record) for record in records]


def build_transcript_turns(
    records: list[AudioTurnRecord],
) -> list[dict[str, str]]:
    """
    Convert audio turn records into the role/content turn list
    used across the existing evaluation stack (same shape as
    artifacts/transcripts/actual_transcript.json). The STT
    transcript is used as the candidate's words - that is what
    the AI interviewer actually 'heard'.
    """

    turns: list[dict[str, str]] = []
    for record in records:
        if record.interviewer_prompt:
            turns.append(
                {
                    "role": "assistant",
                    "content": record.interviewer_prompt,
                }
            )
        if record.stt_transcript:
            turns.append(
                {"role": "user", "content": record.stt_transcript}
            )
    return turns


def to_evaluator_transcript(records: list[AudioTurnRecord]) -> str:
    """
    Produce the plain-text transcript the existing Nemotron
    rubric evaluator consumes (evaluation.evaluator
    .evaluate_interview), via the same formatter the rest of the
    project uses.
    """

    turns = build_transcript_turns(records)
    if not turns:
        raise ValueError(
            "No transcript content in the audio turn records - "
            "nothing to hand to the rubric evaluator."
        )
    return format_transcript(turns)


def run_full_evaluation(
    records: list[AudioTurnRecord],
    *,
    run_nemotron: bool = False,
) -> dict[str, Any]:
    """
    The full audio evaluation layer.

    Always runs the local jiwer/pyannote metrics. Only when
    run_nemotron=True does it import and call the existing
    Nemotron rubric evaluator (evaluation.evaluator) on the
    STT-derived transcript - the import happens lazily HERE so
    that module import and unit tests never touch the NVIDIA
    client or require credits.
    """

    evaluations = evaluate_audio_turns(records)

    report: dict[str, Any] = {
        "turns": [evaluation.as_dict() for evaluation in evaluations],
        "turns_with_suspected_interference": [
            evaluation.turn for evaluation in evaluations
            if evaluation.background_interference_suspected
        ],
        "nemotron": None,
    }

    if run_nemotron:
        # Deferred import: evaluation.evaluator pulls in the
        # NVIDIA client at module import time.
        from evaluation.evaluator import evaluate_interview

        report["nemotron"] = evaluate_interview(
            to_evaluator_transcript(records)
        )

    return report
