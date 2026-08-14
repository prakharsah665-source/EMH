"""
Speaker diarization evaluation (pyannote.metrics).

Compares reference speaker segments (who really spoke when:
the candidate vs background speech) against the segments the
pipeline detected, and reports DER with its components plus
the attribution signals that matter for the EMH freeze
hypothesis: how much background speech overlaps the candidate,
and how much of it got attributed TO the candidate - the
mechanism by which background speech could corrupt STT input
and make the interviewer stop responding.

Separate from the Nemotron rubric; no network calls.

Reference speaker labels use CANDIDATE_SPEAKER / any other
label (e.g. "background") for interference. Detected labels
are arbitrary (diarization systems emit SPEAKER_00 style
labels); the optimal DER mapping links them to reference
speakers.
"""

from dataclasses import dataclass, field
from typing import Any

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    JaccardErrorRate,
)


CANDIDATE_SPEAKER = "candidate"


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(
                f"Malformed speaker segment: start={self.start} "
                f"end={self.end} (end must be > start)."
            )

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """
    Structured diarization verdict.

    available=False (with a reason) means the evaluation could
    not run - missing reference or missing diarization output.
    Metric fields are only meaningful when available is True;
    tests must check `available` first, never read a default 0.0
    as "perfect diarization".
    """

    available: bool
    reason: str | None = None

    der: float = 0.0
    jer: float = 0.0
    confusion_s: float = 0.0
    missed_detection_s: float = 0.0
    false_alarm_s: float = 0.0
    correct_s: float = 0.0
    total_reference_speech_s: float = 0.0

    # Detected label -> reference speaker, from the optimal DER
    # mapping (e.g. {"SPEAKER_00": "candidate"}).
    speaker_mapping: dict[str, str] = field(default_factory=dict)

    candidate_talk_time_s: float = 0.0
    background_talk_time_s: float = 0.0

    # Background speech overlapping candidate speech in the
    # REFERENCE (simultaneous talkers - the contamination risk).
    background_overlap_candidate_s: float = 0.0

    # Reference background speech that the DETECTED segments
    # attribute to the candidate (after optimal mapping) - i.e.
    # background words the pipeline would feed into the
    # candidate's STT stream.
    background_attributed_to_candidate_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "der": self.der,
            "jer": self.jer,
            "confusion_s": self.confusion_s,
            "missed_detection_s": self.missed_detection_s,
            "false_alarm_s": self.false_alarm_s,
            "correct_s": self.correct_s,
            "total_reference_speech_s": self.total_reference_speech_s,
            "speaker_mapping": self.speaker_mapping,
            "candidate_talk_time_s": self.candidate_talk_time_s,
            "background_talk_time_s": self.background_talk_time_s,
            "background_overlap_candidate_s":
                self.background_overlap_candidate_s,
            "background_attributed_to_candidate_s":
                self.background_attributed_to_candidate_s,
        }


def _to_annotation(segments: list[SpeakerSegment]) -> Annotation:
    annotation = Annotation()
    for segment in segments:
        annotation[Segment(segment.start, segment.end)] = segment.speaker
    return annotation


def _overlap_duration(
    first: list[SpeakerSegment], second: list[SpeakerSegment]
) -> float:
    total = 0.0
    for a in first:
        for b in second:
            total += max(
                0.0, min(a.end, b.end) - max(a.start, b.start)
            )
    return total


def evaluate_diarization(
    reference_segments: list[SpeakerSegment] | None,
    detected_segments: list[SpeakerSegment] | None,
) -> DiarizationResult:
    """
    Score detected speaker segments against the reference.

    Missing reference or missing diarization output returns an
    unavailable result with an explicit reason instead of a fake
    score. Malformed segments raise (handled at construction by
    SpeakerSegment).
    """

    if not reference_segments:
        return DiarizationResult(
            available=False,
            reason=(
                "No reference speaker segments - diarization "
                "accuracy cannot be evaluated."
            ),
        )

    if not detected_segments:
        return DiarizationResult(
            available=False,
            reason=(
                "No detected speaker segments - the pipeline "
                "produced no diarization output to evaluate."
            ),
        )

    reference = _to_annotation(reference_segments)
    detected = _to_annotation(detected_segments)

    # Evaluate over the union extent of both annotations so
    # pyannote does not have to guess the scoring region.
    extent_end = max(
        segment.end
        for segment in reference_segments + detected_segments
    )
    uem = Segment(0.0, extent_end)

    der_metric = DiarizationErrorRate()
    detail = der_metric(reference, detected, uem=uem, detailed=True)
    jer = JaccardErrorRate()(reference, detected, uem=uem)
    mapping = {
        str(detected_label): str(reference_label)
        for detected_label, reference_label
        in der_metric.optimal_mapping(reference, detected).items()
    }

    candidate_reference = [
        segment for segment in reference_segments
        if segment.speaker == CANDIDATE_SPEAKER
    ]
    background_reference = [
        segment for segment in reference_segments
        if segment.speaker != CANDIDATE_SPEAKER
    ]

    # Detected segments whose label maps to the candidate.
    detected_as_candidate = [
        segment for segment in detected_segments
        if mapping.get(segment.speaker) == CANDIDATE_SPEAKER
    ]

    return DiarizationResult(
        available=True,
        der=detail["diarization error rate"],
        jer=jer,
        confusion_s=detail["confusion"],
        missed_detection_s=detail["missed detection"],
        false_alarm_s=detail["false alarm"],
        correct_s=detail["correct"],
        total_reference_speech_s=detail["total"],
        speaker_mapping=mapping,
        candidate_talk_time_s=sum(
            segment.duration for segment in candidate_reference
        ),
        background_talk_time_s=sum(
            segment.duration for segment in background_reference
        ),
        background_overlap_candidate_s=_overlap_duration(
            background_reference, candidate_reference
        ),
        background_attributed_to_candidate_s=_overlap_duration(
            background_reference, detected_as_candidate
        ),
    )
