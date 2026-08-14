"""
STT / transcription accuracy evaluation (jiwer).

Compares a reference transcript (what the candidate actually
said) against the STT transcript the interview pipeline
produced, and reports word error rate with the full error
breakdown (substitutions / deletions / insertions).

This module is deliberately separate from the Nemotron rubric
in evaluation/rubric.py: it measures the AUDIO pipeline's
transcription fidelity, not interview quality. It performs no
network calls of any kind.
"""

from dataclasses import dataclass
from typing import Any

import jiwer


class MissingReferenceError(ValueError):
    """
    Raised when there is no usable reference transcript.

    Callers (tests) must treat this as "cannot evaluate", never
    as a passing or failing WER.
    """


class MissingHypothesisError(MissingReferenceError):
    """
    Raised when hypothesis is None: the pipeline never produced
    or rendered an STT transcript for this turn at all.

    None is fundamentally different from "" (STT ran and emitted
    nothing, a legal all-deletions result): scoring None as WER
    1.0 would misattribute a capture/UI gap as STT degradation.
    Subclasses MissingReferenceError so existing "cannot
    evaluate" handlers treat both alike.
    """


# Normalization applied to both sides before alignment so
# casing/punctuation differences are not counted as STT errors.
_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


@dataclass(frozen=True)
class STTAccuracyResult:
    """Structured jiwer output suitable for pytest assertions."""

    wer: float
    mer: float
    wil: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_word_count: int
    hypothesis_word_count: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def as_dict(self) -> dict[str, Any]:
        return {
            "wer": self.wer,
            "mer": self.mer,
            "wil": self.wil,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "total_errors": self.total_errors,
            "reference_word_count": self.reference_word_count,
            "hypothesis_word_count": self.hypothesis_word_count,
        }


def evaluate_stt(
    reference: str | None,
    hypothesis: str | None,
) -> STTAccuracyResult:
    """
    Score an STT transcript against its reference.

    reference  - what the candidate actually said (ground truth,
                 e.g. the text synthesized into the fake mic).
    hypothesis - what the pipeline's STT produced.

    An absent/empty reference raises MissingReferenceError.
    A None hypothesis raises MissingHypothesisError: no STT
    transcript was ever produced/rendered for the turn, so
    accuracy is unmeasurable - NOT an all-deletions WER 1.0.
    An empty-string hypothesis is a legal (very bad) STT output:
    it scores as all-deletions rather than being masked.
    """

    if reference is None or not reference.strip():
        raise MissingReferenceError(
            "No reference transcript available - STT accuracy "
            "cannot be evaluated for this turn."
        )

    if hypothesis is None:
        raise MissingHypothesisError(
            "No STT transcript was produced/rendered for this "
            "turn - STT accuracy is unmeasurable (capture/UI "
            "gap, not proof of STT degradation)."
        )

    reference_words = _NORMALIZE(reference)[0]

    if not hypothesis.strip():
        # jiwer rejects empty hypotheses; an STT that produced
        # nothing has deleted every reference word.
        count = len(reference_words)
        return STTAccuracyResult(
            wer=1.0,
            mer=1.0,
            wil=1.0,
            substitutions=0,
            deletions=count,
            insertions=0,
            hits=0,
            reference_word_count=count,
            hypothesis_word_count=0,
        )

    output = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=_NORMALIZE,
        hypothesis_transform=_NORMALIZE,
    )

    return STTAccuracyResult(
        wer=output.wer,
        mer=output.mer,
        wil=output.wil,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        reference_word_count=len(reference_words),
        hypothesis_word_count=len(_NORMALIZE(hypothesis)[0]),
    )
