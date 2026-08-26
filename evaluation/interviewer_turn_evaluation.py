"""
Per-turn evaluation of the AI INTERVIEWER over the REAL captured
interview.

Unit of evaluation: ONE interviewer turn, judged against the
conversation that preceded it. For every interviewer turn the
judge reasons:

    preceding context -> this interviewer turn -> is it the
    right question/response at this point? -> evaluation -> score
    -> detected issues

Flow (every step is visible in the returned report):

    captured turns
      -> provenance filter   interviewer turns are kept; candidate
                             turns are kept ONLY as conversation
                             context and ONLY when they are genuine
                             transcription of what the interviewer
                             heard (livekit-candidate-stt, app-stt,
                             app-api, stt-local). Harness-authored
                             candidate text (source "injected-audio",
                             i.e. tests/e2e CANDIDATE_ANSWERS) is
                             REJECTED and never shown to the judge.
      -> turn units          one per interviewer turn, with the
                             verbatim preceding context; a candidate
                             response that was not captured is
                             rendered as "(candidate response not
                             captured)" - never guessed.
      -> ONE judge prompt    every interviewer turn evaluated
                             independently on relevance, clarity,
                             question quality, context awareness /
                             logical follow-up, repetition,
                             professionalism and flow alignment,
                             with an overall score, reasoning and
                             detected issues.
      -> Python validation   count/numbering, scores in 0..1, issue
                             types from the allowed set.
      -> Python aggregation  mean of the model-assigned scores,
                             issue counts, flagged turns. There is
                             no predefined score anywhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from collectors.transcript_capture import (
    AUTO_ORDER,
    _BACKENDS,
    TranscriptCapture,
)


# Candidate text sources that are NOT what the interviewer
# heard: harness-authored reference text. Never judge-visible.
FORBIDDEN_CANDIDATE_SOURCES: tuple[str, ...] = ("injected-audio",)

# Candidate context below this confidence is not shown.
MIN_CANDIDATE_CONFIDENCE = "medium"

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

NOT_CAPTURED = "(candidate response not captured)"

SCORE_DIMENSIONS = (
    "relevance",
    "clarity",
    "question_quality",
    "context_awareness",
    "repetition",
    "professionalism",
    "flow_alignment",
)

ISSUE_TYPES = (
    "incorrect",
    "irrelevant",
    "repetitive",
    "nonsensical",
    "unprofessional",
    "ignores_context",
    "scripted",
    "off_flow",
    "truncated",
)

INTENDED_FLOW = (
    "greeting and introduction request -> candidate background / "
    "experience -> role-relevant technical and behavioural "
    "questions that build on what the candidate said -> deeper "
    "follow-ups -> professional close"
)

# Judge type: (prompt, response_format) -> raw model text.
Judge = Callable[[str, dict[str, Any]], str]


# ============================================================
# Data model
# ============================================================

@dataclass
class ContextTurn:
    role: str            # "assistant" | "user"
    text: str            # verbatim captured text, or NOT_CAPTURED
    source: str | None
    confidence: str | None
    captured: bool = True


@dataclass
class InterviewerTurn:
    number: int
    text: str
    source: str
    confidence: str | None
    context: list[ContextTurn] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceReport:
    """What the evaluator was and was not allowed to see."""

    interviewer_turns: int = 0
    candidate_turns_accepted: int = 0
    candidate_turns_rejected: int = 0
    accepted_sources: dict[str, int] = field(default_factory=dict)
    rejected_sources: dict[str, int] = field(default_factory=dict)
    rejected_texts: list[str] = field(default_factory=list)

    @property
    def has_interviewer_turns(self) -> bool:
        return self.interviewer_turns > 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_interviewer_turns"] = self.has_interviewer_turns
        return data


class InterviewerEvidenceError(RuntimeError):
    """No valid interviewer transcript turns reached the evaluator."""


class TurnValidationError(ValueError):
    """The judge output violated the evaluation contract."""


# ============================================================
# Provenance filter
# ============================================================

def is_candidate_context(turn: dict[str, Any]) -> bool:
    """
    True when a user turn may be shown to the judge as what the
    candidate actually said (conversation context).
    """

    if turn.get("role") != "user":
        return False
    source = str(turn.get("source") or "")
    if source.startswith(FORBIDDEN_CANDIDATE_SOURCES):
        return False
    confidence = _CONFIDENCE_ORDER.get(turn.get("confidence"), 0)
    return confidence >= _CONFIDENCE_ORDER[MIN_CANDIDATE_CONFIDENCE]


def filter_turns(
    turns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], ProvenanceReport]:
    """
    Keep interviewer turns and evidence-grade candidate context;
    drop (and account for) everything else.
    """

    report = ProvenanceReport()
    kept: list[dict[str, Any]] = []

    for turn in turns:
        role = turn.get("role")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if role == "assistant":
            report.interviewer_turns += 1
            kept.append(turn)
            continue
        if role != "user":
            continue
        source = str(turn.get("source") or "unknown")
        if is_candidate_context(turn):
            report.candidate_turns_accepted += 1
            report.accepted_sources[source] = (
                report.accepted_sources.get(source, 0) + 1
            )
            kept.append(turn)
        else:
            report.candidate_turns_rejected += 1
            report.rejected_sources[source] = (
                report.rejected_sources.get(source, 0) + 1
            )
            report.rejected_texts.append(text)
            # Keep the TURN BOUNDARY (a candidate reply happened
            # here) without keeping the text: the judge sees
            # NOT_CAPTURED, never the rejected reference text.
            kept.append(_placeholder_turn())

    return kept, report


def _placeholder_turn() -> dict[str, Any]:
    return {
        "role": "user",
        "text": NOT_CAPTURED,
        "source": None,
        "confidence": None,
        "placeholder": True,
    }


# ============================================================
# Utterance assembly
# ============================================================

def _join_without_overlap(previous: str, current: str) -> str:
    """
    Join two consecutive captions of the same speaker. When the
    caption stream restarted mid-utterance ("It's great to hear
    about your diverse" + "great to hear about your diverse
    experience ...") the overlap is emitted once, never twice.
    """

    previous = previous.strip()
    current = current.strip()
    if not previous:
        return current
    if not current:
        return previous
    if current in previous:
        return previous
    if previous in current:
        return current
    longest = min(len(previous), len(current))
    for size in range(longest, 14, -1):
        if previous.endswith(current[:size]):
            return f"{previous}{current[size:]}".strip()
    return f"{previous} {current}"


def merge_consecutive(
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Join consecutive same-role turns (sentence-level captions,
    re-prompts) into one utterance. The merged confidence is the
    lowest among its parts; the source is the first (they are
    homogeneous per backend).
    """

    merged: list[dict[str, Any]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            previous = merged[-1]
            # Placeholders never contribute text: a real caption
            # next to a placeholder wins, two placeholders stay one.
            if turn.get("placeholder"):
                continue
            if previous.get("placeholder"):
                merged[-1] = {
                    "role": turn["role"],
                    "text": (turn.get("text") or "").strip(),
                    "source": turn.get("source"),
                    "confidence": turn.get("confidence"),
                }
                continue
            previous["text"] = _join_without_overlap(
                previous["text"], turn["text"]
            )
            if _CONFIDENCE_ORDER.get(turn.get("confidence"), 0) < (
                _CONFIDENCE_ORDER.get(previous.get("confidence"), 0)
            ):
                previous["confidence"] = turn.get("confidence")
            continue
        item = {
            "role": turn["role"],
            "text": (turn.get("text") or "").strip(),
            "source": turn.get("source"),
            "confidence": turn.get("confidence"),
        }
        if turn.get("placeholder"):
            item["placeholder"] = True
        merged.append(item)
    return merged


def build_interviewer_turns(
    turns: list[dict[str, Any]],
) -> list[InterviewerTurn]:
    """
    One evaluation unit per interviewer utterance, carrying the
    verbatim conversation that preceded it. Between two
    interviewer utterances with no captured candidate text a
    NOT_CAPTURED placeholder is inserted so the judge knows a
    response happened but was not captured - it is never
    invented.
    """

    merged = merge_consecutive(turns)
    units: list[InterviewerTurn] = []
    history: list[ContextTurn] = []

    for index, turn in enumerate(merged):
        if turn["role"] == "user":
            placeholder = bool(turn.get("placeholder"))
            history.append(
                ContextTurn(
                    role="user",
                    text=NOT_CAPTURED if placeholder else turn["text"],
                    source=None if placeholder else turn.get("source"),
                    confidence=(
                        None if placeholder else turn.get("confidence")
                    ),
                    captured=not placeholder,
                )
            )
            continue

        # Interviewer turn directly after another interviewer
        # turn: the candidate's reply (if any) was not captured.
        if history and history[-1].role == "assistant":
            history.append(
                ContextTurn(
                    role="user",
                    text=NOT_CAPTURED,
                    source=None,
                    confidence=None,
                    captured=False,
                )
            )

        units.append(
            InterviewerTurn(
                number=len(units) + 1,
                text=turn["text"],
                source=str(turn.get("source") or ""),
                confidence=turn.get("confidence"),
                context=list(history),
            )
        )
        history.append(
            ContextTurn(
                role="assistant",
                text=turn["text"],
                source=turn.get("source"),
                confidence=turn.get("confidence"),
                captured=True,
            )
        )

    return units


# ============================================================
# Prompt
# ============================================================

def format_context(context: list[ContextTurn]) -> str:
    if not context:
        return "(start of interview - no preceding conversation)"
    lines = []
    for item in context:
        if item.role == "assistant":
            lines.append(f"AI Interviewer: {item.text}")
        elif item.captured:
            lines.append(
                f"Candidate [captured STT, source={item.source}]: "
                f"{item.text}"
            )
        else:
            lines.append(f"Candidate: {NOT_CAPTURED}")
    return "\n".join(lines)


def format_turn_units(units: list[InterviewerTurn]) -> str:
    """
    The exact transcript data placed in the prompt. Only
    captured text appears here.
    """

    blocks = []
    for unit in units:
        blocks.append(
            f"--- INTERVIEWER TURN {unit.number} ---\n"
            f"PRECEDING CONTEXT:\n{format_context(unit.context)}\n\n"
            f"INTERVIEWER TURN {unit.number} TO EVALUATE "
            f"[source={unit.source}]:\n{unit.text}"
        )
    return "\n\n".join(blocks)


def build_turn_prompt(units: list[InterviewerTurn]) -> str:
    """The exact prompt sent to the evaluation model."""

    return f"""You are an impartial evaluator reviewing a REAL technical interview conducted by an AI interviewer (EMH, "Jamie"). Your job is to evaluate the AI INTERVIEWER - never the candidate.

The transcript below was captured from live audio. Interviewer turns are the interviewer's own speech captions. Candidate lines are speech-to-text transcriptions of what the candidate actually said, shown ONLY as conversation context. Where a candidate response was not captured it is marked "{NOT_CAPTURED}".

You will evaluate EACH interviewer turn INDEPENDENTLY, in order, reasoning through this chain for every turn:

  preceding context -> this interviewer turn -> is this the right question/response at this point in the interview? -> evaluation -> score -> issues

For every turn, FIRST state in "context_understanding" what the candidate actually just said (from the captured context only) and what a strong interviewer would ask next; THEN compare the actual turn against that.

ABSOLUTE RULES

1. Evaluate the interviewer's turn ONLY. Do not judge the candidate's ability, and do not let the quality of the candidate's answers change the interviewer's score except where the interviewer should have reacted to them.
2. Never generate, assume or imagine a candidate answer. If a candidate response is marked "{NOT_CAPTURED}", you do not know what was said: do NOT guess it, and do NOT penalize the interviewer for failing to reference content you cannot see. Judge only against context that is actually present.
3. Use only the text in this prompt. No outside assumptions about what the interviewer "usually" asks.
4. Speech-to-text artifacts (misspellings, dropped words, odd punctuation, homophones) are platform artifacts - read through them; never penalize them as interviewer errors.
5. Scores come from YOUR evaluation of THIS turn in THIS context. There are no expected or predefined scores. Each turn is scored independently.

INTENDED INTERVIEW FLOW
{INTENDED_FLOW}

DIMENSIONS (each 0.0 - 1.0)

- relevance:         is the turn relevant to the role/interview and to what has been discussed?
- clarity:           is the question/response clear, specific and answerable?
- question_quality:  is it a good interview question/response (probing, open, purposeful, appropriately difficult)?
- context_awareness: does it logically follow from the preceding context? Calibration: 1.0 = the QUESTION ITSELF builds on something specific the candidate said (probes their example, their claim, their technology); ~0.5 = a generic acknowledgement ("That's great...") followed by a question that would have been asked regardless of the answer; low = the acknowledgement misdescribes what the candidate said or the turn contradicts the context. Be skeptical: a polite opener is NOT context awareness. (If the prior candidate response was not captured, judge only whether the turn is coherent with the visible context.)
- repetition:        1.0 = asks something new; lower when it repeats or near-repeats an earlier interviewer question.
- professionalism:   tone, courtesy, neutrality, no rudeness or inappropriate content.
- flow_alignment:    is this the right kind of turn at this point in the intended flow (e.g. no closing mid-interview, no deep technical grilling before the introduction)?
- score:             your overall judgment of this interviewer turn, 0.0 (bad) to 1.0 (excellent).

ISSUES
For each turn list every detected issue from this set, or an empty list: {", ".join(ISSUE_TYPES)}.
  incorrect = technically wrong statement; irrelevant = off-topic for the interview/context; repetitive = repeats an earlier question; nonsensical = incoherent/garbled beyond STT artifacts; unprofessional; ignores_context = contradicts, misdescribes or ignores a captured candidate statement it should have used; scripted = generic acknowledgement plus a question unrelated to what the candidate just said (would have been asked regardless); off_flow = wrong kind of turn for this point; truncated = clearly cut off mid-sentence.

OUTPUT

Return ONLY a JSON object of the form:
{{
  "turns": [
    {{
      "number": 1,
      "context_understanding": "what has happened so far and what the interviewer should do now",
      "evaluation": {{
        "relevance":         {{"score": 0.0, "reasoning": "..."}},
        "clarity":           {{"score": 0.0, "reasoning": "..."}},
        "question_quality":  {{"score": 0.0, "reasoning": "..."}},
        "context_awareness": {{"score": 0.0, "reasoning": "..."}},
        "repetition":        {{"score": 0.0, "reasoning": "..."}},
        "professionalism":   {{"score": 0.0, "reasoning": "..."}},
        "flow_alignment":    {{"score": 0.0, "reasoning": "..."}}
      }},
      "score": 0.0,
      "reasoning": "why this score, citing the interviewer turn and the visible context",
      "issues": [],
      "issue_details": "what exactly is wrong, or empty"
    }}
  ]
}}

There are exactly {len(units)} interviewer turns. Return exactly {len(units)} entries, numbered 1..{len(units)} in order.

============================================================
INTERVIEWER TURNS WITH PRECEDING CONTEXT
============================================================

{format_turn_units(units)}
"""


def build_turn_response_schema(count: int) -> dict[str, Any]:
    """response_format json_schema enforced server-side."""

    score = {"type": "number", "minimum": 0, "maximum": 1}
    dimension = {
        "type": "object",
        "properties": {
            "score": score,
            "reasoning": {"type": "string"},
        },
        "required": ["score", "reasoning"],
        "additionalProperties": False,
    }
    turn = {
        "type": "object",
        "properties": {
            "number": {"type": "integer"},
            "context_understanding": {"type": "string"},
            "evaluation": {
                "type": "object",
                "properties": {
                    name: dimension for name in SCORE_DIMENSIONS
                },
                "required": list(SCORE_DIMENSIONS),
                "additionalProperties": False,
            },
            "score": score,
            "reasoning": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {"type": "string", "enum": list(ISSUE_TYPES)},
            },
            "issue_details": {"type": "string"},
        },
        "required": [
            "number",
            "context_understanding",
            "evaluation",
            "score",
            "reasoning",
            "issues",
            "issue_details",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "InterviewerTurnEvaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": turn,
                        "minItems": count,
                        "maxItems": count,
                    }
                },
                "required": ["turns"],
                "additionalProperties": False,
            },
        },
    }


# ============================================================
# Validation
# ============================================================

def _score_value(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TurnValidationError(f"{label}: score must be a number.")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise TurnValidationError(f"{label}: score {score} outside 0..1.")
    return score


def validate_turn_result(
    parsed: dict[str, Any],
    units: list[InterviewerTurn],
) -> list[dict[str, Any]]:
    """
    Enforce the contract per turn. Returns cleaned per-turn
    results in unit order, each carrying the verbatim
    interviewer text it was judged on (from the capture, not
    echoed by the model).
    """

    turns = parsed.get("turns")
    if not isinstance(turns, list):
        raise TurnValidationError("Missing 'turns' list.")
    if len(turns) != len(units):
        raise TurnValidationError(
            f"Judge returned {len(turns)} entries for "
            f"{len(units)} interviewer turns."
        )

    results = []
    for unit, item in zip(units, turns):
        label = f"T{unit.number}"
        if not isinstance(item, dict):
            raise TurnValidationError(f"{label}: entry is not an object.")
        if item.get("number") != unit.number:
            raise TurnValidationError(
                f"{label}: numbering mismatch ({item.get('number')})."
            )

        evaluation = item.get("evaluation") or {}
        dimensions = {}
        for name in SCORE_DIMENSIONS:
            entry = evaluation.get(name)
            if not isinstance(entry, dict):
                raise TurnValidationError(f"{label}: missing '{name}'.")
            dimensions[name] = {
                "score": _score_value(entry.get("score"), f"{label}.{name}"),
                "reasoning": str(entry.get("reasoning") or ""),
            }

        raw_issues = item.get("issues") or []
        if not isinstance(raw_issues, list):
            raise TurnValidationError(f"{label}: 'issues' must be a list.")
        issues = []
        for issue in raw_issues:
            issue = str(issue).strip().lower()
            if issue not in ISSUE_TYPES:
                raise TurnValidationError(
                    f"{label}: unknown issue type {issue!r}."
                )
            if issue not in issues:
                issues.append(issue)

        reasoning = str(item.get("reasoning") or "").strip()
        if not reasoning:
            raise TurnValidationError(f"{label}: missing reasoning.")

        results.append(
            {
                "number": unit.number,
                "interviewer_turn": unit.text,
                "source": unit.source,
                "context_understanding": str(
                    item.get("context_understanding") or ""
                ),
                "evaluation": dimensions,
                "score": _score_value(item.get("score"), f"{label}.score"),
                "reasoning": reasoning,
                "issues": issues,
                "issue_details": str(item.get("issue_details") or ""),
            }
        )

    return results


# ============================================================
# Judge + aggregation
# ============================================================

def nemotron_judge(prompt: str, response_format: dict[str, Any]) -> str:
    """Default judge: NVIDIA Nemotron via evaluation.models."""

    # Imported lazily: evaluation.models requires NVIDIA_API_KEY
    # at import time and the offline tests inject a stub judge.
    from evaluation.models import evaluate_with_nvidia

    return evaluate_with_nvidia(prompt, response_format=response_format)


def _parse_json(raw: str) -> dict[str, Any]:
    from evaluation.evaluator import extract_json

    return extract_json(raw)


def aggregate_scores(
    results: list[dict[str, Any]],
    *,
    flag_below: float = 0.5,
) -> dict[str, Any]:
    """Python-side aggregation of the model-assigned scores."""

    scores = [r["score"] for r in results]
    issue_counts: dict[str, int] = {}
    for result in results:
        for issue in result["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    flagged = [
        {
            "number": r["number"],
            "score": r["score"],
            "issues": r["issues"],
            "interviewer_turn": r["interviewer_turn"],
        }
        for r in results
        if r["issues"] or r["score"] < flag_below
    ]
    dimension_means = {}
    for name in SCORE_DIMENSIONS:
        values = [r["evaluation"][name]["score"] for r in results]
        dimension_means[name] = (
            round(sum(values) / len(values), 4) if values else None
        )
    return {
        "turns_total": len(results),
        "overall_score": (
            round(sum(scores) / len(scores), 4) if scores else None
        ),
        "min_score": min(scores) if scores else None,
        "dimension_means": dimension_means,
        "issue_counts": issue_counts,
        "turns_with_issues": sum(1 for r in results if r["issues"]),
        "flagged_turns": flagged,
    }


def evaluate_interviewer_turns(
    units: list[InterviewerTurn],
    judge: Judge | None = None,
) -> dict[str, Any]:
    """
    Send the turn units to the judge once and return validated
    per-turn results plus Python aggregation. The prompt and the
    exact transcript data are included verbatim.
    """

    if not units:
        raise InterviewerEvidenceError(
            "No interviewer turns to evaluate."
        )

    prompt = build_turn_prompt(units)
    schema = build_turn_response_schema(len(units))
    raw = (judge or nemotron_judge)(prompt, schema)
    parsed = _parse_json(raw)
    results = validate_turn_result(parsed, units)

    return {
        "prompt": prompt,
        "transcript_data": format_turn_units(units),
        "units": [u.as_dict() for u in units],
        "results": results,
        "aggregate": aggregate_scores(results),
    }


def evaluate_captured_turns(
    turns: list[dict[str, Any]],
    judge: Judge | None = None,
) -> dict[str, Any]:
    """
    End-to-end: provenance filter -> turn units -> judge ->
    validation -> aggregation. Raises InterviewerEvidenceError
    when no interviewer turn exists. Candidate text from harness
    fixtures is rejected before anything reaches the judge.
    """

    kept, provenance = filter_turns(turns)
    if not provenance.has_interviewer_turns:
        raise InterviewerEvidenceError(
            "No interviewer transcript turns reached the evaluator "
            f"(candidate accepted={provenance.candidate_turns_accepted}, "
            f"rejected={provenance.rejected_sources or {}})."
        )

    units = build_interviewer_turns(kept)
    report = evaluate_interviewer_turns(units, judge=judge)
    report["provenance"] = provenance.as_dict()
    return report


# ============================================================
# Capture selection
# ============================================================

def select_turn_capture() -> tuple[TranscriptCapture, list[dict[str, Any]]]:
    """
    Pick the capture backend that can support per-turn
    evaluation: the first (EMH_TRANSCRIPT_CAPTURE or AUTO_ORDER)
    whose turns contain >= 1 interviewer turn. Among backends
    with interviewer turns, one that also carries evidence-grade
    candidate context is preferred (it is what makes
    context-awareness judgeable). Returns (backend, raw turns).
    """

    choice = os.getenv("EMH_TRANSCRIPT_CAPTURE", "auto").lower()
    names = [choice] if choice != "auto" else list(AUTO_ORDER)
    attempts = []
    fallback: tuple[TranscriptCapture, list[dict[str, Any]]] | None = None

    for name in names:
        if name not in _BACKENDS:
            raise InterviewerEvidenceError(
                f"Unknown EMH_TRANSCRIPT_CAPTURE={choice!r}."
            )
        backend = _BACKENDS[name]()
        if not backend.available():
            attempts.append(f"{name}: unavailable")
            continue
        turns = backend.get_turns()
        _, provenance = filter_turns(turns)
        if provenance.interviewer_turns and provenance.candidate_turns_accepted:
            return backend, turns
        if provenance.interviewer_turns and fallback is None:
            fallback = (backend, turns)
        attempts.append(
            f"{name}: interviewer={provenance.interviewer_turns}, "
            f"candidate context accepted="
            f"{provenance.candidate_turns_accepted}, "
            f"rejected={provenance.rejected_sources or {}}"
        )

    if fallback is not None:
        return fallback

    raise InterviewerEvidenceError(
        "No capture backend provides interviewer turns: "
        + "; ".join(attempts)
    )


def dump_report(report: dict[str, Any], path: str) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
