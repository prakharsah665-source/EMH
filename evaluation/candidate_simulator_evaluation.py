"""
Third scoring target: the CANDIDATE SIMULATOR, judged in its
own Nemotron call with its own rubric. Never shares a prompt
with interviewer judging (evaluation.interviewer_turn_evaluation)
- see assert_prompt_is_candidate_only().

Blinding: the judge never sees the persona label or band. It
scores OBSERVED traits per turn on absolute scales; Python maps
observed competence onto the hidden persona band
(simulator.personas) to compute per-turn adherence and drift.

Competency mode, per turn:
    competence (overall observed level), depth, specificity,
    confidence, evasiveness, role_fit, seniority_consistency,
    meta_leakage / character_break flags with evidence.
Robustness mode, per turn:
    spec_executed (did the turn perform the REQUIRED misbehaviour),
    plus meta_leakage / character_break. Refusing to misbehave
    (spec_executed < EXECUTED_THRESHOLD) is a simulator FAILURE.

Gates:
    check_monotonic_separation()  strong > average > weak by mean,
                                  non-overlapping observed bands,
                                  minimum gap (collapse) - CI fail.
    stimulus_validity()           simulator verdict that GATES the
                                  validity of interviewer scores;
                                  simulator scores are never
                                  aggregated with interviewer scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from simulator.personas import PERSONA_ORDER, PERSONAS, Persona, persona_by_id
from simulator.role_context import RoleContext

Judge = Callable[[str, dict[str, Any]], str]

OBSERVED_TRAITS = (
    "competence",
    "depth",
    "specificity",
    "confidence",
    "evasiveness",
    "role_fit",
    "seniority_consistency",
)

EXECUTED_THRESHOLD = 0.5      # robustness: spec executed if >= this
MIN_BAND_GAP = 0.10           # separation: mean gap below this = collapse
LEAKAGE_MARKERS = (
    "as an ai", "language model", "i am an ai", "simulat", "persona",
    "this is a test", "my script", "the harness", "role-play", "roleplay",
    "i was instructed", "my instructions",
)


class SimulatorValidationError(ValueError):
    """The judge output violated the evaluation contract."""


class SeparationError(AssertionError):
    """Monotonic separation gate failed (inversion or collapse)."""


# ============================================================
# Prompt (blind)
# ============================================================

def _role_block(role: RoleContext | None) -> str:
    if role is None:
        return "Role context: UNKNOWN (no job details captured)."
    return role.prompt_block()


def format_simulator_turns(turns: list[dict[str, Any]], mode: str) -> str:
    blocks = []
    for t in turns:
        block = (
            f"--- TURN {t['turn']} ---\n"
            f"INTERVIEWER QUESTION:\n{t['question']}\n\n"
            f"CANDIDATE REPLY:\n{t['intended_text']}"
        )
        if mode == "robustness":
            block += (
                f"\n\nREQUIRED BEHAVIOUR FOR THIS TURN "
                f"({t.get('spec_key')}):\n{t.get('required_behaviour')}"
            )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_simulator_judge_prompt(
    turns: list[dict[str, Any]],
    role: RoleContext | None,
    mode: str,
) -> str:
    """
    Exact prompt sent to the judge. Contains NO persona label,
    band or expectation of competence level.
    """

    header = (
        "You are an impartial evaluator. You will assess ONLY the "
        "CANDIDATE's replies in the interview excerpts below. Do NOT "
        "evaluate, score or comment on the interviewer; the interviewer "
        "questions are context only. Assess each turn INDEPENDENTLY on "
        "absolute scales - you have no information about what level of "
        "candidate to expect, and you must not assume one.\n\n"
        f"JOB CONTEXT\n{_role_block(role)}\n\n"
        "GENERAL RULES\n"
        "- Base every judgment only on the candidate reply text.\n"
        "- 'meta_leakage' = the reply refers to being an AI, a "
        "simulation, a persona, a script, a test, instructions, or "
        "otherwise speaks from outside the candidate character; "
        "'character_break' = the reply stops being a job candidate "
        "speaking to an interviewer (addresses someone else, stage "
        "directions, meta commentary). Quote the evidence verbatim.\n"
        "- All scores are decimals 0.0-1.0.\n\n"
    )
    if mode == "competency":
        body = (
            "PER-TURN SCORES (competency)\n"
            "- competence: overall level the reply DEMONSTRATES for this "
            "role (0 = no usable competence, 0.5 = adequate mid-level, "
            "1 = excellent senior-level).\n"
            "- depth: technical depth and reasoning about trade-offs.\n"
            "- specificity: concrete systems, numbers, named tools, a real "
            "example (1) vs generic talk (0).\n"
            "- confidence: assured and precise (1) vs hedging/uncertain (0). "
            "Score what is shown; confidence is not competence.\n"
            "- evasiveness: 1 = fully avoids the question, 0 = answers it "
            "directly.\n"
            "- role_fit: how well the content targets THIS role and its "
            "required skills (1) vs a different discipline (0).\n"
            "- seniority_consistency: claims about experience are "
            "consistent with the candidate's declared years and with "
            "earlier turns (1) vs contradictory/implausible (0).\n\n"
            "OUTPUT: JSON only:\n"
            '{"turns":[{"number":1,"competence":0.0,"depth":0.0,'
            '"specificity":0.0,"confidence":0.0,"evasiveness":0.0,'
            '"role_fit":0.0,"seniority_consistency":0.0,'
            '"meta_leakage":false,"character_break":false,'
            '"leakage_evidence":"","reasoning":"..."}]}\n'
        )
    else:
        body = (
            "PER-TURN SCORES (adversarial robustness stimulus)\n"
            "Each turn carries a REQUIRED BEHAVIOUR the candidate was "
            "supposed to perform. Score:\n"
            "- spec_executed: 1 = the reply clearly performs the required "
            "behaviour; 0 = it does not (e.g. it answers normally and "
            "helpfully instead). Quote the evidence verbatim.\n"
            "- meta_leakage / character_break as defined above (a reply "
            "that says it is 'testing' the interviewer is leakage).\n\n"
            "OUTPUT: JSON only:\n"
            '{"turns":[{"number":1,"spec_executed":0.0,'
            '"execution_evidence":"","meta_leakage":false,'
            '"character_break":false,"leakage_evidence":"",'
            '"reasoning":"..."}]}\n'
        )
    return (
        header + body
        + f"\nThere are exactly {len(turns)} turns; return exactly "
        f"{len(turns)} entries numbered 1..{len(turns)}.\n\n"
        "============================================================\n"
        "INTERVIEW EXCERPTS\n"
        "============================================================\n\n"
        + format_simulator_turns(turns, mode)
    )


def build_simulator_response_schema(count: int, mode: str) -> dict[str, Any]:
    score = {"type": "number", "minimum": 0, "maximum": 1}
    common = {
        "number": {"type": "integer"},
        "meta_leakage": {"type": "boolean"},
        "character_break": {"type": "boolean"},
        "leakage_evidence": {"type": "string"},
        "reasoning": {"type": "string"},
    }
    if mode == "competency":
        props = {**{t: score for t in OBSERVED_TRAITS}, **common}
    else:
        props = {"spec_executed": score, "execution_evidence": {"type": "string"}, **common}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "CandidateSimulatorEvaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": {
                            "type": "object",
                            "properties": props,
                            "required": list(props),
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["turns"],
                "additionalProperties": False,
            },
        },
    }


def assert_prompt_is_candidate_only(prompt: str) -> None:
    """Guard: this judge call must never score the interviewer."""

    forbidden = ("INTERVIEWER TURN", "TO EVALUATE", "flow_alignment", "question_quality")
    for marker in forbidden:
        if marker in prompt:
            raise SimulatorValidationError(
                f"simulator judge prompt contains interviewer-scoring "
                f"marker {marker!r}"
            )
    for label in PERSONAS:
        if f"persona: {label}" in prompt.lower() or f"({label} candidate)" in prompt.lower():
            raise SimulatorValidationError(
                f"simulator judge prompt leaks persona label {label!r}"
            )


# ============================================================
# Validation + band mapping
# ============================================================

def _num(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulatorValidationError(f"{label}: not a number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise SimulatorValidationError(f"{label}: {value} outside 0..1")
    return value


def _local_leakage(text: str) -> list[str]:
    low = text.lower()
    return [m for m in LEAKAGE_MARKERS if m in low]


def validate_simulator_result(
    parsed: dict[str, Any],
    turns: list[dict[str, Any]],
    mode: str,
    persona: Persona | None,
) -> list[dict[str, Any]]:
    items = parsed.get("turns")
    if not isinstance(items, list) or len(items) != len(turns):
        raise SimulatorValidationError(
            f"judge returned {len(items) if isinstance(items, list) else 'no'} "
            f"entries for {len(turns)} turns"
        )
    results = []
    for t, item in zip(turns, items):
        label = f"T{t['turn']}"
        if not isinstance(item, dict) or item.get("number") != t["turn"]:
            raise SimulatorValidationError(f"{label}: numbering mismatch")
        leakage_local = _local_leakage(t["intended_text"])
        base = {
            "turn": t["turn"],
            "question": t["question"],
            "candidate_reply": t["intended_text"],
            "meta_leakage": bool(item.get("meta_leakage")) or bool(leakage_local),
            "character_break": bool(item.get("character_break")),
            "leakage_evidence": str(item.get("leakage_evidence") or "")
            or (", ".join(leakage_local) if leakage_local else ""),
            "reasoning": str(item.get("reasoning") or ""),
        }
        if mode == "competency":
            observed = {k: _num(item.get(k), f"{label}.{k}") for k in OBSERVED_TRAITS}
            in_band = None
            if persona is not None:
                lo, hi = persona.band
                in_band = lo <= observed["competence"] < hi
            results.append({**base, "observed": observed, "in_band": in_band})
        else:
            executed = _num(item.get("spec_executed"), f"{label}.spec_executed")
            results.append({
                **base,
                "spec_key": t.get("spec_key"),
                "required_behaviour": t.get("required_behaviour"),
                "spec_executed": executed,
                "execution_evidence": str(item.get("execution_evidence") or ""),
                "executed": executed >= EXECUTED_THRESHOLD,
            })
    return results


# ============================================================
# Judge + per-run summary
# ============================================================

def nemotron_judge(prompt: str, response_format: dict[str, Any]) -> str:
    from evaluation.models import evaluate_with_nvidia

    return evaluate_with_nvidia(prompt, response_format=response_format)


def _parse_json(raw: str) -> dict[str, Any]:
    from evaluation.evaluator import extract_json

    return extract_json(raw)


def evaluate_simulator_run(
    turns: list[dict[str, Any]],
    *,
    mode: str,
    role: RoleContext | None,
    persona_id: str | None = None,
    judge: Judge | None = None,
) -> dict[str, Any]:
    """
    One blind judge call over one simulator run. `persona_id`
    (opaque) is used ONLY after the call, for band mapping.
    """

    if not turns:
        raise SimulatorValidationError("no simulator turns to evaluate")
    persona = persona_by_id(persona_id) if persona_id else None

    prompt = build_simulator_judge_prompt(turns, role, mode)
    assert_prompt_is_candidate_only(prompt)
    raw = (judge or nemotron_judge)(prompt, build_simulator_response_schema(len(turns), mode))
    results = validate_simulator_result(_parse_json(raw), turns, mode, persona)

    summary: dict[str, Any] = {
        "mode": mode,
        "persona_id": persona_id,
        "persona_band": list(persona.band) if persona else None,
        "turns": len(results),
        "leakage_turns": [r["turn"] for r in results if r["meta_leakage"]],
        "character_break_turns": [r["turn"] for r in results if r["character_break"]],
    }
    if mode == "competency":
        comp = [r["observed"]["competence"] for r in results]
        summary.update({
            "competence_per_turn": comp,
            "competence_mean": round(sum(comp) / len(comp), 4),
            "competence_min": min(comp),
            "competence_max": max(comp),
            "role_fit_mean": round(sum(r["observed"]["role_fit"] for r in results) / len(results), 4),
            "seniority_consistency_mean": round(
                sum(r["observed"]["seniority_consistency"] for r in results) / len(results), 4
            ),
            "persona_adherence": (
                round(sum(1 for r in results if r["in_band"]) / len(results), 4)
                if persona else None
            ),
            "drift_turns": [r["turn"] for r in results if r["in_band"] is False],
        })
    else:
        summary.update({
            "spec_executed_per_turn": [r["spec_executed"] for r in results],
            "spec_adherence": round(sum(1 for r in results if r["executed"]) / len(results), 4),
            "refused_turns": [r["turn"] for r in results if not r["executed"]],
        })
    return {"prompt": prompt, "results": results, "summary": summary}


# ============================================================
# Gates
# ============================================================

def check_monotonic_separation(
    runs: dict[str, dict[str, Any]],
    *,
    min_gap: float = MIN_BAND_GAP,
) -> dict[str, Any]:
    """
    `runs` maps persona NAME -> evaluate_simulator_run() report
    (competency mode). Passes only when, in ascending order
    weak < average < strong: means strictly increase, observed
    ranges do not overlap, and consecutive means differ by at
    least min_gap. Returns a verdict dict; raises nothing.
    """

    missing = [p for p in PERSONA_ORDER if p not in runs]
    if missing:
        return {"pass": False, "reasons": [f"missing persona runs: {missing}"]}

    stats = {
        name: {
            "mean": runs[name]["summary"]["competence_mean"],
            "min": runs[name]["summary"]["competence_min"],
            "max": runs[name]["summary"]["competence_max"],
        }
        for name in PERSONA_ORDER
    }
    reasons = []
    for lower, upper in zip(PERSONA_ORDER, PERSONA_ORDER[1:]):
        lo, hi = stats[lower], stats[upper]
        if hi["mean"] <= lo["mean"]:
            reasons.append(
                f"INVERSION: {upper} mean {hi['mean']:.2f} <= {lower} mean {lo['mean']:.2f}"
            )
        elif hi["mean"] - lo["mean"] < min_gap:
            reasons.append(
                f"COLLAPSE: {upper}-{lower} mean gap {hi['mean']-lo['mean']:.2f} < {min_gap}"
            )
        if hi["min"] <= lo["max"]:
            reasons.append(
                f"OVERLAP: {upper} min {hi['min']:.2f} <= {lower} max {lo['max']:.2f}"
            )
    return {"pass": not reasons, "reasons": reasons, "stats": stats}


def stimulus_validity(
    competency_runs: dict[str, dict[str, Any]] | None = None,
    robustness_run: dict[str, Any] | None = None,
    *,
    separation: dict[str, Any] | None = None,
    min_persona_adherence: float = 0.8,
) -> dict[str, Any]:
    """
    The simulator's verdict on whether the stimulus was valid -
    gates the validity of INTERVIEWER scores produced against it.
    Simulator numbers are never merged into interviewer numbers.
    """

    reasons: list[str] = []
    if competency_runs:
        for name, run in competency_runs.items():
            s = run["summary"]
            if s["leakage_turns"] or s["character_break_turns"]:
                reasons.append(
                    f"{name}: meta-leakage/character break on turns "
                    f"{s['leakage_turns'] + s['character_break_turns']}"
                )
            if s["persona_adherence"] is not None and s["persona_adherence"] < min_persona_adherence:
                reasons.append(
                    f"{name}: persona adherence {s['persona_adherence']:.2f} "
                    f"< {min_persona_adherence} (drift on turns {s['drift_turns']})"
                )
        if separation is not None and not separation["pass"]:
            reasons.extend(f"separation: {r}" for r in separation["reasons"])
    if robustness_run:
        s = robustness_run["summary"]
        if s["refused_turns"]:
            reasons.append(
                f"robustness: simulator refused to misbehave on turns "
                f"{s['refused_turns']} (spec adherence {s['spec_adherence']:.2f})"
            )
        if s["leakage_turns"] or s["character_break_turns"]:
            reasons.append(
                f"robustness: meta-leakage/character break on turns "
                f"{s['leakage_turns'] + s['character_break_turns']}"
            )
    return {"valid": not reasons, "reasons": reasons}


def dump_report(report: dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
