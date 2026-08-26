"""
LIVE simulator-validity gate (third scoring target).

Runs the candidate simulator (non-GPT, non-Nemotron family) over
the interviewer question bank captured from the real interview -
once per hidden persona (weak / average / strong) and once in
robustness mode - then judges each run BLIND in its own Nemotron
call (evaluation.candidate_simulator_evaluation).

CI gates:
  * monotonic separation  strong > average > weak by mean, no
                          overlap, no collapse  -> FAIL on inversion
  * persona adherence     per-turn, drift to competent is flagged
  * meta-leakage / character breaks
  * robustness            refusing to misbehave = simulator FAILURE

Output: artifacts/reports/candidate_simulator_evaluation.json
(prompts, per-turn observed traits, band mapping, separation
matrix) and artifacts/reports/stimulus_validity.json - the
verdict that gates the validity of interviewer scores. Simulator
scores are NEVER aggregated with interviewer scores.
"""

import json
import os

import pytest

from evaluation.candidate_simulator_evaluation import (
    check_monotonic_separation,
    dump_report,
    evaluate_simulator_run,
    stimulus_validity,
)
from evaluation.interviewer_turn_evaluation import (
    InterviewerEvidenceError,
    build_interviewer_turns,
    filter_turns,
    select_turn_capture,
)
from simulator.candidate_simulator import CandidateSimulator, simulator_model_name
from simulator.personas import PERSONAS, PERSONA_ORDER
from simulator.role_context import load_role_context

REPORT_PATH = "artifacts/reports/candidate_simulator_evaluation.json"
VALIDITY_PATH = "artifacts/reports/stimulus_validity.json"
MAX_QUESTIONS = int(os.getenv("EMH_SIMULATOR_MAX_QUESTIONS", "6"))


def question_bank() -> list[str]:
    """Interviewer questions from the real capture (bot text only)."""

    try:
        _, raw = select_turn_capture()
    except InterviewerEvidenceError as error:
        pytest.skip(f"simulator gate: {error}")
    units = build_interviewer_turns(filter_turns(raw)[0])
    questions = [u.text for u in units if len(u.text.split()) >= 5]
    if len(questions) < 3:
        pytest.skip("simulator gate: fewer than 3 captured interviewer questions")
    return questions[:MAX_QUESTIONS]


def drive(sim: CandidateSimulator, questions: list[str]) -> list[dict]:
    for i, q in enumerate(questions, 1):
        sim.answer(i, q)
    return [t.as_dict() for t in sim.turns]


@pytest.mark.ai_evaluation
def test_simulator_personas_separate_and_hold_character():
    questions = question_bank()
    role = load_role_context()
    model = simulator_model_name()
    print(f"\n[simulator] model={model} questions={len(questions)} "
          f"role={role.role if role else 'UNKNOWN'}")

    runs = {}
    for name in PERSONA_ORDER:
        sim = CandidateSimulator(mode="competency", role=role, persona=PERSONAS[name])
        turns = drive(sim, questions)
        runs[name] = evaluate_simulator_run(
            turns, mode="competency", role=role, persona_id=PERSONAS[name].id
        )
        s = runs[name]["summary"]
        print(f"[simulator] persona={name} competence={s['competence_per_turn']} "
              f"adherence={s['persona_adherence']} drift={s['drift_turns']} "
              f"leakage={s['leakage_turns']} role_fit={s['role_fit_mean']}")

    separation = check_monotonic_separation(runs)
    validity = stimulus_validity(competency_runs=runs, separation=separation)

    report = {
        "model": model,
        "role": role.as_dict() if role else None,
        "questions": questions,
        "judge_prompt_example": runs["average"]["prompt"],
        "runs": {n: {"results": r["results"], "summary": r["summary"]} for n, r in runs.items()},
        "separation": separation,
        "validity": validity,
    }
    dump_report(report, REPORT_PATH)
    dump_report({"competency": validity, "source": REPORT_PATH}, VALIDITY_PATH)
    print("[simulator] separation:", json.dumps(separation, indent=1))
    print("[simulator] validity:", json.dumps(validity, indent=1))

    assert separation["pass"], "SEPARATION GATE FAILED: " + "; ".join(separation["reasons"])
    assert validity["valid"], "SIMULATOR VALIDITY FAILED: " + "; ".join(validity["reasons"])


@pytest.mark.ai_evaluation
def test_simulator_executes_adversarial_spec_in_robustness_mode():
    questions = question_bank()
    role = load_role_context()
    sim = CandidateSimulator(mode="robustness", role=role)
    turns = drive(sim, questions)
    run = evaluate_simulator_run(turns, mode="robustness", role=role)
    s = run["summary"]
    for r in run["results"]:
        print(f"\n[robustness] turn {r['turn']} spec={r['spec_key']} "
              f"executed={r['spec_executed']:.2f}\n  reply: {r['candidate_reply'][:160]}")
    validity = stimulus_validity(robustness_run=run)

    existing = {}
    try:
        existing = json.loads(open(VALIDITY_PATH).read())
    except (OSError, ValueError):
        pass
    existing["robustness"] = validity
    dump_report(existing, VALIDITY_PATH)
    dump_report({"model": sim.model, "results": run["results"], "summary": s,
                 "prompt": run["prompt"]},
                "artifacts/reports/candidate_simulator_robustness.json")

    assert validity["valid"], (
        "ROBUSTNESS STIMULUS INVALID (simulator refused to misbehave / broke "
        "character): " + "; ".join(validity["reasons"])
    )
