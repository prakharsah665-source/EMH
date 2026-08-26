"""
Offline contract tests for the candidate simulator and its judge.
No network. Prove: model-family rule, blinding, separate judge
call, band mapping / drift, robustness inversion (refusing to
misbehave = failure), monotonic-separation gate, validity gate.
"""

import json

import pytest

from evaluation.candidate_simulator_evaluation import (
    OBSERVED_TRAITS,
    SimulatorValidationError,
    assert_prompt_is_candidate_only,
    build_simulator_judge_prompt,
    check_monotonic_separation,
    evaluate_simulator_run,
    stimulus_validity,
)
from evaluation.interviewer_turn_evaluation import build_turn_prompt, InterviewerTurn
from simulator.candidate_simulator import (
    CandidateSimulator,
    SimulatorModelError,
    assert_model_family_allowed,
)
from simulator.personas import ADVERSARIAL_SPECS, PERSONAS, PERSONA_ORDER
from simulator.role_context import RoleContext

ROLE = RoleContext(role="DevOps Engineer", skills=["Docker", "Kubernetes"],
                   experience_required="3", candidate_years=2.0)
QUESTIONS = ["Please introduce yourself.", "Tell me about a hard incident.",
             "How do you design a CI pipeline?"]


def run_simulator(mode, persona=None, text_for=lambda t, q: f"reply {t}"):
    sim = CandidateSimulator(mode=mode, role=ROLE, persona=persona,
                             model="meta/llama-3.3-70b-instruct",
                             generate=lambda prompt: text_for(sim_turn[0], prompt))
    sim_turn = [0]
    for i, q in enumerate(QUESTIONS, 1):
        sim_turn[0] = i
        sim.answer(i, q)
    return [t.as_dict() for t in sim.turns]


def judge_factory(competence=None, executed=None, leak_turns=()):
    seen = {}

    def judge(prompt, schema):
        seen["prompt"] = prompt
        n = schema["json_schema"]["schema"]["properties"]["turns"]["maxItems"]
        rows = []
        for i in range(1, n + 1):
            row = {"number": i, "meta_leakage": i in leak_turns,
                   "character_break": False, "leakage_evidence": "", "reasoning": "r"}
            if competence is not None:
                c = competence(i)
                row.update({t: c for t in OBSERVED_TRAITS})
            else:
                row.update({"spec_executed": executed(i), "execution_evidence": "e"})
            rows.append(row)
        return json.dumps({"turns": rows})

    return judge, seen


# ---------------- model family rule ----------------

def test_simulator_model_family_must_differ_from_interviewer_and_judge():
    for bad in ("gpt-4o", "openai/gpt-5.6-luna", "nvidia/nemotron-3-nano-30b-a3b"):
        with pytest.raises(SimulatorModelError):
            assert_model_family_allowed(bad)
    assert_model_family_allowed("meta/llama-3.3-70b-instruct")
    assert_model_family_allowed("google/diffusiongemma-26b-a4b-it")
    assert_model_family_allowed("mistralai/mistral-large")


# ---------------- blinding + separation of judge calls ----------------

def test_judge_prompt_is_blind_to_persona_and_candidate_only():
    turns = run_simulator("competency", PERSONAS["weak"])
    prompt = build_simulator_judge_prompt(turns, ROLE, "competency")
    low = prompt.lower()
    for label in PERSONAS:                       # no persona label anywhere
        assert f"{label} candidate" not in low and f"persona: {label}" not in low
    assert PERSONAS["weak"].id not in prompt     # not even the opaque id
    import re
    # No competence label, band or expectation leaks into the prompt.
    assert not re.search(r"\b(weak|average|strong)\b", low)
    assert not re.search(r"\b(band|expected level|should be)\b", low)
    assert "Do NOT evaluate, score or comment on the interviewer" in prompt
    assert_prompt_is_candidate_only(prompt)

    # The interviewer judge prompt and the simulator judge prompt
    # are disjoint in what they ask to score.
    unit = InterviewerTurn(number=1, text="Q?", source="livekit-agent-session",
                           confidence="medium", context=[])
    interviewer_prompt = build_turn_prompt([unit])
    with pytest.raises(SimulatorValidationError):
        assert_prompt_is_candidate_only(interviewer_prompt)
    assert "competence" not in interviewer_prompt.lower().split("dimensions")[0]


def test_simulator_never_uses_candidate_answers_script():
    bot = pytest.importorskip("tests.fixtures.legacy_candidate_answers")
    turns = run_simulator("competency", PERSONAS["strong"])
    texts = " ".join(t["intended_text"] for t in turns)
    for scripted in bot.CANDIDATE_ANSWERS:
        assert scripted[:40] not in texts


# ---------------- band mapping / drift ----------------

def test_band_mapping_and_drift_detection():
    turns = run_simulator("competency", PERSONAS["weak"])
    # weak persona, but turn 3 drifts to competent (0.8)
    judge, _ = judge_factory(competence=lambda i: 0.2 if i < 3 else 0.8)
    report = evaluate_simulator_run(turns, mode="competency", role=ROLE,
                                    persona_id=PERSONAS["weak"].id, judge=judge)
    s = report["summary"]
    assert s["competence_per_turn"] == [0.2, 0.2, 0.8]
    assert s["drift_turns"] == [3]
    assert s["persona_adherence"] == round(2 / 3, 4)
    assert [r["in_band"] for r in report["results"]] == [True, True, False]


def test_local_leakage_markers_are_flagged_even_if_judge_misses_them():
    turns = run_simulator("competency", PERSONAS["average"],
                          text_for=lambda t, p: "As an AI language model I would say" if t == 2 else "fine answer")
    judge, _ = judge_factory(competence=lambda i: 0.5)
    report = evaluate_simulator_run(turns, mode="competency", role=ROLE,
                                    persona_id=PERSONAS["average"].id, judge=judge)
    assert report["summary"]["leakage_turns"] == [2]


# ---------------- robustness inversion ----------------

def test_robustness_refusing_to_misbehave_is_a_failure():
    turns = run_simulator("robustness")
    assert [t["spec_key"] for t in turns] == [s.key for s in ADVERSARIAL_SPECS[:3]]
    judge, seen = judge_factory(executed=lambda i: 0.9 if i != 2 else 0.1)
    report = evaluate_simulator_run(turns, mode="robustness", role=ROLE, judge=judge)
    assert "REQUIRED BEHAVIOUR FOR THIS TURN" in seen["prompt"]
    assert report["summary"]["refused_turns"] == [2]
    assert report["summary"]["spec_adherence"] == round(2 / 3, 4)
    validity = stimulus_validity(robustness_run=report)
    assert validity["valid"] is False
    assert "refused to misbehave on turns [2]" in validity["reasons"][0]


# ---------------- separation gate ----------------

def _runs(weak, average, strong):
    out = {}
    for name, series in (("weak", weak), ("average", average), ("strong", strong)):
        turns = run_simulator("competency", PERSONAS[name])
        judge, _ = judge_factory(competence=lambda i, s=series: s[i - 1])
        out[name] = evaluate_simulator_run(turns, mode="competency", role=ROLE,
                                           persona_id=PERSONAS[name].id, judge=judge)
    return out


def test_monotonic_separation_pass_inversion_collapse_overlap():
    ok = check_monotonic_separation(_runs([0.1, 0.2, 0.2], [0.5, 0.55, 0.6], [0.85, 0.9, 0.95]))
    assert ok["pass"], ok

    inverted = check_monotonic_separation(_runs([0.8, 0.8, 0.8], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]))
    assert not inverted["pass"] and any("INVERSION" in r for r in inverted["reasons"])

    collapsed = check_monotonic_separation(_runs([0.50, 0.50, 0.50], [0.55, 0.55, 0.55], [0.60, 0.60, 0.60]))
    assert not collapsed["pass"] and any("COLLAPSE" in r for r in collapsed["reasons"])

    overlap = check_monotonic_separation(_runs([0.1, 0.2, 0.65], [0.5, 0.55, 0.6], [0.85, 0.9, 0.95]))
    assert not overlap["pass"] and any("OVERLAP" in r for r in overlap["reasons"])

    validity = stimulus_validity(competency_runs=_runs([0.1, 0.2, 0.2], [0.5, 0.55, 0.6], [0.85, 0.9, 0.95]),
                                 separation=ok)
    assert validity["valid"], validity


def test_persona_order_and_bands_are_non_overlapping():
    bands = [PERSONAS[n].band for n in PERSONA_ORDER]
    for (lo1, hi1), (lo2, hi2) in zip(bands, bands[1:]):
        assert hi1 <= lo2
