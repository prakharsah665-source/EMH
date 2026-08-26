# Candidate simulator — the third scoring target

Package `simulator/` + `evaluation/candidate_simulator_evaluation.py`.
Tests: `tests/simulator/test_candidate_simulator_offline.py` (contract,
no network) and `tests/simulator/test_simulator_validity_gate.py` (live:
simulator + blind judge). Report section: "Candidate Simulator
(stimulus validity)" in `scripts/run_all_tests.py`.

## Pieces

| piece | file | notes |
|---|---|---|
| personas (opaque ids, non-overlapping bands) | `simulator/personas.py` | weak [0,.4) · average [.4,.7) · strong [.7,1]; adversarial specs for robustness |
| role / seniority context | `simulator/role_context.py` | parsed from the socket.io `job-candidate-details` frame the E2E drive records (role, skills, JD experience, candidate years) |
| simulator | `simulator/candidate_simulator.py` | model family must differ from interviewer (GPT) and judge (Nemotron) — enforced by `assert_model_family_allowed`; default `google/diffusiongemma-26b-a4b-it` via the NVIDIA endpoint, override `EMH_SIMULATOR_MODEL`; records `intended_text` per turn |
| judge (own call, blind) | `evaluation/candidate_simulator_evaluation.py` | prompt carries NO persona label/band; scores observed traits per turn; Python maps to the hidden band → adherence + drift; robustness: `spec_executed`, refusing = FAILURE |
| gates | same | `check_monotonic_separation` (inversion / collapse / overlap → CI fail), `stimulus_validity` (gates interviewer-score validity; never aggregated) |

## Outputs
* `artifacts/reports/candidate_simulator_evaluation.json` — prompts, per-turn observed traits, band mapping, separation matrix
* `artifacts/reports/candidate_simulator_robustness.json`
* `artifacts/reports/stimulus_validity.json` — `{competency: {valid, reasons}, robustness: {valid, reasons}}`; interviewer reports should be read as INVALID when this says so.

## Calibration log (2026-08-20, Gemma, Frontend-developer question bank)
* round 1: average 0.6–0.8 overlapped strong → COLLAPSE/OVERLAP (gate caught it)
* round 2: strong 0.9 flat; average pushed into weak → OVERLAP
* round 3: means 0.13/0.67/0.86 but average drifted 0.55→0.80 across turns (per-turn drift caught) → OVERLAP at 0.80
* round 4 (anti-drift rule): weak 0.0–0.2 · average 0.5 flat · strong 0.7–0.85, adherence 1.0, no leakage → PASS; robustness 6/6 specs executed.

## Live wiring (2026-08-20)
`tests/e2e/test_bot_responsiveness.py` no longer has a scripted answer
bank. Per turn, `simulator/live_answers.py::LiveSimulatorAnswerSource`
reads the interviewer's ACTUAL completed question from the in-page
caption stream (agent-attributed only; stability-polled; new-since-last-
answer; duplicate-guarded), the one `CandidateSimulator` answers it
(`EMH_TEST_MODE` competency|robustness, `EMH_SIMULATOR_PERSONA`
weak|average|strong), the text is synthesized with say/afconvert into
`artifacts/audio/candidate/` and injected. Every generated turn
(`intended_text`, the question it answered) is written to
`artifacts/transcripts/simulator_turns.json`. If no valid live question
can be read the drive FAILS with the reason - there is no fallback.
The legacy script survives only as `tests/fixtures/legacy_candidate_answers.py`
for the offline proofs.
