"""
Offline contract tests for evaluation.interviewer_turn_evaluation.

They prove, without any LLM call:
  * harness-authored candidate text (injected-audio, i.e.
    tests/e2e CANDIDATE_ANSWERS) never reaches the judge prompt,
  * the prompt/data are byte-identical whatever CANDIDATE_ANSWERS
    contains (changing or removing the fixture list cannot
    determine the evaluation output),
  * uncaptured candidate responses are rendered as NOT_CAPTURED,
    never filled in,
  * scores/issues flow from the judge output through validation
    into Python aggregation - there is no predefined score,
  * malformed judge output is rejected.
"""

import json

import pytest

from evaluation.interviewer_turn_evaluation import (
    ISSUE_TYPES,
    NOT_CAPTURED,
    SCORE_DIMENSIONS,
    InterviewerEvidenceError,
    TurnValidationError,
    build_interviewer_turns,
    build_turn_prompt,
    evaluate_captured_turns,
    filter_turns,
    format_turn_units,
    merge_consecutive,
)


def turn(role, text, source, confidence):
    return {
        "role": role, "text": text, "ts": None,
        "source": source, "confidence": confidence,
    }


CAPTURED = [
    turn("assistant", "Hi, I'm Jamie. Please introduce yourself.",
         "livekit-agent-session", "medium"),
    turn("user", "I'm Sam, six years of backend work in Go.",
         "livekit-candidate-stt", "high"),
    turn("assistant", "Great. What is the hardest Go concurrency bug "
         "you have debugged?", "livekit-agent-session", "medium"),
    turn("user", "A goroutine leak caused by an unbuffered channel.",
         "livekit-candidate-stt", "high"),
    turn("assistant", "Thanks. What is the hardest Go concurrency bug "
         "you have debugged?", "livekit-agent-session", "medium"),
]


def stub_judge_factory(score_for_turn, issues_for_turn=None):
    """A deterministic stand-in for Nemotron that records the prompt."""

    seen = {}

    def judge(prompt, response_format):
        seen["prompt"] = prompt
        seen["schema"] = response_format
        count = response_format["json_schema"]["schema"]["properties"][
            "turns"]["maxItems"]
        turns = []
        for n in range(1, count + 1):
            s = score_for_turn(n)
            turns.append({
                "number": n,
                "context_understanding": f"ctx {n}",
                "evaluation": {
                    d: {"score": s, "reasoning": f"{d} {n}"}
                    for d in SCORE_DIMENSIONS
                },
                "score": s,
                "reasoning": f"turn {n} reasoning",
                "issues": (issues_for_turn or (lambda n: []))(n),
                "issue_details": "",
            })
        return json.dumps({"turns": turns})

    return judge, seen


# ------------------------------------------------------------
# Provenance: fixture text never reaches the judge
# ------------------------------------------------------------

def test_injected_audio_candidate_text_is_rejected_as_context():
    turns = [
        turn("assistant", "Please introduce yourself.",
             "livekit-agent-session", "medium"),
        turn("user", "SCRIPTED FIXTURE ANSWER about six years of "
             "experience", "injected-audio", "low"),
        turn("assistant", "Tell me about a hard project.",
             "livekit-agent-session", "medium"),
    ]
    kept, provenance = filter_turns(turns)
    assert provenance.candidate_turns_rejected == 1
    assert provenance.rejected_sources == {"injected-audio": 1}
    # The rejected text is gone; only a NOT_CAPTURED boundary
    # marker remains where the reply happened.
    assert all("SCRIPTED" not in t["text"] for t in kept)
    assert [t["role"] for t in kept] == ["assistant", "user", "assistant"]
    assert kept[1]["text"] == NOT_CAPTURED and kept[1].get("placeholder")

    units = build_interviewer_turns(kept)
    prompt = build_turn_prompt(units)
    assert "SCRIPTED FIXTURE ANSWER" not in prompt
    # The uncaptured reply is declared, not invented.
    assert NOT_CAPTURED in format_turn_units(units)


def test_prompt_is_invariant_to_candidate_answers_fixture(monkeypatch):
    """
    The proof requested: mutate / remove tests/e2e
    CANDIDATE_ANSWERS and show the evaluator input and output do
    not change. The only candidate text the judge can ever see
    is captured STT.
    """

    bot = pytest.importorskip("tests.fixtures.legacy_candidate_answers")

    judge, seen = stub_judge_factory(lambda n: 0.8)
    baseline = evaluate_captured_turns(CAPTURED, judge=judge)
    baseline_prompt = seen["prompt"]

    # 1. Replace the fixture list with a unique marker.
    monkeypatch.setattr(
        bot, "CANDIDATE_ANSWERS", ["ZORKLEBLATT-7742 unique marker"]
    )
    judge, seen = stub_judge_factory(lambda n: 0.8)
    mutated = evaluate_captured_turns(CAPTURED, judge=judge)
    assert seen["prompt"] == baseline_prompt
    assert "ZORKLEBLATT" not in seen["prompt"]
    assert mutated["results"] == baseline["results"]
    assert mutated["aggregate"] == baseline["aggregate"]

    # 2. Remove the fixture list entirely.
    monkeypatch.setattr(bot, "CANDIDATE_ANSWERS", [])
    judge, seen = stub_judge_factory(lambda n: 0.8)
    removed = evaluate_captured_turns(CAPTURED, judge=judge)
    assert seen["prompt"] == baseline_prompt
    assert removed["results"] == baseline["results"]

    # 3. Even if fixture text is fed in AS a transcript turn, its
    #    provenance tag keeps it out of the prompt.
    leaked = CAPTURED + [
        turn("user", bot.CANDIDATE_ANSWERS[0] if bot.CANDIDATE_ANSWERS
             else "ZORKLEBLATT-7742 unique marker", "injected-audio", "low")
    ]
    judge, seen = stub_judge_factory(lambda n: 0.8)
    evaluate_captured_turns(leaked, judge=judge)
    assert "ZORKLEBLATT" not in seen["prompt"]


def test_real_candidate_stt_is_shown_as_context_only():
    units = build_interviewer_turns(filter_turns(CAPTURED)[0])
    assert len(units) == 3
    data = format_turn_units(units)
    assert "Candidate [captured STT, source=livekit-candidate-stt]: " \
           "I'm Sam, six years of backend work in Go." in data
    # Turn 1 has no context; turn 3 sees both earlier exchanges.
    assert units[0].context == []
    assert [c.role for c in units[2].context] == [
        "assistant", "user", "assistant", "user"
    ]
    prompt = build_turn_prompt(units)
    assert "evaluate the AI INTERVIEWER - never the candidate" in prompt
    assert "Never generate, assume or imagine a candidate answer" in prompt


# ------------------------------------------------------------
# Utterance assembly
# ------------------------------------------------------------

def test_merge_consecutive_dedupes_restarted_captions():
    merged = merge_consecutive([
        turn("assistant", "It's great to hear about your diverse",
             "livekit-agent-session", "medium"),
        turn("assistant", "great to hear about your diverse experience "
             "in backend systems. What skills make you a fit?",
             "livekit-agent-session", "medium"),
        turn("user", "First sentence.", "livekit-candidate-stt", "high"),
        turn("user", "Second sentence.", "livekit-candidate-stt", "high"),
    ])
    assert [m["role"] for m in merged] == ["assistant", "user"]
    assert merged[0]["text"] == (
        "It's great to hear about your diverse experience in backend "
        "systems. What skills make you a fit?"
    )
    assert merged[1]["text"] == "First sentence. Second sentence."


# ------------------------------------------------------------
# Score flow: judge output -> validation -> aggregation
# ------------------------------------------------------------

def test_scores_and_issues_come_from_the_judge_not_a_constant():
    judge, _ = stub_judge_factory(
        lambda n: {1: 0.9, 2: 0.8, 3: 0.2}[n],
        lambda n: ["repetitive"] if n == 3 else [],
    )
    report = evaluate_captured_turns(CAPTURED, judge=judge)

    assert [r["score"] for r in report["results"]] == [0.9, 0.8, 0.2]
    assert report["aggregate"]["overall_score"] == round((0.9+0.8+0.2)/3, 4)
    assert report["aggregate"]["issue_counts"] == {"repetitive": 1}
    assert [f["number"] for f in report["aggregate"]["flagged_turns"]] == [3]
    # The interviewer text in the result is the CAPTURED text.
    assert report["results"][2]["interviewer_turn"].startswith(
        "Thanks. What is the hardest Go concurrency bug"
    )
    assert report["provenance"]["accepted_sources"] == {
        "livekit-candidate-stt": 2
    }

    # A different judge verdict -> a different output; nothing is
    # pinned in Python.
    judge2, _ = stub_judge_factory(lambda n: 0.5)
    report2 = evaluate_captured_turns(CAPTURED, judge=judge2)
    assert report2["aggregate"]["overall_score"] == 0.5
    assert report2["aggregate"]["flagged_turns"] == []


def test_validation_rejects_bad_judge_output():
    def bad_count(prompt, schema):
        return json.dumps({"turns": []})
    with pytest.raises(TurnValidationError, match="returned 0 entries"):
        evaluate_captured_turns(CAPTURED, judge=bad_count)

    judge, _ = stub_judge_factory(lambda n: 1.5)
    with pytest.raises(TurnValidationError, match="outside 0..1"):
        evaluate_captured_turns(CAPTURED, judge=judge)

    judge, _ = stub_judge_factory(lambda n: 0.7, lambda n: ["made_up"])
    with pytest.raises(TurnValidationError, match="unknown issue type"):
        evaluate_captured_turns(CAPTURED, judge=judge)

    assert set(ISSUE_TYPES) >= {"repetitive", "irrelevant", "nonsensical"}


def test_no_interviewer_turns_is_an_evidence_error_not_a_score():
    only_candidate = [
        turn("user", "Hello there.", "livekit-candidate-stt", "high")
    ]
    with pytest.raises(InterviewerEvidenceError):
        evaluate_captured_turns(only_candidate, judge=lambda p, s: "{}")
