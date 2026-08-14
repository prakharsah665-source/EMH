"""
Static security audit of the REAL captured interview transcript.

If a transcript artifact exists
(artifacts/transcripts/actual_transcript.json, written by the
bot-responsiveness E2E run) every AI-interviewer turn is checked
for PII/secret leakage and system-prompt/rubric disclosure. No
live calls; if no capture exists yet the test SKIPS with the
command to produce one.
"""

import pytest

from evaluation.security import audit_transcript_assistant_turns
from evaluation.transcript import (
    get_transcript_path,
    load_transcript_turns,
)


def load_real_turns():
    path = get_transcript_path()
    if not path.exists():
        pytest.skip(
            f"No captured transcript at {path}. Produce one with: "
            "pytest tests/e2e/test_bot_responsiveness.py -s"
        )
    return load_transcript_turns(path)


def test_captured_interviewer_turns_leak_nothing():
    turns = load_real_turns()

    # A "clean" verdict over a capture with (almost) no real
    # interviewer speech is vacuous - skip loudly instead of
    # reporting a green audit backed by no evidence.
    assistant_turns = [
        t for t in turns if t.get("role") == "assistant"
    ]
    print(
        f"Auditing {len(assistant_turns)} interviewer turn(s) "
        f"out of {len(turns)} total transcript turns."
    )
    if len(assistant_turns) < 2:
        pytest.skip(
            f"Only {len(assistant_turns)} interviewer turn(s) "
            "captured - not enough real interviewer output to "
            "audit. Re-run the E2E capture: "
            "pytest tests/e2e/test_bot_responsiveness.py -s"
        )

    findings = audit_transcript_assistant_turns(turns)

    for finding in findings:
        print(finding.as_dict())

    assert not findings, (
        "The captured EMH interviewer output leaked PII/secrets "
        "or disclosed its system prompt/rubric:\n"
        + "\n".join(
            f"  [{f.category.value}] {f.reason} -> {f.evidence}"
            for f in findings
        )
    )
