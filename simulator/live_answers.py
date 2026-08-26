"""
Live wiring: actual interviewer question -> CandidateSimulator ->
spoken answer for the E2E drive (tests/e2e/test_bot_responsiveness).

    live AI interviewer
      -> its caption stream (window.__emhTranscriptEvents, already
         recorded in-page by TRANSCRIPT_HOOK_JS)
      -> extract_interviewer_question(): ONLY agent-attributed,
         completed, NEW text since the previous candidate answer
      -> CandidateSimulator.answer(turn, question)  (the one
         simulator; competency persona or robustness spec)
      -> synthesize_speech_wav()  (same say/afconvert path the
         fixtures used)
      -> injected into the fake microphone by the driver

No hardcoded answer bank exists on this path. When a valid live
question cannot be obtained the source raises
LiveQuestionUnavailable with the reason - it never substitutes
scripted text.
"""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

from collectors.transcript_capture import LiveKitEventsCapture
from evaluation.interviewer_turn_evaluation import merge_consecutive
from simulator.candidate_simulator import CandidateSimulator, SimulatedTurn

CANDIDATE_AUDIO_DIR = Path("artifacts/audio/candidate")
MIN_QUESTION_WORDS = 4
STABILITY_POLL_S = 1.5
STABILITY_MAX_S = 12.0


class LiveQuestionUnavailable(RuntimeError):
    """No valid completed interviewer question could be read live."""


class _InMemoryEvents(LiveKitEventsCapture):
    """LiveKitEventsCapture over an in-memory event slice."""

    def __init__(self, events: list[dict[str, Any]]):
        super().__init__(path=Path("/dev/null"))
        self._memory = events

    def _events(self) -> list[dict]:
        return self._memory


def interviewer_text_from_events(events: list[dict[str, Any]]) -> str:
    """
    Agent-attributed speech in `events`, merged into one string.
    Candidate-attributed captions (the agent's STT of the
    candidate, background noise on the candidate track) are
    excluded by the capture layer's speaker attribution.
    """

    turns = _InMemoryEvents(events).get_turns()
    merged = merge_consecutive([t for t in turns if t["role"] == "assistant"])
    return " ".join(t["text"] for t in merged).strip()


def extract_interviewer_question(
    events: list[dict[str, Any]],
    previous_question: str | None,
) -> tuple[str, dict[str, Any]]:
    """
    The completed interviewer question contained in `events`
    (which must be the slice recorded SINCE the previous candidate
    answer). Raises LiveQuestionUnavailable with diagnostics when
    there is none, it is too short, or it merely repeats the
    previous question (stale/duplicate caption).
    """

    text = interviewer_text_from_events(events)
    diagnostics = {
        "events_since_last_answer": len(events),
        "interviewer_words": len(text.split()),
        "text_preview": text[:160],
    }
    if len(text.split()) < MIN_QUESTION_WORDS:
        raise LiveQuestionUnavailable(
            "no completed interviewer utterance was captured since the "
            f"previous candidate answer ({diagnostics})"
        )
    if previous_question and text.strip().lower() == previous_question.strip().lower():
        raise LiveQuestionUnavailable(
            f"interviewer text since the previous answer is identical to "
            f"the previous question (stale/duplicate caption): {text[:120]!r}"
        )
    return text, diagnostics


def synthesize_speech_wav(text: str, out_dir: Path = CANDIDATE_AUDIO_DIR) -> Path:
    """
    macOS say + afconvert -> mono 48 kHz 16-bit WAV, content-
    hashed so identical text is synthesized once.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    wav = out_dir / f"candidate_{digest}.wav"
    if wav.exists():
        return wav
    aiff = wav.with_suffix(".aiff")
    try:
        subprocess.run(["say", "-o", str(aiff), text], check=True, capture_output=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@48000", "-c", "1", str(aiff), str(wav)],
            check=True, capture_output=True,
        )
    finally:
        aiff.unlink(missing_ok=True)
    return wav


class LiveSimulatorAnswerSource:
    """
    Per-turn answers for the live drive. `next_answer(turn, page)`
    reads the interviewer's actual question from the page, asks
    the ONE CandidateSimulator for the answer, synthesizes it and
    returns (wav_path, intended_text, question, diagnostics).
    """

    name = "live-simulator"

    def __init__(
        self,
        simulator: CandidateSimulator,
        *,
        save_path: Path | None = None,
        audio_dir: Path = CANDIDATE_AUDIO_DIR,
        log=print,
    ) -> None:
        self.simulator = simulator
        self.save_path = save_path
        self.audio_dir = audio_dir
        self.log = log
        self.cursor = 0                     # events consumed so far
        self.previous_question: str | None = None
        self.answers: list[SimulatedTurn] = []

    async def _events_since_cursor(self, page) -> list[dict[str, Any]]:
        events = await page.evaluate(
            "(start) => (window.__emhTranscriptEvents || []).slice(start)",
            self.cursor,
        )
        return events or []

    async def _stable_question(self, page) -> tuple[str, dict[str, Any]]:
        """
        Poll until the interviewer text since the cursor stops
        changing (caption stream complete); bounded by
        STABILITY_MAX_S. Partial captions can therefore never be
        answered.
        """

        deadline = time.monotonic() + STABILITY_MAX_S
        last_text: str | None = None
        last_error: LiveQuestionUnavailable | None = None
        while True:
            events = await self._events_since_cursor(page)
            try:
                text, diagnostics = extract_interviewer_question(
                    events, self.previous_question
                )
                last_error = None
            except LiveQuestionUnavailable as error:
                text, diagnostics, last_error = None, {}, error
            if text is not None and text == last_text:
                return text, diagnostics
            last_text = text
            if time.monotonic() >= deadline:
                if last_error is not None:
                    raise last_error
                # Text still changing at the cap: treat as complete
                # (wait_for_bot_silence already saw the bot stop).
                return text, {**diagnostics, "stability": "cap"}
            await asyncio.sleep(STABILITY_POLL_S)

    async def next_answer(self, turn: int, page) -> tuple[Path, str, str, dict[str, Any]]:
        question, diagnostics = await self._stable_question(page)
        # Everything recorded up to now belongs to this question;
        # the bot's reply to our answer will arrive after the cursor.
        self.cursor = self.cursor + diagnostics.get("events_since_last_answer", 0)
        total = await page.evaluate("() => (window.__emhTranscriptEvents || []).length")
        self.cursor = max(self.cursor, int(total or 0))
        self.previous_question = question

        record = await asyncio.to_thread(self.simulator.answer, turn, question)
        self.answers.append(record)
        if self.save_path is not None:
            self.simulator.save(self.save_path)

        wav = await asyncio.to_thread(synthesize_speech_wav, record.intended_text, self.audio_dir)
        self.log(
            f"[Turn {turn}] LIVE QUESTION ({diagnostics.get('interviewer_words')} words): "
            f"{question[:200]!r}\n        -> {self.simulator.mode} simulator "
            f"({self.simulator.model}"
            + (f", persona {record.persona_id}" if record.persona_id else f", spec {record.spec_key}")
            + f") answer ({len(record.intended_text.split())} words): "
            f"{record.intended_text[:200]!r}"
        )
        return wav, record.intended_text, question, diagnostics


class ScriptedAnswerSource:
    """
    OFFLINE HARNESS ONLY (tests/e2e_offline FakeBot): cycles fixed
    WAV paths. Never constructed by the live test - the live drive
    has no scripted fallback.
    """

    name = "scripted-offline"

    def __init__(self, paths: list[Path]):
        self.paths = list(paths)

    async def next_answer(self, turn: int, page) -> tuple[Path, str, str, dict[str, Any]]:
        wav = self.paths[(turn - 1) % len(self.paths)]
        return wav, wav.stem, "(offline harness - no live question)", {}
