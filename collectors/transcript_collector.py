"""
Real interview transcript capture.

Turns observations made during an E2E interview run (rendered
page text plus the candidate audio the test injected) into the
transcript artifact every evaluation layer consumes:

    artifacts/transcripts/actual_transcript.json
        [{"role": "assistant"|"user", "content": "..."}, ...]

The collector is deliberately Playwright-agnostic: the E2E test
feeds it plain text (DOM line diffs, injected answer text) and
it handles role attribution, ordering, de-duplication and
serialization. That keeps it fully unit-testable offline.

Also writes artifacts/transcripts/audio_turn_records.json - the
per-turn reference/STT pairs consumed by the jiwer/pyannote
audio evaluation layer (evaluation.audio_pipeline).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


TRANSCRIPT_DIR = Path("artifacts/transcripts")
TRANSCRIPT_PATH = TRANSCRIPT_DIR / "actual_transcript.json"
AUDIO_RECORDS_PATH = TRANSCRIPT_DIR / "audio_turn_records.json"
TRANSCRIPT_STATUS_PATH = TRANSCRIPT_DIR / "actual_transcript_status.json"

# Page chrome that must never be mistaken for conversation.
_UI_NOISE = re.compile(
    r"^(start|continue|submit|next|skip|close|done|mute|unmute"
    r"|end interview|leave|settings|camera|microphone|speaker"
    r"|recording|\d{1,2}:\d{2})\b|^[\W\d\s]*$",
    re.IGNORECASE,
)

# Guided-tour / interview-room chrome: long prose-like lines
# that pass the generic noise filter but are app UI, not the
# bot speaking. Matched ANYWHERE in the line (the room status
# bar concatenates them: "Frontend developer Elapsed time :
# 01 secs AI is speaking Jamie (Interviewer) Welcome to Your
# AI Interview Let's take a quick tour...").
_UI_CHROME = re.compile(
    r"elapsed time\s*:"
    r"|ai is speaking"
    r"|\(interviewer\)"
    r"|welcome to your ai interview"
    r"|take a quick tour"
    r"|status indicators, controls",
    re.IGNORECASE,
)

_MIN_CONTENT_CHARS = 12


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def looks_like_conversation(line: str) -> bool:
    """Filter DOM lines down to plausible spoken content."""

    line = line.strip()
    if len(line) < _MIN_CONTENT_CHARS:
        return False
    if _UI_NOISE.search(line):
        return False
    if _UI_CHROME.search(line):
        return False
    return True


@dataclass
class CapturedTurn:
    role: str
    content: str
    source: str
    turn: int


@dataclass
class TranscriptCollector:
    """
    Accumulates conversation turns during an interview run.

    Roles:
      assistant - text the AI interviewer produced (captured
                  from newly rendered page text)
      user      - what the candidate said (the audio the test
                  actually injected, optionally refined with the
                  app's own rendered STT of that audio)
    """

    turns: list[CapturedTurn] = field(default_factory=list)
    _seen: set[tuple[str, str]] = field(default_factory=set)

    def record_turn(
        self, role: str, content: str, *, source: str, turn: int
    ) -> bool:
        """
        Add one conversation turn. Returns False for duplicates,
        empty content or invalid roles instead of corrupting the
        artifact.
        """

        if role not in ("assistant", "user"):
            return False
        content = re.sub(r"\s+", " ", content or "").strip()
        if not content:
            return False
        key = (role, _normalize(content))
        if key in self._seen:
            return False
        self._seen.add(key)
        self.turns.append(
            CapturedTurn(
                role=role, content=content, source=source, turn=turn
            )
        )
        return True

    def record_assistant_lines(
        self, lines: list[str], *, source: str, turn: int
    ) -> int:
        """
        Record newly rendered page lines as one assistant turn
        (joined in DOM order, UI noise filtered). Returns the
        number of turns actually added (0 or 1).
        """

        content_lines = [
            line for line in lines if looks_like_conversation(line)
        ]
        if not content_lines:
            return 0
        added = self.record_turn(
            "assistant",
            " ".join(content_lines),
            source=source,
            turn=turn,
        )
        return int(added)

    def as_transcript(self) -> list[dict[str, str]]:
        return [
            {"role": captured.role, "content": captured.content}
            for captured in self.turns
        ]

    def has_conversation(self) -> bool:
        roles = {captured.role for captured in self.turns}
        return "assistant" in roles or "user" in roles

    def save(self, path: Path = TRANSCRIPT_PATH) -> Path | None:
        """
        Write the transcript artifact. Returns the path, or None
        when nothing was captured (an empty file would poison
        downstream evaluation with a fake "empty interview").
        """

        if not self.has_conversation():
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_transcript(), indent=2),
            encoding="utf-8",
        )

        metadata = [
            {
                "turn": captured.turn,
                "role": captured.role,
                "source": captured.source,
                "content": captured.content,
            }
            for captured in self.turns
        ]
        path.with_name(path.stem + "_capture_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return path


def save_transcript_status(
    *,
    complete: bool,
    turn_count: int,
    reached_cap: bool,
    captured_at: float,
    room_disconnected: bool | None = None,
    conclusion_reason: str | None = None,
    path: Path = TRANSCRIPT_STATUS_PATH,
) -> Path:
    """
    Persist whole-interview capture status so the evaluation
    layer can reject stale/truncated transcripts. `captured_at`
    is a unix timestamp supplied by the caller (the E2E test).
    `room_disconnected`/`conclusion_reason` record WHY an
    incomplete interview ended, so downstream classification can
    distinguish a room/server disconnect (environment) from an
    agent that died (bot failure).
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "complete": complete,
        "turn_count": turn_count,
        "reached_cap": reached_cap,
        "captured_at": captured_at,
    }
    if room_disconnected is not None:
        payload["room_disconnected"] = room_disconnected
    if conclusion_reason is not None:
        payload["conclusion_reason"] = conclusion_reason
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return path


def save_audio_turn_records(
    records: list[dict],
    path: Path = AUDIO_RECORDS_PATH,
) -> Path | None:
    """
    Persist per-turn audio evaluation records (reference vs STT
    transcript, speaker segments when available) for
    evaluation.audio_pipeline.load_audio_turn_records.
    """

    if not records:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path
