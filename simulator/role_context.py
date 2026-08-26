"""
Job role / seniority context for the simulator and its judge.

Source of truth is the EMH socket.io `job-candidate-details`
payload (role, experience, skills, company.botName, candidate
work_experience_years). It is read from, in order:

  1. artifacts/transcripts/interview_context.json (if a capture
     run wrote one),
  2. the newest artifacts/debug/bot_responsiveness_*/
     websocket_frames.jsonl recorded by the E2E drive.

Nothing here is ever invented: when no payload exists,
load_role_context() returns None and callers must say so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONTEXT_PATH = Path("artifacts/transcripts/interview_context.json")
DEBUG_ROOT = Path("artifacts/debug")


@dataclass
class RoleContext:
    role: str
    skills: list[str] = field(default_factory=list)
    experience_required: str | None = None   # JD requirement, as sent
    candidate_years: float | None = None     # work_experience_years
    bot_name: str | None = None
    question_count: int | None = None
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def seniority(self) -> str:
        """Coarse seniority from the candidate's declared years."""

        years = self.candidate_years
        if years is None:
            return "unknown"
        if years < 2:
            return "junior"
        if years < 5:
            return "mid"
        return "senior"

    def prompt_block(self) -> str:
        lines = [f"Role: {self.role}"]
        if self.skills:
            lines.append("Required skills: " + ", ".join(self.skills))
        if self.experience_required not in (None, ""):
            lines.append(
                f"Experience required (JD): {self.experience_required}"
            )
        if self.candidate_years is not None:
            lines.append(
                f"Candidate declared experience: {self.candidate_years} "
                f"years ({self.seniority()})"
            )
        return "\n".join(lines)


def _from_payload(payload: dict[str, Any], source: str) -> RoleContext | None:
    details = payload.get("interviewDetails") or payload
    job = details.get("candidateJobDetails") or {}
    cand = details.get("candidateDetails") or {}
    role = job.get("role")
    if not role:
        return None
    years = cand.get("work_experience_years")
    months = cand.get("work_experience_months") or 0
    candidate_years = None
    if years is not None:
        try:
            candidate_years = round(float(years) + float(months) / 12, 2)
        except (TypeError, ValueError):
            candidate_years = None
    return RoleContext(
        role=str(role),
        skills=[str(s) for s in (job.get("skills") or [])],
        experience_required=(
            None if job.get("experience") is None
            else str(job.get("experience"))
        ),
        candidate_years=candidate_years,
        bot_name=(job.get("company") or {}).get("botName"),
        question_count=job.get("noOfQuestions"),
        source=source,
    )


def role_context_from_frame_rows(
    rows: list[dict[str, Any]],
    source: str = "live websocket frames",
) -> RoleContext | None:
    """
    Parse the `job-candidate-details` socket.io frame out of
    in-memory recorder rows ({"payload": '42["job-candidate-details",{...}]'}).
    Used by the LIVE drive so the simulator gets this interview's
    role, not a previous run's.
    """

    for row in rows:
        text = row.get("payload") or row.get("text") or row.get("data") or ""
        if "job-candidate-details" not in text or "interviewDetails" not in text:
            continue
        index = text.find("42[")
        if index < 0:
            continue
        try:
            event = json.loads(text[index + 2:])
        except ValueError:
            continue
        if (
            isinstance(event, list)
            and len(event) > 1
            and isinstance(event[1], dict)
        ):
            context = _from_payload(event[1], source)
            if context:
                return context
    return None


def _from_frames(path: Path) -> RoleContext | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if "job-candidate-details" not in line or "interviewDetails" not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        text = row.get("payload") or row.get("text") or row.get("data") or ""
        index = text.find("42[")
        if index < 0:
            continue
        try:
            event = json.loads(text[index + 2:])
        except ValueError:
            continue
        if (
            isinstance(event, list)
            and len(event) > 1
            and isinstance(event[1], dict)
        ):
            context = _from_payload(event[1], str(path))
            if context:
                return context
    return None


def load_role_context() -> RoleContext | None:
    if CONTEXT_PATH.exists():
        try:
            data = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if isinstance(data, dict):
            if "role" in data:
                return RoleContext(
                    role=str(data["role"]),
                    skills=list(data.get("skills") or []),
                    experience_required=data.get("experience_required"),
                    candidate_years=data.get("candidate_years"),
                    bot_name=data.get("bot_name"),
                    question_count=data.get("question_count"),
                    source=str(CONTEXT_PATH),
                )
            context = _from_payload(data, str(CONTEXT_PATH))
            if context:
                return context

    if DEBUG_ROOT.exists():
        for run_dir in sorted(DEBUG_ROOT.glob("bot_responsiveness_*"), reverse=True):
            frames = run_dir / "websocket_frames.jsonl"
            if frames.exists():
                context = _from_frames(frames)
                if context:
                    return context
    return None
