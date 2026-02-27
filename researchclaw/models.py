"""
Shared data models for ResearchClaw.
TrialInfo is defined here to avoid circular imports between
access_control.py and sandbox_manager.py, both of which need it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TrialStatus(str, Enum):
    """Trial lifecycle status.

    Extends ``str`` so that comparisons like ``status == "active"`` still
    work, ensuring backward compatibility with ``.trials.jsonl``.
    """

    ACTIVE = "active"
    REVIEW = "review"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class TrialInfo:
    date: str                        # YYYYMMDD
    number: int                      # 1-indexed trial number
    status: TrialStatus              # lifecycle stage
    started_at: str                  # ISO timestamp
    finished_at: str | None = None
    command_history: list[str] = field(default_factory=list)
    goal: str = ""                   # research goal description for this trial
    env_id: int | None = None        # which env_NNN this trial uses (None = legacy/unmanaged)

    @property
    def trial_name(self) -> str:
        return f"trial_{self.number:03d}"

    @property
    def sandbox_path(self) -> Path:
        return Path(f"sandbox/{self.date}/{self.trial_name}")

    @property
    def report_path(self) -> Path:
        return Path(f"experiment_reports/{self.date}/{self.trial_name}")

    @property
    def is_writable(self) -> bool:
        """Write access is only allowed when the trial is actively in progress."""
        return self.status == TrialStatus.ACTIVE

    @property
    def is_reviewable(self) -> bool:
        """Trial can be reviewed (approve/reject/continue)."""
        return self.status == TrialStatus.REVIEW

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "number": self.number,
            "status": self.status.value if isinstance(self.status, TrialStatus) else self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "command_history": self.command_history,
            "goal": self.goal,
            "env_id": self.env_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrialInfo":
        raw_status = d["status"]
        try:
            status = TrialStatus(raw_status)
        except ValueError:
            status = raw_status  # type: ignore[assignment]  # graceful fallback
        return cls(
            date=d["date"],
            number=d["number"],
            status=status,
            started_at=d["started_at"],
            finished_at=d.get("finished_at"),
            command_history=d.get("command_history", []),
            goal=d.get("goal", ""),
            env_id=d.get("env_id"),
        )
