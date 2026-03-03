from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TrialMeta:
    """Metadata for a single experiment trial, persisted as meta.json."""

    trial_number: int = 1
    status: str = "pending"
    state: str = "experiment_plan"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    plan_approved_at: str | None = None
    experiment_exit_code: int | None = None
    experiment_retry_count: int = 0
    eval_exit_code: int | None = None
    eval_retry_count: int = 0
    decision: str | None = None
    decision_reasoning: str | None = None

    def to_json(self, path: str | Path) -> None:
        """Write this TrialMeta to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> TrialMeta:
        """Read a TrialMeta from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
