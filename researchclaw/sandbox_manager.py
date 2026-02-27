"""
sandbox_manager.py — Trial creation, lifecycle management, and persistence.

Manages the full lifecycle of experiment trials:
  - CREATE: Copy github_codes/ → sandbox/{date}/trial_{N}/, create report dirs.
            reference/ is NOT copied — the agent reads it directly from base_dir (read-only).
  - ACTIVE: Agent has write access to sandbox and report dirs; reference/ remains read-only.
  - REVIEW: Experiment done, REPORT.md generated
  - FINALIZE: User approves (merge) or rejects (preserve for reference)

Trial state is persisted to .trials.jsonl (append-only log).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from .access_control import PathValidator
from .models import TrialInfo

# Default patterns to exclude when copying github_codes/ into sandbox
DEFAULT_IGNORE_PATTERNS = (
    ".git",
    "__pycache__",
    "*.pyc",
    "wandb",
    "outputs",
    "checkpoints",
    "*.pt",
    "*.ckpt",
    "*.safetensors",
    "*.bin",
    "node_modules",
)


class SandboxManager:
    """Manages trial creation, lifecycle transitions, and persistent state."""

    def __init__(
        self,
        base_dir: str,
        validator: PathValidator,
        ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.validator = validator
        self.ignore_patterns = ignore_patterns
        self.trials_file = self.base_dir / ".trials.jsonl"

    # ------------------------------------------------------------------
    # Trial lifecycle
    # ------------------------------------------------------------------

    def create_trial(self, goal: str = "") -> TrialInfo:
        """
        Create a new trial for today.

        1. Determine today's date (YYYYMMDD)
        2. Find the next trial number (max existing + 1)
        3. Copy github_codes/ → sandbox/{date}/trial_{N}/
        4. Create experiment_reports/{date}/trial_{N}/{log,eval_results}/
        5. Persist TrialInfo to .trials.jsonl
        6. Grant write access via validator.set_trial()
        7. Return TrialInfo
        """
        today = datetime.now().strftime("%Y%m%d")
        next_num = self._next_trial_number(today)

        trial = TrialInfo(
            date=today,
            number=next_num,
            status="active",
            started_at=datetime.now().isoformat(),
            goal=goal,
        )

        # Copy codebase into sandbox (bypass validator — this is infrastructure setup)
        src = self.base_dir / "github_codes"
        dst = self.base_dir / trial.sandbox_path
        if not src.exists():
            raise FileNotFoundError(
                f"github_codes/ not found at {src}. "
                "Please clone your repository there first."
            )
        if dst.exists():
            raise FileExistsError(f"Trial directory already exists: {dst}")

        shutil.copytree(
            src, dst,
            ignore=shutil.ignore_patterns(*self.ignore_patterns),
        )

        # Create report directory structure
        report_dir = self.base_dir / trial.report_path
        (report_dir / "log").mkdir(parents=True, exist_ok=True)
        (report_dir / "eval_results").mkdir(parents=True, exist_ok=True)

        # Persist and grant access
        self._save_trial(trial)
        self.validator.set_trial(trial)

        return trial

    def finalize_trial(self, trial: TrialInfo, status: str) -> None:
        """
        Mark trial as approved or rejected. Permanently revokes write access.

        Args:
            trial: The TrialInfo to finalize (will be mutated in-place)
            status: Must be "approved" or "rejected"
        """
        if status not in ("approved", "rejected"):
            raise ValueError(f"status must be 'approved' or 'rejected', got: {status!r}")

        trial.status = status
        trial.finished_at = datetime.now().isoformat()
        self._save_trial(trial)

        # Revoke write access immediately
        self.validator.set_trial(None)

    def mark_review(self, trial: TrialInfo) -> None:
        """Transition trial to 'review' state (experiment done, awaiting researcher decision)."""
        trial.status = "review"
        self._save_trial(trial)
        # Write access remains until finalize_trial() is called

    def reactivate_trial(self, trial: TrialInfo) -> None:
        """Transition a trial from 'review' back to 'active' (for "continue iterating").

        Re-grants write access to the sandbox so the agent can apply more changes.
        """
        if trial.status != "review":
            raise ValueError(f"Can only reactivate a 'review' trial, got: {trial.status!r}")
        trial.status = "active"
        self._save_trial(trial)
        self.validator.set_trial(trial)

    # ------------------------------------------------------------------
    # Trial queries
    # ------------------------------------------------------------------

    def get_active_trial(self) -> TrialInfo | None:
        """
        Return the most recent trial with status='active', or None.
        Scans .trials.jsonl from the end (most recent first).
        """
        trials = self._load_all_trials()
        for trial in reversed(trials):
            if trial.status == "active":
                return trial
        return None

    def get_review_trial(self) -> TrialInfo | None:
        """Return the most recent trial with status='review', or None."""
        trials = self._load_all_trials()
        for trial in reversed(trials):
            if trial.status == "review":
                return trial
        return None

    def get_latest_trial(self) -> TrialInfo | None:
        """Return the most recently created trial regardless of status."""
        trials = self._load_all_trials()
        return trials[-1] if trials else None

    def list_trials(self, date: str | None = None) -> list[TrialInfo]:
        """
        Return all trials, optionally filtered by date (YYYYMMDD).
        Returns deduplicated list: last entry per (date, number) wins.
        """
        trials = self._load_all_trials()
        if date:
            trials = [t for t in trials if t.date == date]
        return trials

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_trial_number(self, date: str) -> int:
        """Find the highest trial number for this date and return +1."""
        sandbox_date_dir = self.base_dir / "sandbox" / date
        if not sandbox_date_dir.exists():
            return 1
        existing = [
            int(d.name.split("_")[1])
            for d in sandbox_date_dir.iterdir()
            if d.is_dir() and d.name.startswith("trial_") and d.name.split("_")[1].isdigit()
        ]
        return max(existing, default=0) + 1

    def _save_trial(self, trial: TrialInfo) -> None:
        """Append current trial state to .trials.jsonl (one JSON object per line)."""
        with open(self.trials_file, "a") as f:
            f.write(json.dumps(trial.to_dict()) + "\n")

    def _load_all_trials(self) -> list[TrialInfo]:
        """
        Read .trials.jsonl and return deduplicated list.
        Since we append on every state change, we keep only the LAST
        entry per (date, number) pair to get the current state.
        """
        if not self.trials_file.exists():
            return []

        # Map (date, number) → latest TrialInfo seen
        seen: dict[tuple[str, int], TrialInfo] = {}
        with open(self.trials_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    trial = TrialInfo.from_dict(d)
                    seen[(trial.date, trial.number)] = trial
                except (json.JSONDecodeError, KeyError):
                    continue  # skip malformed lines

        # Sort by (date, number) chronologically
        return sorted(seen.values(), key=lambda t: (t.date, t.number))
