from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from researchclaw.config import ResearchClawConfig
from researchclaw.models import TrialMeta


SANDBOX_DIR_NAME = "sandbox_researchclaw"


class SandboxManager:
    """Manages the sandbox_researchclaw/ directory structure and trial creation."""

    @staticmethod
    def sandbox_path(project_dir: str | Path) -> Path:
        """Return the sandbox root path for a project."""
        return Path(project_dir) / SANDBOX_DIR_NAME

    @staticmethod
    def is_initialized(project_dir: str | Path) -> bool:
        """Return True if sandbox_researchclaw/ exists in the project directory."""
        return SandboxManager.sandbox_path(project_dir).is_dir()

    @staticmethod
    def initialize(project_dir: str | Path) -> Path:
        """Initialize the sandbox_researchclaw/ directory structure.

        Creates:
          sandbox_researchclaw/
            project_settings/
              researchclaw.yaml   (defaults)
              PROJECT_MEMORY.md   (empty)
            EXPERIMENT_LOGS.md    (empty)
            experiments/

        Idempotent: skips files/dirs that already exist.
        Returns the sandbox root path.
        """
        sandbox = SandboxManager.sandbox_path(project_dir)

        # project_settings/
        settings_dir = sandbox / "project_settings"
        settings_dir.mkdir(parents=True, exist_ok=True)

        # researchclaw.yaml with defaults (only if not exists)
        config_path = settings_dir / "researchclaw.yaml"
        if not config_path.exists():
            ResearchClawConfig().save_to_yaml(config_path)

        # PROJECT_MEMORY.md (empty, only if not exists)
        memory_path = settings_dir / "PROJECT_MEMORY.md"
        if not memory_path.exists():
            memory_path.touch()

        # EXPERIMENT_LOGS.md (empty, only if not exists)
        logs_path = sandbox / "EXPERIMENT_LOGS.md"
        if not logs_path.exists():
            logs_path.touch()

        # experiments/
        experiments_dir = sandbox / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)

        return sandbox

    @staticmethod
    def create_trial(project_dir: str | Path, date: datetime | None = None) -> Path:
        """Create a new trial directory under experiments/.

        Trial naming: {YYYYMMDD}_trial_{N:03}
        N resets per date (first trial on a given date is 001).

        Args:
            project_dir: Project root directory.
            date: Optional datetime for trial date. Defaults to UTC now.

        Returns:
            Path to the created trial directory.
        """
        if date is None:
            date = datetime.now(timezone.utc)
        date_str = date.strftime("%Y%m%d")

        experiments_dir = SandboxManager.sandbox_path(project_dir) / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)

        # Find the next trial number for this date
        existing = sorted(
            d.name
            for d in experiments_dir.iterdir()
            if d.is_dir() and d.name.startswith(date_str + "_trial_")
        )
        if existing:
            # Parse the last trial number
            last_name = existing[-1]
            last_num = int(last_name.split("_trial_")[1])
            trial_num = last_num + 1
        else:
            trial_num = 1

        trial_name = f"{date_str}_trial_{trial_num:03d}"
        trial_dir = experiments_dir / trial_name

        # Create trial directory structure
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "experiment" / "codes_exp").mkdir(parents=True, exist_ok=True)
        (trial_dir / "experiment" / "codes_eval").mkdir(parents=True, exist_ok=True)
        (trial_dir / "experiment" / "outputs").mkdir(parents=True, exist_ok=True)

        # Create empty requirements.txt
        (trial_dir / "requirements.txt").touch()

        # Create meta.json with defaults
        meta = TrialMeta(trial_number=trial_num)
        meta.to_json(trial_dir / "meta.json")

        return trial_dir

    @staticmethod
    def save_trial_meta(trial_dir: str | Path, meta: TrialMeta) -> None:
        """Write a TrialMeta to meta.json in the given trial directory."""
        meta.to_json(Path(trial_dir) / "meta.json")

    @staticmethod
    def get_trial_meta(trial_dir: str | Path) -> TrialMeta:
        """Read and return a TrialMeta from meta.json in the given trial directory."""
        return TrialMeta.from_json(Path(trial_dir) / "meta.json")

    @staticmethod
    def get_latest_trial(project_dir: str | Path) -> Path | None:
        """Return the path to the most recent trial directory, or None if no trials exist."""
        experiments_dir = SandboxManager.sandbox_path(project_dir) / "experiments"
        if not experiments_dir.is_dir():
            return None
        trial_dirs = sorted(
            d for d in experiments_dir.iterdir() if d.is_dir() and "_trial_" in d.name
        )
        if not trial_dirs:
            return None
        return trial_dirs[-1]
