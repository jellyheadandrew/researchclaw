from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class VenvManager:
    """Manages per-trial virtual environments for experiment isolation.

    ResearchClaw NEVER activates experiment venvs — all execution happens
    via subprocess (run_exp.sh / run_eval.sh).
    """

    @staticmethod
    def ensure_venv(trial_dir: str | Path, python_command: str = "python3") -> Path:
        """Create a venv in trial_dir/env/ if it does not already exist.

        Args:
            trial_dir: Path to the trial directory.
            python_command: Python interpreter to use for venv creation.

        Returns:
            Path to the venv directory (trial_dir/env/).

        Raises:
            subprocess.CalledProcessError: If venv creation fails.
        """
        trial_dir = Path(trial_dir)
        venv_dir = trial_dir / "env"

        if venv_dir.is_dir():
            return venv_dir

        subprocess.run(
            [python_command, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return venv_dir

    @staticmethod
    def install_requirements(trial_dir: str | Path) -> bool:
        """Install requirements.txt into the trial's venv.

        Uses the pip inside trial_dir/env/ to install packages from
        trial_dir/requirements.txt. Skips if requirements.txt is empty
        or does not exist.

        Args:
            trial_dir: Path to the trial directory.

        Returns:
            True if installation succeeded or was skipped, False on failure.
        """
        trial_dir = Path(trial_dir)
        requirements_path = trial_dir / "requirements.txt"
        venv_pip = trial_dir / "env" / "bin" / "pip"

        # Skip if no requirements or empty file
        if not requirements_path.exists():
            return True
        content = requirements_path.read_text().strip()
        if not content:
            return True

        # Skip if venv doesn't exist
        if not venv_pip.exists():
            return False

        try:
            subprocess.run(
                [str(venv_pip), "install", "-r", str(requirements_path), "-q"],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def get_venv_python(trial_dir: str | Path) -> Path:
        """Return the path to the Python interpreter inside the trial's venv.

        Args:
            trial_dir: Path to the trial directory.

        Returns:
            Path to env/bin/python within the trial directory.
        """
        return Path(trial_dir) / "env" / "bin" / "python"

    @staticmethod
    def venv_exists(trial_dir: str | Path) -> bool:
        """Check if a venv already exists for this trial.

        Args:
            trial_dir: Path to the trial directory.

        Returns:
            True if env/ directory exists in the trial directory.
        """
        return (Path(trial_dir) / "env").is_dir()
