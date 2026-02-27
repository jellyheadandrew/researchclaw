"""
runner.py — Subprocess launcher with log capture and sandbox enforcement.

All experiment commands go through Runner, which:
  1. Validates the command against PathValidator before launching
  2. Sets RESEARCHCLAW_SANDBOX env var so researcher scripts can reference it
  3. Captures stdout/stderr to experiment_reports/{date}/trial_{N}/log/
  4. Returns a Popen handle for the Watcher to monitor
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .access_control import PathValidator
from .models import TrialInfo

if TYPE_CHECKING:
    from .env_manager import EnvManager

logger = logging.getLogger("researchclaw.runner")


@dataclass
class RunResult:
    pid: int
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    log_path: str


class Runner:
    """Launch and manage experiment subprocesses."""

    def __init__(
        self,
        base_dir: str,
        validator: PathValidator,
        env_manager: "EnvManager | None" = None,
        venv_path: str = "",
        conda_env: str = "",
    ):
        self.base_dir = Path(base_dir).resolve()
        self.validator = validator
        self.env_manager = env_manager
        self.venv_path = venv_path.strip()
        self.conda_env = conda_env.strip()

    def run_async(
        self,
        cmd: str,
        trial: TrialInfo,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.Popen:
        """
        Launch a shell command asynchronously (non-blocking).
        Returns a Popen object for the Watcher to monitor.

        Args:
            cmd: Shell command string to execute
            trial: Active trial (determines cwd and log directory)
            extra_env: Additional environment variables to inject

        stdout and stderr are written to:
            experiment_reports/{date}/trial_{N}/log/stdout.log
            experiment_reports/{date}/trial_{N}/log/stderr.log
        """
        # Validate the command paths
        self.validator.validate_shell_command(cmd)

        cwd = str(self.base_dir / trial.sandbox_path)
        log_dir = self.base_dir / trial.report_path / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        env = self._build_env(trial, extra_env)

        logger.info("Starting: %s (cwd=%s, PID=pending)", cmd, cwd)

        stdout_log = open(log_dir / "stdout.log", "a")
        stderr_log = open(log_dir / "stderr.log", "a")
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=stdout_log,
                stderr=stderr_log,
                text=True,
            )
        finally:
            # On Unix the child inherits the file descriptors across fork.
            # The parent must close its copies or they will not be
            # garbage-collected until this Runner is destroyed, causing
            # descriptor exhaustion for long-lived sessions with many trials.
            stdout_log.close()
            stderr_log.close()

        logger.info("Started PID %d", proc.pid)
        return proc

    def run_and_wait(
        self,
        cmd: str,
        trial: TrialInfo,
        timeout: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> RunResult:
        """
        Run a command synchronously and wait for completion.
        Captures output to log directory.

        Suitable for short commands (git operations, file diffs, etc.).
        For long-running experiments, use run_async() + Watcher instead.
        """
        self.validator.validate_shell_command(cmd)

        cwd = str(self.base_dir / trial.sandbox_path)
        log_dir = self.base_dir / trial.report_path / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        env = self._build_env(trial, extra_env)

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start
            logger.error("Command timed out after %.0fs: %s", duration, cmd)
            raise

        duration = time.monotonic() - start

        # Append to log files
        stdout_log = log_dir / "stdout.log"
        stderr_log = log_dir / "stderr.log"
        with open(stdout_log, "a") as f:
            f.write(result.stdout)
        with open(stderr_log, "a") as f:
            f.write(result.stderr)

        return RunResult(
            pid=0,  # no PID available for a completed process
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=duration,
            log_path=str(log_dir),
        )

    def _build_env(
        self,
        trial: TrialInfo,
        extra_env: dict[str, str] | None,
    ) -> dict[str, str]:
        """
        Build environment dict for subprocess.

        Activation priority:
          0. env_manager — if set, use the managed env path (copy-on-write envs)
          1. venv_path — if set, prepend {venv_path}/bin to PATH and set VIRTUAL_ENV
          2. conda_env — if set (and venv_path empty), locate conda env bin dir and
                         prepend it to PATH
          3. fallback  — system PATH unchanged
        """
        env = os.environ.copy()

        # --- managed environment (highest priority) ---
        if self.env_manager and self.env_manager.current_env_id >= 0:
            try:
                env_path = self.env_manager.get_active_env_path()
                if env_path.exists():
                    venv_bin = str(env_path / "bin")
                    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
                    env["VIRTUAL_ENV"] = str(env_path)
                    env.pop("CONDA_DEFAULT_ENV", None)
                    env.pop("CONDA_PREFIX", None)
            except RuntimeError:
                logger.warning("EnvManager has no active env; falling back to legacy config")

        # --- venv activation (legacy fallback, takes priority over conda) ---
        elif self.venv_path:
            venv = Path(self.venv_path)
            venv_bin = str(venv / "bin")
            env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
            env["VIRTUAL_ENV"] = str(venv)
            # Unset conda vars to avoid cross-contamination
            env.pop("CONDA_DEFAULT_ENV", None)
            env.pop("CONDA_PREFIX", None)

        # --- conda activation (only if venv_path not set) ---
        elif self.conda_env:
            conda_prefix = env.get("CONDA_PREFIX", "")
            candidate: Path | None = None

            if conda_prefix:
                # We are inside a conda env; look for the named env as a sibling
                base = Path(conda_prefix)
                sibling = base.parent / self.conda_env
                if (sibling / "bin").exists():
                    candidate = sibling
                else:
                    # Try {conda_base}/envs/{name} (for nested envs)
                    candidate = base / "envs" / self.conda_env
            else:
                # No active conda — try standard installation locations
                for prefix in (
                    Path.home() / "miniconda3",
                    Path.home() / "anaconda3",
                    Path.home() / "miniforge3",
                ):
                    c = prefix / "envs" / self.conda_env
                    if (c / "bin").exists():
                        candidate = c
                        break

            if candidate is not None and (candidate / "bin").exists():
                conda_bin = str(candidate / "bin")
                env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
                env["CONDA_DEFAULT_ENV"] = self.conda_env
                env["CONDA_PREFIX"] = str(candidate)
            else:
                logger.warning(
                    "conda_env=%r specified but env directory not found; "
                    "falling back to system Python",
                    self.conda_env,
                )

        # --- Always inject ResearchClaw context vars ---
        env["RESEARCHCLAW_SANDBOX"] = str(self.base_dir / trial.sandbox_path)
        env["RESEARCHCLAW_REPORT"] = str(self.base_dir / trial.report_path)
        env["RESEARCHCLAW_TRIAL"] = trial.trial_name

        if extra_env:
            env.update(extra_env)
        return env
