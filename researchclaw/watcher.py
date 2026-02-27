"""
watcher.py — Experiment process monitoring.

Watches a running experiment subprocess for:
  - FINISHED: process exited with code 0
  - CRASHED: process exited with non-zero code
  - HUNG: no new log output for heartbeat_timeout seconds
  - NAN_DETECTED: NaN/Inf detected in log output (training instability)
  - STATUS_UPDATE: periodic progress update (every status_update_interval seconds)

Also monitors:
  - nvidia-smi: GPU utilization and memory usage
  - New output files in the sandbox (checkpoints, plots, CSVs)

All logs go to: experiment_reports/{date}/trial_{N}/log/
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generator

from .models import TrialInfo
from .utils import format_duration, tail_file

logger = logging.getLogger("researchclaw.watcher")


class ExperimentEvent(Enum):
    FINISHED = "finished"
    CRASHED = "crashed"
    HUNG = "hung"
    NAN_DETECTED = "nan_detected"
    STATUS_UPDATE = "status_update"


@dataclass
class WatcherState:
    pid: int
    start_time: float
    last_log_time: float
    last_log_size: int = 0
    last_gpu_util: float = 0.0
    last_status_update_time: float = 0.0
    new_files: list[str] = field(default_factory=list)


@dataclass
class ExperimentStatus:
    event: ExperimentEvent
    trial_name: str
    duration: str
    log_tail: str
    gpu_info: dict
    new_files: list[str]
    returncode: int | None = None
    message: str = ""


class Watcher:
    """Monitor a running experiment process and yield events."""

    def __init__(
        self,
        base_dir: str,
        poll_interval: int = 10,
        heartbeat_timeout: int = 300,
        status_update_interval: int = 7200,
        gpu_idle_threshold: int = 60,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.poll_interval = poll_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.status_update_interval = status_update_interval
        self.gpu_idle_threshold = gpu_idle_threshold

    def watch(
        self,
        proc: subprocess.Popen,
        trial: TrialInfo,
    ) -> Generator[ExperimentStatus, None, None]:
        """
        Monitor the experiment process. Yields ExperimentStatus objects.
        Caller should run this in a background thread.

        Usage:
            proc = runner.run_async(cmd, trial)
            for status in watcher.watch(proc, trial):
                messenger.send(format_status(status))
                if status.event in (FINISHED, CRASHED):
                    break
        """
        log_dir = self.base_dir / trial.report_path / "log"
        stdout_log = log_dir / "stdout.log"

        state = WatcherState(
            pid=proc.pid,
            start_time=time.monotonic(),
            last_log_time=time.monotonic(),
            last_status_update_time=time.monotonic(),
        )

        logger.info("Watching PID %d for trial %s", proc.pid, trial.trial_name)

        while True:
            time.sleep(self.poll_interval)

            # Check if process finished
            returncode = proc.poll()
            if returncode is not None:
                duration = format_duration(time.monotonic() - state.start_time)
                log_tail = tail_file(str(stdout_log), n=30)
                gpu_info = self.check_gpu()
                event = ExperimentEvent.FINISHED if returncode == 0 else ExperimentEvent.CRASHED
                logger.info(
                    "Process %d %s (rc=%d, duration=%s)",
                    proc.pid, event.value, returncode, duration
                )
                yield ExperimentStatus(
                    event=event,
                    trial_name=trial.trial_name,
                    duration=duration,
                    log_tail=log_tail,
                    gpu_info=gpu_info,
                    new_files=self._scan_new_files(trial, state),
                    returncode=returncode,
                )
                return

            # Check log activity
            log_size = stdout_log.stat().st_size if stdout_log.exists() else 0
            if log_size > state.last_log_size:
                state.last_log_time = time.monotonic()
                state.last_log_size = log_size

                # Check for NaN in new log content
                log_tail = tail_file(str(stdout_log), n=10)
                if self.detect_nan(log_tail):
                    logger.warning("NaN/Inf detected in log output for %s", trial.trial_name)
                    yield ExperimentStatus(
                        event=ExperimentEvent.NAN_DETECTED,
                        trial_name=trial.trial_name,
                        duration=format_duration(time.monotonic() - state.start_time),
                        log_tail=log_tail,
                        gpu_info=self.check_gpu(),
                        new_files=[],
                        message="NaN or Inf detected in training output. Training may be unstable.",
                    )

            # Check for hung process
            time_since_log = time.monotonic() - state.last_log_time
            if time_since_log > self.heartbeat_timeout:
                logger.warning(
                    "No log output for %.0fs — process may be hung (PID %d)",
                    time_since_log, proc.pid,
                )
                yield ExperimentStatus(
                    event=ExperimentEvent.HUNG,
                    trial_name=trial.trial_name,
                    duration=format_duration(time.monotonic() - state.start_time),
                    log_tail=tail_file(str(stdout_log), n=10),
                    gpu_info=self.check_gpu(),
                    new_files=[],
                    message=f"No log output for {format_duration(time_since_log)}. Process may be hung.",
                )
                # Reset timer to avoid spamming
                state.last_log_time = time.monotonic()

            # Periodic status update
            if (
                self.status_update_interval > 0
                and time.monotonic() - state.last_status_update_time > self.status_update_interval
            ):
                state.last_status_update_time = time.monotonic()
                yield ExperimentStatus(
                    event=ExperimentEvent.STATUS_UPDATE,
                    trial_name=trial.trial_name,
                    duration=format_duration(time.monotonic() - state.start_time),
                    log_tail=tail_file(str(stdout_log), n=20),
                    gpu_info=self.check_gpu(),
                    new_files=self._scan_new_files(trial, state),
                )

    def check_gpu(self) -> dict:
        """
        Query GPU status via nvidia-smi.
        Returns dict with 'utilization', 'memory_used', 'memory_total'.
        Returns empty dict if nvidia-smi is not available.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 3:
                    return {
                        "utilization": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                    }
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return {}

    def detect_nan(self, log_tail: str) -> bool:
        """
        Scan recent log output for NaN/Inf values in loss or metric lines.
        Avoids false positives from legitimate mentions of "nan" in text.
        """
        import re
        # Match loss/metric values that are nan or inf
        pattern = re.compile(
            r'\b(?:loss|acc|accuracy|metric|val_loss|train_loss)\s*[=:]\s*(?:nan|inf|-inf)\b',
            re.IGNORECASE,
        )
        return bool(pattern.search(log_tail))

    def _scan_new_files(self, trial: TrialInfo, state: WatcherState) -> list[str]:
        """List notable output files in the sandbox (checkpoints, plots, CSVs)."""
        sandbox = self.base_dir / trial.sandbox_path
        if not sandbox.exists():
            return []

        notable_extensions = {".pt", ".ckpt", ".safetensors", ".png", ".pdf", ".csv", ".json"}
        new_files = []
        for p in sandbox.rglob("*"):
            if p.is_file() and p.suffix in notable_extensions:
                new_files.append(str(p.relative_to(self.base_dir)))

        return new_files
