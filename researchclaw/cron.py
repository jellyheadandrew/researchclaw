"""Simple timer-based cron system for ResearchClaw."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable


@dataclass
class CronJob:
    job_id: str
    interval_seconds: int
    callback: Callable[[], None]
    last_run: datetime | None = None
    next_run: datetime | None = None
    enabled: bool = True

    def is_due(self) -> bool:
        if not self.enabled:
            return False
        if self.next_run is None:
            return True
        return datetime.now() >= self.next_run

    def mark_run(self) -> None:
        self.last_run = datetime.now()
        self.next_run = self.last_run + timedelta(seconds=self.interval_seconds)


class CronScheduler:
    """Background thread scheduler for periodic jobs."""

    def __init__(self, state_path: Path | None = None):
        self._jobs: dict[str, CronJob] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_path = state_path

    def register(self, job: CronJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def unregister(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def set_enabled(self, job_id: str, enabled: bool) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = enabled

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._load_state()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._save_state()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                due_jobs = [j for j in self._jobs.values() if j.is_due()]

            for job in due_jobs:
                try:
                    job.callback()
                except Exception:
                    pass
                job.mark_run()

            if due_jobs:
                self._save_state()

            self._stop_event.wait(timeout=30)

    def _save_state(self) -> None:
        if not self._state_path:
            return
        state: dict[str, dict] = {}
        with self._lock:
            for jid, job in self._jobs.items():
                state[jid] = {
                    "last_run": job.last_run.isoformat() if job.last_run else None,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                    "enabled": job.enabled,
                }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_state(self) -> None:
        if not self._state_path or not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return

        with self._lock:
            for jid, data in raw.items():
                if jid in self._jobs:
                    job = self._jobs[jid]
                    if data.get("last_run"):
                        job.last_run = datetime.fromisoformat(data["last_run"])
                    if data.get("next_run"):
                        job.next_run = datetime.fromisoformat(data["next_run"])
                    job.enabled = data.get("enabled", True)


CADENCE_SECONDS = {
    "disabled": 0,
    "ask": 0,
    "hourly": 3600,
    "6h": 21600,
    "daily": 86400,
}


def cadence_to_seconds(cadence: str) -> int:
    return CADENCE_SECONDS.get(cadence, 0)
