from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .states import State, TrialStatus


@dataclass
class ProjectGitConfig:
    remote_url: str = ""
    default_branch: str = "main"
    auth_source: str = "system"

    def to_dict(self) -> dict[str, str]:
        return {
            "remote_url": self.remote_url,
            "default_branch": self.default_branch,
            "auth_source": self.auth_source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProjectGitConfig":
        return cls(
            remote_url=str(raw.get("remote_url", "")),
            default_branch=str(raw.get("default_branch", "main")),
            auth_source=str(raw.get("auth_source", "system")),
        )


@dataclass
class Settings:
    experiment_max_iterations: int = 3
    eval_max_iterations: int = 3
    view_summary_page_size: int = 10
    autopilot_enabled: bool = False
    autopilot_max_consecutive_trials: int = 10
    autopilot_max_consecutive_failures: int = 3
    research_cadence: str = "ask"  # ask | disabled | hourly | 6h | daily
    git_user_name: str = ""
    git_user_email: str = ""
    git_auth_method: str = "system"  # system | token
    projects: dict[str, ProjectGitConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_max_iterations": self.experiment_max_iterations,
            "eval_max_iterations": self.eval_max_iterations,
            "view_summary_page_size": self.view_summary_page_size,
            "autopilot_enabled": self.autopilot_enabled,
            "autopilot_max_consecutive_trials": self.autopilot_max_consecutive_trials,
            "autopilot_max_consecutive_failures": self.autopilot_max_consecutive_failures,
            "research_cadence": self.research_cadence,
            "git_user_name": self.git_user_name,
            "git_user_email": self.git_user_email,
            "git_auth_method": self.git_auth_method,
            "projects": {k: v.to_dict() for k, v in self.projects.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Settings":
        projects_raw = raw.get("projects", {})
        projects: dict[str, ProjectGitConfig] = {}
        if isinstance(projects_raw, dict):
            for name, cfg in projects_raw.items():
                if isinstance(cfg, dict):
                    projects[name] = ProjectGitConfig.from_dict(cfg)

        return cls(
            experiment_max_iterations=int(raw.get("experiment_max_iterations", 3)),
            eval_max_iterations=int(raw.get("eval_max_iterations", 3)),
            view_summary_page_size=int(raw.get("view_summary_page_size", 10)),
            autopilot_enabled=bool(raw.get("autopilot_enabled", False)),
            autopilot_max_consecutive_trials=int(raw.get("autopilot_max_consecutive_trials", 10)),
            autopilot_max_consecutive_failures=int(raw.get("autopilot_max_consecutive_failures", 3)),
            research_cadence=str(raw.get("research_cadence", "ask")),
            git_user_name=str(raw.get("git_user_name", "")),
            git_user_email=str(raw.get("git_user_email", "")),
            git_auth_method=str(raw.get("git_auth_method", "system")),
            projects=projects,
        )


@dataclass
class TrialRecord:
    trial_id: str
    date: str
    trial_number: int
    state: State
    status: TrialStatus
    selected_project: str | None
    sandbox_path: str
    outputs_path: str
    results_path: str
    plan_approved: bool = False
    experiment_iter: int = 0
    eval_iter: int = 0
    terminated: bool = False
    termination_reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str | None = None

    @property
    def trial_name(self) -> str:
        return f"trial_{self.trial_number:03d}"

    @property
    def sandbox_root(self) -> Path:
        return Path(self.sandbox_path)

    @property
    def outputs_root(self) -> Path:
        return Path(self.outputs_path)

    @property
    def results_root(self) -> Path:
        return Path(self.results_path)

    def touch(self) -> None:
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "date": self.date,
            "trial_number": self.trial_number,
            "state": self.state.value,
            "status": self.status.value,
            "selected_project": self.selected_project,
            "sandbox_path": self.sandbox_path,
            "outputs_path": self.outputs_path,
            "results_path": self.results_path,
            "plan_approved": self.plan_approved,
            "experiment_iter": self.experiment_iter,
            "eval_iter": self.eval_iter,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrialRecord":
        return cls(
            trial_id=str(raw["trial_id"]),
            date=str(raw["date"]),
            trial_number=int(raw["trial_number"]),
            state=State(str(raw.get("state", State.PLAN.value))),
            status=TrialStatus(str(raw.get("status", TrialStatus.ACTIVE.value))),
            selected_project=raw.get("selected_project"),
            sandbox_path=str(raw["sandbox_path"]),
            outputs_path=str(raw["outputs_path"]),
            results_path=str(raw["results_path"]),
            plan_approved=bool(raw.get("plan_approved", False)),
            experiment_iter=int(raw.get("experiment_iter", 0)),
            eval_iter=int(raw.get("eval_iter", 0)),
            terminated=bool(raw.get("terminated", False)),
            termination_reason=str(raw.get("termination_reason", "")),
            created_at=str(raw.get("created_at", datetime.now().isoformat())),
            updated_at=str(raw.get("updated_at", datetime.now().isoformat())),
            finished_at=raw.get("finished_at"),
        )
