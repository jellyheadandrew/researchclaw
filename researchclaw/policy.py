from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .models import TrialRecord
from .states import State


class Operation(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    GIT_MUTATE = "git_mutate"
    SETTINGS_MUTATE = "settings_mutate"
    NETWORK = "network"


class AuthorityError(PermissionError):
    pass


@dataclass(frozen=True)
class StatePolicy:
    allow_read: bool
    allow_write: bool
    allow_execute: bool
    allow_git_mutate: bool
    allow_settings_mutate: bool
    allow_network: bool


STATE_POLICIES: dict[State, StatePolicy] = {
    State.DECIDE: StatePolicy(True, False, False, False, False, False),
    State.PLAN: StatePolicy(True, True, False, False, False, True),
    State.EXPERIMENT_IMPLEMENT: StatePolicy(True, True, False, False, False, False),
    State.EXPERIMENT_EXECUTE: StatePolicy(True, True, True, False, False, False),
    State.EVAL_IMPLEMENT: StatePolicy(True, True, False, False, False, False),
    State.EVAL_EXECUTE: StatePolicy(True, True, True, False, False, False),
    State.REPORT_SUMMARY: StatePolicy(True, True, False, False, False, False),
    State.VIEW_SUMMARY: StatePolicy(True, False, False, False, False, False),
    State.UPDATE_AND_PUSH: StatePolicy(True, True, True, True, False, True),
    State.SETTINGS: StatePolicy(True, True, False, False, True, False),
    State.RESEARCH: StatePolicy(True, True, True, False, False, True),
}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class AuthorityManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        else:
            p = p.resolve()
        return p

    def assert_operation(self, state: State, operation: Operation) -> None:
        pol = STATE_POLICIES[state]
        allowed = {
            Operation.READ: pol.allow_read,
            Operation.WRITE: pol.allow_write,
            Operation.EXECUTE: pol.allow_execute,
            Operation.GIT_MUTATE: pol.allow_git_mutate,
            Operation.SETTINGS_MUTATE: pol.allow_settings_mutate,
            Operation.NETWORK: pol.allow_network,
        }[operation]
        if not allowed:
            raise AuthorityError(f"{operation.value} denied in state {state.value}")

    def validate_read_path(self, state: State, path: str | Path, trial: TrialRecord | None) -> Path:
        self.assert_operation(state, Operation.READ)
        p = self._resolve(path)
        if not _is_relative_to(p, self.base_dir):
            raise AuthorityError(f"read denied outside base_dir: {p}")

        # Restrictive states: DECIDE and VIEW_SUMMARY are summary-only.
        if state in {State.DECIDE, State.VIEW_SUMMARY}:
            allowed_roots = [
                self.base_dir / "results",
                self.base_dir / "EXPERIMENT_LOGS.md",
                self.base_dir / "trials.jsonl",
                self.base_dir / "settings.json",
            ]
            if not self._path_matches_any(p, allowed_roots):
                raise AuthorityError(f"read denied in {state.value}: {p}")

        # EXPERIMENT_EXECUTE: only trial sandbox
        elif state == State.EXPERIMENT_EXECUTE:
            if not trial:
                raise AuthorityError("read denied in EXPERIMENT_EXECUTE: no active trial")
            allowed_roots = [self.base_dir / trial.sandbox_path]
            if not self._path_matches_any(p, allowed_roots):
                raise AuthorityError(f"read denied in {state.value}: {p}")

        # EVAL_EXECUTE: outputs + eval_codes + eval.sh + results
        elif state == State.EVAL_EXECUTE:
            if not trial:
                raise AuthorityError("read denied in EVAL_EXECUTE: no active trial")
            trial_sandbox = self.base_dir / trial.sandbox_path
            allowed_roots = [
                self.base_dir / trial.outputs_path,
                trial_sandbox / "eval_codes",
                self.base_dir / trial.results_path,
            ]
            allowed_files = [trial_sandbox / "eval.sh"]
            if self._file_matches_any(p, allowed_files):
                return p
            if not self._path_matches_any(p, allowed_roots):
                raise AuthorityError(f"read denied in {state.value}: {p}")

        # REPORT_SUMMARY: trial sandbox + results + EXPERIMENT_LOGS.md + prior results
        elif state == State.REPORT_SUMMARY:
            if not trial:
                raise AuthorityError("read denied in REPORT_SUMMARY: no active trial")
            allowed_roots = [
                self.base_dir / trial.sandbox_path,
                self.base_dir / "results",
            ]
            allowed_files = [self.base_dir / "EXPERIMENT_LOGS.md"]
            if self._file_matches_any(p, allowed_files):
                return p
            if not self._path_matches_any(p, allowed_roots):
                raise AuthorityError(f"read denied in {state.value}: {p}")

        # RESEARCH: references/ + EXPERIMENT_LOGS.md + results/
        elif state == State.RESEARCH:
            allowed_roots = [
                self.base_dir / "references",
                self.base_dir / "results",
            ]
            allowed_files = [self.base_dir / "EXPERIMENT_LOGS.md"]
            if self._file_matches_any(p, allowed_files):
                return p
            if not self._path_matches_any(p, allowed_roots):
                raise AuthorityError(f"read denied in {state.value}: {p}")

        return p

    def validate_write_path(
        self,
        state: State,
        path: str | Path,
        trial: TrialRecord | None,
        selected_project: str | None = None,
    ) -> Path:
        self.assert_operation(state, Operation.WRITE)
        p = self._resolve(path)
        if not _is_relative_to(p, self.base_dir):
            raise AuthorityError(f"write denied outside base_dir: {p}")

        allowed_roots: list[Path] = []
        allowed_files: list[Path] = []

        if state == State.PLAN:
            if not trial:
                raise AuthorityError("write denied in PLAN: no active trial")
            allowed_files = [self.base_dir / trial.sandbox_path / "PLAN.md"]

        elif state == State.EXPERIMENT_IMPLEMENT:
            if not trial:
                raise AuthorityError("write denied in EXPERIMENT_IMPLEMENT: no active trial")
            trial_root = self.base_dir / trial.sandbox_path
            allowed_roots = [trial_root / "codes"]
            allowed_files = [trial_root / "run.sh"]

        elif state == State.EXPERIMENT_EXECUTE:
            if not trial:
                raise AuthorityError("write denied in EXPERIMENT_EXECUTE: no active trial")
            allowed_roots = [self.base_dir / trial.outputs_path]

        elif state == State.EVAL_IMPLEMENT:
            if not trial:
                raise AuthorityError("write denied in EVAL_IMPLEMENT: no active trial")
            trial_root = self.base_dir / trial.sandbox_path
            allowed_roots = [trial_root / "eval_codes"]
            allowed_files = [trial_root / "eval.sh"]

        elif state == State.EVAL_EXECUTE:
            if not trial:
                raise AuthorityError("write denied in EVAL_EXECUTE: no active trial")
            allowed_roots = [self.base_dir / trial.results_path]

        elif state == State.REPORT_SUMMARY:
            if not trial:
                raise AuthorityError("write denied in REPORT_SUMMARY: no active trial")
            allowed_roots = [self.base_dir / trial.results_path]
            allowed_files = [self.base_dir / "EXPERIMENT_LOGS.md"]

        elif state == State.UPDATE_AND_PUSH:
            if not selected_project:
                raise AuthorityError("write denied in UPDATE_AND_PUSH: no selected project")
            allowed_roots = [self.base_dir / "projects" / selected_project]

        elif state == State.SETTINGS:
            allowed_files = [
                self.base_dir / "settings.json",
                self.base_dir / "session_state.json",
                self.base_dir / "config.yaml",
            ]

        elif state == State.RESEARCH:
            allowed_roots = [self.base_dir / "references"]

        if p in allowed_files:
            return p
        if self._path_matches_any(p, allowed_roots):
            return p

        raise AuthorityError(f"write denied in {state.value}: {p}")

    def validate_execute_path(self, state: State, path: str | Path, trial: TrialRecord | None) -> Path:
        self.assert_operation(state, Operation.EXECUTE)
        p = self._resolve(path)
        if not _is_relative_to(p, self.base_dir):
            raise AuthorityError(f"execute denied outside base_dir: {p}")

        if state == State.EXPERIMENT_EXECUTE:
            if not trial:
                raise AuthorityError("execute denied: no trial")
            expected = (self.base_dir / trial.sandbox_path / "run.sh").resolve()
            if p != expected:
                raise AuthorityError(f"execute denied: only run.sh allowed ({expected})")

        elif state == State.EVAL_EXECUTE:
            if not trial:
                raise AuthorityError("execute denied: no trial")
            expected = (self.base_dir / trial.sandbox_path / "eval.sh").resolve()
            if p != expected:
                raise AuthorityError(f"execute denied: only eval.sh allowed ({expected})")

        return p

    def assert_git_mutate(self, state: State) -> None:
        self.assert_operation(state, Operation.GIT_MUTATE)

    def assert_settings_mutate(self, state: State) -> None:
        self.assert_operation(state, Operation.SETTINGS_MUTATE)

    def assert_network(self, state: State) -> None:
        self.assert_operation(state, Operation.NETWORK)

    @staticmethod
    def _path_matches_any(p: Path, roots: list[Path]) -> bool:
        for root in roots:
            if _is_relative_to(p, root.resolve()):
                return True
        return False

    @staticmethod
    def _file_matches_any(p: Path, files: list[Path]) -> bool:
        resolved = p.resolve() if not p.is_absolute() else p
        for f in files:
            if resolved == f.resolve():
                return True
        return False
