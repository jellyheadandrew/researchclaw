from __future__ import annotations

import filecmp
import shutil
import subprocess
from pathlib import Path

from .models import ProjectGitConfig, Settings, TrialRecord
from .policy import AuthorityManager, Operation
from .states import State


class GitManager:
    def __init__(self, base_dir: str, authority: AuthorityManager):
        self.base_dir = Path(base_dir).resolve()
        self.authority = authority

    def add_project_clone(
        self,
        state: State,
        project_name: str,
        remote_url: str,
        branch: str,
    ) -> str:
        self.authority.assert_operation(state, Operation.GIT_MUTATE)
        projects_root = self.base_dir / "projects"
        projects_root.mkdir(parents=True, exist_ok=True)
        target = projects_root / project_name
        if target.exists():
            raise FileExistsError(f"project already exists: {project_name}")

        result = subprocess.run(
            ["git", "clone", remote_url, str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

        if branch:
            self._run_git(target, "checkout", branch)
        return f"cloned {project_name}"

    def add_project_init(
        self,
        state: State,
        project_name: str,
        remote_url: str,
        branch: str,
    ) -> str:
        self.authority.assert_operation(state, Operation.GIT_MUTATE)
        target = self.base_dir / "projects" / project_name
        if target.exists():
            raise FileExistsError(f"project already exists: {project_name}")
        target.mkdir(parents=True, exist_ok=True)

        self._run_git(target, "init")
        self._ensure_identity(target)
        if branch:
            self._run_git(target, "checkout", "-b", branch)
        if remote_url:
            self._run_git(target, "remote", "add", "origin", remote_url)

        readme = target / "README.md"
        if not readme.exists():
            readme.write_text(f"# {project_name}\n", encoding="utf-8")
        self._run_git(target, "add", "-A")
        self._run_git(target, "commit", "-m", "Initialize project scaffold")
        return f"initialized {project_name}"

    def assimilate_trial_codes(
        self,
        state: State,
        trial: TrialRecord,
        project_name: str,
    ) -> list[str]:
        self.authority.assert_operation(state, Operation.GIT_MUTATE)

        project_root = self.base_dir / "projects" / project_name
        trial_codes = self.base_dir / trial.sandbox_path / "codes"
        if not project_root.exists():
            raise FileNotFoundError(f"project not found: {project_name}")
        if not trial_codes.exists():
            raise FileNotFoundError(f"trial codes missing: {trial_codes}")

        copied: list[str] = []
        for src in trial_codes.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(trial_codes)
            dst = project_root / rel
            self.authority.validate_write_path(
                state,
                dst,
                trial=None,
                selected_project=project_name,
            )
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(rel))

        return copied

    def commit_and_push(
        self,
        state: State,
        project_name: str,
        cfg: ProjectGitConfig,
        message: str,
    ) -> str:
        self.authority.assert_operation(state, Operation.GIT_MUTATE)
        project_root = self.base_dir / "projects" / project_name
        if not project_root.exists():
            raise FileNotFoundError(f"project not found: {project_name}")

        # Keep remote URL aligned with settings when provided.
        if cfg.remote_url:
            remotes = self._run_git(project_root, "remote").stdout.strip().splitlines()
            if "origin" not in remotes:
                self._run_git(project_root, "remote", "add", "origin", cfg.remote_url)
            else:
                self._run_git(project_root, "remote", "set-url", "origin", cfg.remote_url)

        if cfg.default_branch:
            self._run_git(project_root, "checkout", "-B", cfg.default_branch)

        self._ensure_identity(project_root)
        self._run_git(project_root, "add", "-A")
        commit = subprocess.run(
            ["git", "-C", str(project_root), "commit", "-m", message],
            capture_output=True,
            text=True,
        )
        if commit.returncode != 0:
            combined = (commit.stderr or commit.stdout).strip().lower()
            if "nothing to commit" in combined:
                # no-op commit is acceptable in assimilation flow
                return "no changes to commit"
            raise RuntimeError(f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}")

        push = subprocess.run(
            ["git", "-C", str(project_root), "push", "-u", "origin", cfg.default_branch or "main"],
            capture_output=True,
            text=True,
        )
        if push.returncode != 0:
            raise RuntimeError(f"git push failed: {push.stderr.strip() or push.stdout.strip()}")
        return push.stdout.strip() or "pushed"

    @staticmethod
    def _run_git(project_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
        return result

    def _ensure_identity(self, project_root: Path) -> None:
        email = subprocess.run(
            ["git", "-C", str(project_root), "config", "--get", "user.email"],
            capture_output=True,
            text=True,
        )
        name = subprocess.run(
            ["git", "-C", str(project_root), "config", "--get", "user.name"],
            capture_output=True,
            text=True,
        )

        if email.returncode != 0 or not email.stdout.strip():
            self._run_git(project_root, "config", "user.email", "researchclaw@local")
        if name.returncode != 0 or not name.stdout.strip():
            self._run_git(project_root, "config", "user.name", "ResearchClaw")


def ensure_project_config(settings: Settings, project_name: str) -> ProjectGitConfig:
    if project_name not in settings.projects:
        settings.projects[project_name] = ProjectGitConfig()
    return settings.projects[project_name]
