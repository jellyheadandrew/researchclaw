"""Tests for researchclaw/git_manager.py — uses a real git repo in tmp_path."""

from __future__ import annotations

import subprocess

import pytest

from researchclaw.git_manager import GitManager
from researchclaw.models import TrialInfo


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _git(*args, cwd):
    """Run a git command, fail loudly if it errors."""
    subprocess.run(["git"] + list(args), cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    """
    Set up tmp_path/github_codes/ as a real git repo with one committed file.
    Returns tmp_path (the ResearchClaw base_dir).
    """
    repo = tmp_path / "github_codes"
    repo.mkdir()
    _git("init", "-b", "main", str(repo), cwd=tmp_path)
    _git("-C", str(repo), "config", "user.email", "test@researchclaw.test", cwd=tmp_path)
    _git("-C", str(repo), "config", "user.name", "ResearchClaw Test", cwd=tmp_path)
    (repo / "train.py").write_text("# original training script")
    (repo / "model.py").write_text("# original model")
    _git("-C", str(repo), "add", "-A", cwd=tmp_path)
    _git("-C", str(repo), "commit", "-m", "initial commit", cwd=tmp_path)
    return tmp_path


@pytest.fixture
def mgr(git_repo):
    return GitManager(str(git_repo))


@pytest.fixture
def trial(git_repo):
    """Trial whose sandbox is a copy of github_codes (identical initially)."""
    t = TrialInfo(
        date="20260225",
        number=1,
        status="active",
        started_at="2026-02-25T10:00:00",
    )
    sandbox = git_repo / t.sandbox_path
    sandbox.mkdir(parents=True)
    (sandbox / "train.py").write_text("# original training script")
    (sandbox / "model.py").write_text("# original model")
    return t


# ──────────────────────────────────────────────────────────────────────────────
# TestCompareTrees
# ──────────────────────────────────────────────────────────────────────────────

class TestCompareTrees:
    def test_identical_trees_returns_empty_sets(self, git_repo, mgr, trial):
        changed, added, removed = mgr._compare_trees(
            mgr.repo_path, git_repo / trial.sandbox_path
        )
        assert changed == set()
        assert added == set()
        assert removed == set()

    def test_detects_modified_file(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "train.py").write_text("# MODIFIED")
        changed, added, removed = mgr._compare_trees(
            mgr.repo_path, git_repo / trial.sandbox_path
        )
        assert "train.py" in changed

    def test_detects_added_file(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "new_module.py").write_text("# new")
        changed, added, removed = mgr._compare_trees(
            mgr.repo_path, git_repo / trial.sandbox_path
        )
        assert "new_module.py" in added

    def test_detects_removed_file(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "model.py").unlink()
        changed, added, removed = mgr._compare_trees(
            mgr.repo_path, git_repo / trial.sandbox_path
        )
        assert "model.py" in removed

    def test_ignores_git_directory(self, git_repo, mgr, trial):
        # .git/ in sandbox should not appear in diff (it's excluded)
        (git_repo / trial.sandbox_path / ".git").mkdir(exist_ok=True)
        (git_repo / trial.sandbox_path / ".git" / "config").write_text("[core]")
        changed, added, removed = mgr._compare_trees(
            mgr.repo_path, git_repo / trial.sandbox_path
        )
        all_files = changed | added | removed
        assert not any(".git" in f for f in all_files)

    def test_ignores_pyc_files(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "train.pyc").write_bytes(b"\x00pyc")
        changed, added, removed = mgr._compare_trees(
            mgr.repo_path, git_repo / trial.sandbox_path
        )
        all_files = changed | added | removed
        assert not any(".pyc" in f for f in all_files)


# ──────────────────────────────────────────────────────────────────────────────
# TestGetDiff
# ──────────────────────────────────────────────────────────────────────────────

class TestGetDiff:
    def test_no_changes_returns_no_differences(self, git_repo, mgr, trial):
        result = mgr.get_diff(trial)
        assert "(no differences found)" in result

    def test_shows_modified_files(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "train.py").write_text("# modified")
        result = mgr.get_diff(trial)
        assert "Modified files" in result
        assert "train.py" in result

    def test_shows_added_files(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "new.py").write_text("# new")
        result = mgr.get_diff(trial)
        assert "New files" in result
        assert "new.py" in result

    def test_sandbox_missing_returns_error_message(self, git_repo, mgr):
        t = TrialInfo(
            date="29991231", number=99, status="active", started_at="2099"
        )
        result = mgr.get_diff(t)
        assert "not found" in result


# ──────────────────────────────────────────────────────────────────────────────
# TestMergeTrial
# ──────────────────────────────────────────────────────────────────────────────

class TestMergeTrial:
    def test_merge_requires_authorization(self, git_repo, mgr, trial):
        with pytest.raises(PermissionError):
            mgr.merge_trial(trial, "test merge")

    def test_merge_copies_changed_files(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "train.py").write_text("# changed training")
        mgr.authorize_merge()
        mgr.merge_trial(trial, "test: update train.py")
        assert (git_repo / "github_codes" / "train.py").read_text() == "# changed training"

    def test_merge_creates_git_commit(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "train.py").write_text("# new content")
        mgr.authorize_merge()
        mgr.merge_trial(trial, "test: updated train")
        result = subprocess.run(
            ["git", "-C", str(git_repo / "github_codes"), "log", "--oneline"],
            capture_output=True, text=True,
        )
        assert "test: updated train" in result.stdout

    def test_merge_returns_commit_hash(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "train.py").write_text("# changed")
        mgr.authorize_merge()
        commit_hash = mgr.merge_trial(trial, "test commit")
        # git rev-parse --short HEAD returns a 7-character hex hash
        assert len(commit_hash) >= 7
        assert all(c in "0123456789abcdef" for c in commit_hash.lower())

    def test_authorization_consumed_after_merge(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "train.py").write_text("# changed")
        mgr.authorize_merge()
        mgr.merge_trial(trial, "first merge")
        with pytest.raises(PermissionError):
            mgr.merge_trial(trial, "second merge")

    def test_no_changes_returns_no_changes_message(self, git_repo, mgr, trial):
        # Sandbox is identical to github_codes
        mgr.authorize_merge()
        result = mgr.merge_trial(trial, "no-op merge")
        assert "(no changes to commit)" in result

    def test_merge_adds_new_file(self, git_repo, mgr, trial):
        (git_repo / trial.sandbox_path / "scheduler.py").write_text("# new scheduler")
        mgr.authorize_merge()
        mgr.merge_trial(trial, "add scheduler")
        assert (git_repo / "github_codes" / "scheduler.py").exists()


# ──────────────────────────────────────────────────────────────────────────────
# TestStatus
# ──────────────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_contains_branch(self, git_repo, mgr):
        result = mgr.status()
        assert "Branch:" in result

    def test_status_contains_last_commit(self, git_repo, mgr):
        result = mgr.status()
        assert "initial commit" in result

    def test_status_contains_clean_or_status(self, git_repo, mgr):
        result = mgr.status()
        # Either "clean" (no changes) or some status output
        assert "Status:" in result
