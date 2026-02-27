"""Tests for researchclaw/env_manager.py"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from researchclaw.env_manager import EnvManager


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def project_root(tmp_path):
    """A temporary project root with an envs/ parent directory."""
    return tmp_path


@pytest.fixture
def mgr(project_root):
    """EnvManager pointed at tmp_path, no ledger yet."""
    return EnvManager(str(project_root), backend="venv")


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_creates_env_000(self, mgr, project_root):
        env_path = mgr.bootstrap()
        assert env_path == project_root / "envs" / "env_000"
        assert env_path.exists()
        assert (env_path / "bin" / "python").exists() or (env_path / "bin" / "python3").exists()

    def test_creates_ledger(self, mgr, project_root):
        mgr.bootstrap()
        ledger = project_root / ".envs.jsonl"
        assert ledger.exists()
        entries = [json.loads(line) for line in ledger.read_text().strip().splitlines()]
        assert len(entries) == 1
        assert entries[0]["env_id"] == 0
        assert entries[0]["trigger"] == "bootstrap"
        assert entries[0]["parent_id"] is None

    def test_idempotent(self, mgr, project_root):
        mgr.bootstrap()
        mgr.bootstrap()  # second call should be a no-op

        ledger = project_root / ".envs.jsonl"
        entries = [json.loads(line) for line in ledger.read_text().strip().splitlines()]
        assert len(entries) == 1  # only one entry, not two

    def test_sets_current_env_id(self, mgr):
        mgr.bootstrap()
        assert mgr.current_env_id == 0


# ──────────────────────────────────────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────────────────────────────────────

class TestStatePersistence:
    def test_load_state_from_existing_ledger(self, project_root):
        # Write a ledger manually
        ledger = project_root / ".envs.jsonl"
        entries = [
            {"env_id": 0, "created_at": "2026-01-01T00:00:00", "parent_id": None, "backend": "venv", "trigger": "bootstrap", "trial": None},
            {"env_id": 1, "created_at": "2026-01-01T01:00:00", "parent_id": 0, "backend": "venv", "trigger": "pip install torch", "trial": "trial_001"},
        ]
        ledger.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        # New EnvManager should pick up the state
        mgr = EnvManager(str(project_root), backend="venv")
        assert mgr.current_env_id == 1

    def test_empty_ledger(self, project_root):
        ledger = project_root / ".envs.jsonl"
        ledger.write_text("")
        mgr = EnvManager(str(project_root), backend="venv")
        assert mgr.current_env_id == -1

    def test_no_ledger(self, project_root):
        mgr = EnvManager(str(project_root), backend="venv")
        assert mgr.current_env_id == -1


# ──────────────────────────────────────────────────────────────────────────────
# Mutation detection
# ──────────────────────────────────────────────────────────────────────────────

class TestIsMutation:
    @pytest.mark.parametrize("cmd", [
        "pip install torch",
        "pip install -r requirements.txt",
        "pip install --upgrade numpy",
        "pip install -e .",
        "pip install -e .[dev]",
        "pip3 install flask",
        "python -m pip install requests",
        "python3 -m pip install transformers",
        "python3.12 -m pip install scipy",
        "pip uninstall torch",
        "conda install pytorch",
        "conda install -y numpy",
        "mamba install pandas",
        "pip install torch && python train.py",
        "pip update torch",
        "pip upgrade torch",
        "conda remove pytorch",
    ])
    def test_positive(self, mgr, cmd):
        assert mgr.is_env_mutation(cmd), f"Expected True for: {cmd}"

    @pytest.mark.parametrize("cmd", [
        "python train.py",
        "python -c 'import torch; print(torch.__version__)'",
        "pip --version",
        "pip list",
        "pip freeze",
        "pip show torch",
        "conda list",
        "conda info",
        "echo pip install torch",
        "git status",
        "ls -la",
    ])
    def test_negative(self, mgr, cmd):
        assert not mgr.is_env_mutation(cmd), f"Expected False for: {cmd}"


# ──────────────────────────────────────────────────────────────────────────────
# Active env path
# ──────────────────────────────────────────────────────────────────────────────

class TestGetActiveEnvPath:
    def test_returns_correct_path(self, mgr, project_root):
        mgr.bootstrap()
        assert mgr.get_active_env_path() == project_root / "envs" / "env_000"

    def test_raises_if_not_bootstrapped(self, project_root):
        mgr = EnvManager(str(project_root), backend="venv")
        with pytest.raises(RuntimeError, match="No active environment"):
            mgr.get_active_env_path()


# ──────────────────────────────────────────────────────────────────────────────
# Fork env
# ──────────────────────────────────────────────────────────────────────────────

class TestForkEnv:
    def test_creates_new_env_dir(self, mgr, project_root):
        mgr.bootstrap()
        new_path = mgr.fork_env("pip install requests", trial_name="trial_001")
        assert new_path == project_root / "envs" / "env_001"
        assert new_path.exists()
        assert (new_path / "bin" / "python").exists() or (new_path / "bin" / "python3").exists()

    def test_increments_env_id(self, mgr):
        mgr.bootstrap()
        assert mgr.current_env_id == 0
        mgr.fork_env("test fork")
        assert mgr.current_env_id == 1

    def test_ledger_records_fork(self, mgr, project_root):
        mgr.bootstrap()
        mgr.fork_env("pip install torch", trial_name="trial_002")

        ledger = project_root / ".envs.jsonl"
        entries = [json.loads(line) for line in ledger.read_text().strip().splitlines()]
        assert len(entries) == 2
        assert entries[1]["env_id"] == 1
        assert entries[1]["parent_id"] == 0
        assert entries[1]["trigger"] == "pip install torch"
        assert entries[1]["trial"] == "trial_002"

    def test_fork_existing_raises(self, mgr, project_root):
        mgr.bootstrap()
        mgr.fork_env("first fork")
        # Manually create env_002 to simulate collision
        (project_root / "envs" / "env_002").mkdir(parents=True)
        with pytest.raises(FileExistsError):
            mgr.fork_env("second fork")

    def test_multiple_forks(self, mgr, project_root):
        mgr.bootstrap()
        mgr.fork_env("fork 1")
        mgr.fork_env("fork 2")
        assert mgr.current_env_id == 2
        assert (project_root / "envs" / "env_002").exists()
        assert mgr.get_active_env_path() == project_root / "envs" / "env_002"


# ──────────────────────────────────────────────────────────────────────────────
# Apply mutation (integration — actually runs pip in the new env)
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyMutation:
    @pytest.mark.slow
    def test_install_package(self, mgr, project_root):
        """Install a small pure-Python package and verify it's importable."""
        mgr.bootstrap()
        new_path = mgr.apply_mutation("pip install six", trial_name="trial_001")

        assert mgr.current_env_id == 1
        assert new_path == project_root / "envs" / "env_001"

        # Verify 'six' is importable in the new env
        python = new_path / "bin" / "python"
        result = subprocess.run(
            [str(python), "-c", "import six; print(six.__version__)"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

    @pytest.mark.slow
    def test_failed_install_raises(self, mgr):
        """Installing a non-existent package should raise RuntimeError."""
        mgr.bootstrap()
        with pytest.raises(RuntimeError, match="Environment command failed"):
            mgr.apply_mutation("pip install this-package-does-not-exist-zzz-999")
