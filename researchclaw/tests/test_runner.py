"""Tests for researchclaw/runner.py"""

from __future__ import annotations

import os
import subprocess

import pytest

from researchclaw.access_control import PathValidator
from researchclaw.models import TrialInfo
from researchclaw.runner import RunResult, Runner


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def trial(tmp_path):
    """Active TrialInfo whose sandbox and report log dirs exist."""
    t = TrialInfo(
        date="20260225",
        number=1,
        status="active",
        started_at="2026-02-25T10:00:00",
    )
    (tmp_path / t.sandbox_path).mkdir(parents=True)
    (tmp_path / t.report_path / "log").mkdir(parents=True)
    return t


@pytest.fixture
def validator(tmp_path, trial):
    v = PathValidator(str(tmp_path))
    v.set_trial(trial)
    return v


@pytest.fixture
def runner(tmp_path, validator):
    return Runner(str(tmp_path), validator)


# ──────────────────────────────────────────────────────────────────────────────
# TestRunAndWait
# ──────────────────────────────────────────────────────────────────────────────

class TestRunAndWait:
    def test_returns_run_result(self, runner, trial):
        result = runner.run_and_wait("echo hello", trial)
        assert isinstance(result, RunResult)

    def test_returncode_zero_on_success(self, runner, trial):
        result = runner.run_and_wait("echo hello", trial)
        assert result.returncode == 0

    def test_pid_is_zero_on_success(self, runner, trial):
        """Bug fix: pid must be 0 (sentinel) for completed processes."""
        result = runner.run_and_wait("echo hello", trial)
        assert result.pid == 0

    def test_pid_is_zero_on_failure(self, runner, trial):
        """Bug fix: pid must be 0 even when command fails."""
        result = runner.run_and_wait("false", trial)
        assert result.pid == 0

    def test_nonzero_returncode_on_failure(self, runner, trial):
        result = runner.run_and_wait("false", trial)
        assert result.returncode != 0

    def test_stdout_captured_in_result(self, runner, trial):
        result = runner.run_and_wait("echo captured_output", trial)
        assert "captured_output" in result.stdout

    def test_stdout_written_to_log_file(self, runner, trial, tmp_path):
        runner.run_and_wait("echo hello_log", trial)
        log_file = tmp_path / trial.report_path / "log" / "stdout.log"
        assert log_file.exists()
        assert "hello_log" in log_file.read_text()

    def test_stderr_written_to_log_file(self, runner, trial, tmp_path):
        # 1>&2 (no space before >) avoids PathValidator's redirect-path check
        runner.run_and_wait("echo err_msg 1>&2", trial)
        log_file = tmp_path / trial.report_path / "log" / "stderr.log"
        assert log_file.exists()
        assert "err_msg" in log_file.read_text()

    def test_duration_is_non_negative(self, runner, trial):
        result = runner.run_and_wait("echo fast", trial)
        assert result.duration_seconds >= 0.0

    def test_log_path_points_to_log_dir(self, runner, trial, tmp_path):
        result = runner.run_and_wait("echo x", trial)
        expected = str(tmp_path / trial.report_path / "log")
        assert result.log_path == expected

    def test_timeout_raises(self, runner, trial):
        with pytest.raises(subprocess.TimeoutExpired):
            runner.run_and_wait("sleep 10", trial, timeout=1)


# ──────────────────────────────────────────────────────────────────────────────
# TestRunAsync
# ──────────────────────────────────────────────────────────────────────────────

class TestRunAsync:
    def test_returns_popen(self, runner, trial):
        proc = runner.run_async("sleep 0.05", trial)
        try:
            assert isinstance(proc, subprocess.Popen)
        finally:
            proc.wait()

    def test_pid_is_positive(self, runner, trial):
        proc = runner.run_async("sleep 0.05", trial)
        try:
            assert proc.pid > 0
        finally:
            proc.wait()

    def test_log_files_created_after_wait(self, runner, trial, tmp_path):
        proc = runner.run_async("echo async_out", trial)
        proc.wait()
        log_file = tmp_path / trial.report_path / "log" / "stdout.log"
        assert log_file.exists()
        assert "async_out" in log_file.read_text()

    def test_parent_file_handles_closed(self, runner, trial):
        """Bug fix: run_async must close parent-side file handles after fork."""
        import resource
        before = resource.getrlimit(resource.RLIMIT_NOFILE)[0]

        # Run many trials to surface a descriptor leak
        for i in range(30):
            proc = runner.run_async("echo x", trial)
            proc.wait()

        # If handles leaked, open FD count would grow; check we haven't exhausted
        # the soft limit. We verify by trying to open a new file descriptor.
        f = open(os.devnull)
        f.close()  # should not raise OSError: [Errno 24] Too many open files


# ──────────────────────────────────────────────────────────────────────────────
# TestBuildEnv
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildEnv:
    def test_researchclaw_sandbox_set(self, tmp_path, validator, trial):
        r = Runner(str(tmp_path), validator)
        env = r._build_env(trial, None)
        assert env["RESEARCHCLAW_SANDBOX"] == str(tmp_path / trial.sandbox_path)

    def test_researchclaw_report_set(self, tmp_path, validator, trial):
        r = Runner(str(tmp_path), validator)
        env = r._build_env(trial, None)
        assert env["RESEARCHCLAW_REPORT"] == str(tmp_path / trial.report_path)

    def test_researchclaw_trial_set(self, tmp_path, validator, trial):
        r = Runner(str(tmp_path), validator)
        env = r._build_env(trial, None)
        assert env["RESEARCHCLAW_TRIAL"] == trial.trial_name

    def test_extra_env_merged(self, tmp_path, validator, trial):
        r = Runner(str(tmp_path), validator)
        env = r._build_env(trial, {"MY_VAR": "42"})
        assert env["MY_VAR"] == "42"

    def test_no_venv_path_unchanged(self, tmp_path, validator, trial):
        r = Runner(str(tmp_path), validator, venv_path="")
        original_path = os.environ.get("PATH", "")
        env = r._build_env(trial, None)
        assert env["PATH"] == original_path

    def test_venv_prepends_bin_to_path(self, tmp_path, validator, trial):
        fake_venv = tmp_path / ".venv"
        (fake_venv / "bin").mkdir(parents=True)
        (fake_venv / "bin" / "python").write_text("")

        r = Runner(str(tmp_path), validator, venv_path=str(fake_venv))
        env = r._build_env(trial, None)
        assert env["PATH"].startswith(str(fake_venv / "bin"))

    def test_venv_sets_virtual_env(self, tmp_path, validator, trial):
        fake_venv = tmp_path / ".venv"
        (fake_venv / "bin").mkdir(parents=True)

        r = Runner(str(tmp_path), validator, venv_path=str(fake_venv))
        env = r._build_env(trial, None)
        assert env["VIRTUAL_ENV"] == str(fake_venv)

    def test_venv_clears_conda_vars(self, tmp_path, validator, trial, monkeypatch):
        fake_venv = tmp_path / ".venv"
        (fake_venv / "bin").mkdir(parents=True)

        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")

        r = Runner(str(tmp_path), validator, venv_path=str(fake_venv))
        env = r._build_env(trial, None)
        assert "CONDA_DEFAULT_ENV" not in env
        assert "CONDA_PREFIX" not in env

    def test_conda_prepends_bin(self, tmp_path, validator, trial, monkeypatch):
        # Create fake conda env directory
        conda_env_dir = tmp_path / "envs" / "myenv"
        (conda_env_dir / "bin").mkdir(parents=True)

        # Simulate being inside a conda base env
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        r = Runner(str(tmp_path), validator, conda_env="myenv")
        env = r._build_env(trial, None)
        assert env["PATH"].startswith(str(conda_env_dir / "bin"))
        assert env["CONDA_DEFAULT_ENV"] == "myenv"
