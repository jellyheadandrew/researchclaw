from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.evaluate import (
    _run_eval_subprocess,
    _save_eval_output_log,
    handle_eval_execute,
)
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput
from researchclaw.sandbox import SandboxManager


# --- Fake helpers ---


class FakeChat:
    """Fake chat interface with pre-programmed responses."""

    def __init__(self, responses: list[ChatInput] | None = None) -> None:
        self.responses = list(responses) if responses else []
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def send_image(self, path: str, caption: str | None = None) -> None:
        pass

    def receive(self) -> ChatInput:
        if not self.responses:
            raise SystemExit("No more responses")
        return self.responses.pop(0)


def _setup_sandbox(project_dir: Path) -> Path:
    """Initialize sandbox and create a trial, return trial dir."""
    SandboxManager.initialize(project_dir)
    return SandboxManager.create_trial(project_dir)


def _create_run_eval_sh(trial_dir: Path, script_body: str = "") -> Path:
    """Create a run_eval.sh script in the trial directory."""
    script_path = trial_dir / "experiment" / "run_eval.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_content = f"#!/usr/bin/env bash\n{script_body}\n"
    script_path.write_text(script_content)
    script_path.chmod(
        script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    return script_path


# --- Tests for _save_eval_output_log ---


class TestSaveEvalOutputLog:
    def test_creates_log_file(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "outputs").mkdir(parents=True)
        log_path = _save_eval_output_log(trial_dir, 0, "some eval output")
        assert log_path.exists()
        assert log_path.name == "eval_log_iter000"

    def test_log_content(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "outputs").mkdir(parents=True)
        log_path = _save_eval_output_log(trial_dir, 0, "eval line1\neval line2\n")
        assert log_path.read_text() == "eval line1\neval line2\n"

    def test_retry_numbering(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "outputs").mkdir(parents=True)
        log0 = _save_eval_output_log(trial_dir, 0, "attempt 0")
        log1 = _save_eval_output_log(trial_dir, 1, "attempt 1")
        log2 = _save_eval_output_log(trial_dir, 2, "attempt 2")
        assert log0.name == "eval_log_iter000"
        assert log1.name == "eval_log_iter001"
        assert log2.name == "eval_log_iter002"

    def test_creates_outputs_dir_if_missing(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        log_path = _save_eval_output_log(trial_dir, 0, "output")
        assert log_path.exists()
        assert (trial_dir / "experiment" / "outputs").is_dir()


# --- Tests for _run_eval_subprocess ---


class TestRunEvalSubprocess:
    def test_success_returns_zero(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "eval done"')
        config = ResearchClawConfig()
        exit_code, output = _run_eval_subprocess(trial_dir, config, None)
        assert exit_code == 0
        assert "eval done" in output

    def test_failure_returns_nonzero(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 42")
        config = ResearchClawConfig()
        exit_code, output = _run_eval_subprocess(trial_dir, config, None)
        assert exit_code == 42

    def test_missing_script_returns_error(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        config = ResearchClawConfig()
        exit_code, output = _run_eval_subprocess(trial_dir, config, None)
        assert exit_code == 1
        assert "not found" in output

    def test_streams_output_to_chat(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "eval1"\necho "eval2"')
        config = ResearchClawConfig()
        chat = FakeChat()
        exit_code, output = _run_eval_subprocess(trial_dir, config, chat)
        assert exit_code == 0
        assert any("eval1" in m for m in chat.sent)
        assert any("eval2" in m for m in chat.sent)

    def test_captures_stderr(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "eval stderr" >&2')
        config = ResearchClawConfig()
        exit_code, output = _run_eval_subprocess(trial_dir, config, None)
        assert "eval stderr" in output

    def test_timeout_kills_process(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "sleep 60")
        config = ResearchClawConfig(experiment_timeout_seconds=1)
        exit_code, output = _run_eval_subprocess(trial_dir, config, None)
        assert exit_code == 1
        assert "TIMEOUT" in output

    def test_none_chat_still_works(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "ok"')
        config = ResearchClawConfig()
        exit_code, output = _run_eval_subprocess(trial_dir, config, None)
        assert exit_code == 0
        assert "ok" in output


# --- Tests for handle_eval_execute ---


class TestHandleEvalExecute:
    def test_success_returns_experiment_report(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "eval done"')
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChat()
        result = handle_eval_execute(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_REPORT

    def test_success_saves_exit_code_zero(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "ok"')
        meta = TrialMeta()
        config = ResearchClawConfig()
        result = handle_eval_execute(trial_dir, meta, config, None)
        assert meta.eval_exit_code == 0

    def test_failure_returns_eval_implement(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")
        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChat()
        result = handle_eval_execute(trial_dir, meta, config, chat)
        assert result == State.EVAL_IMPLEMENT

    def test_failure_increments_retry_count(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")
        meta = TrialMeta(eval_retry_count=0)
        config = ResearchClawConfig(max_retries=5)
        handle_eval_execute(trial_dir, meta, config, None)
        assert meta.eval_retry_count == 1

    def test_max_retries_returns_experiment_report(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")
        meta = TrialMeta(eval_retry_count=4)
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChat()
        result = handle_eval_execute(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_REPORT

    def test_max_retries_message(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")
        meta = TrialMeta(eval_retry_count=4)
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChat()
        handle_eval_execute(trial_dir, meta, config, chat)
        assert any("Max retries" in m for m in chat.sent)

    def test_creates_eval_output_log(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "eval output"')
        meta = TrialMeta()
        config = ResearchClawConfig()
        handle_eval_execute(trial_dir, meta, config, None)
        log = trial_dir / "experiment" / "outputs" / "eval_log_iter000"
        assert log.exists()
        assert "eval output" in log.read_text()

    def test_retry_creates_sequential_logs(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")
        meta = TrialMeta(eval_retry_count=2)
        config = ResearchClawConfig(max_retries=5)
        handle_eval_execute(trial_dir, meta, config, None)
        assert (trial_dir / "experiment" / "outputs" / "eval_log_iter002").exists()

    def test_sends_status_messages(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "ok"')
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChat()
        handle_eval_execute(trial_dir, meta, config, chat)
        assert any("EVAL_EXECUTE" in m for m in chat.sent)
        assert any("successfully" in m for m in chat.sent)

    def test_none_chat_interface(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "ok"')
        meta = TrialMeta()
        config = ResearchClawConfig()
        result = handle_eval_execute(trial_dir, meta, config, None)
        assert result == State.EXPERIMENT_REPORT

    def test_failure_saves_exit_code(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 3")
        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=5)
        handle_eval_execute(trial_dir, meta, config, None)
        assert meta.eval_exit_code == 3

    def test_retry_message_on_failure(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")
        meta = TrialMeta(eval_retry_count=1)
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChat()
        handle_eval_execute(trial_dir, meta, config, chat)
        assert any("Retrying" in m for m in chat.sent)

    def test_max_retries_one(self, tmp_path: Path) -> None:
        """With max_retries=1, first failure should skip to report."""
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")
        meta = TrialMeta(eval_retry_count=0)
        config = ResearchClawConfig(max_retries=1)
        result = handle_eval_execute(trial_dir, meta, config, None)
        assert result == State.EXPERIMENT_REPORT

    def test_timeout_returns_implement_for_retry(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, "sleep 60")
        meta = TrialMeta(eval_retry_count=0)
        config = ResearchClawConfig(max_retries=5, experiment_timeout_seconds=1)
        result = handle_eval_execute(trial_dir, meta, config, None)
        assert result == State.EVAL_IMPLEMENT
        assert meta.eval_exit_code == 1

    def test_attempt_number_in_message(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        _create_run_eval_sh(trial_dir, 'echo "ok"')
        meta = TrialMeta(eval_retry_count=2)
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChat()
        handle_eval_execute(trial_dir, meta, config, chat)
        assert any("attempt 3/5" in m for m in chat.sent)


# --- Tests for TrialMeta eval_retry_count ---


class TestTrialMetaEvalRetryCount:
    def test_default_eval_retry_count_is_zero(self) -> None:
        meta = TrialMeta()
        assert meta.eval_retry_count == 0

    def test_eval_retry_count_round_trip(self, tmp_path: Path) -> None:
        meta = TrialMeta(eval_retry_count=3)
        path = tmp_path / "meta.json"
        meta.to_json(path)
        loaded = TrialMeta.from_json(path)
        assert loaded.eval_retry_count == 3
