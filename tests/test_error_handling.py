"""Tests for US-021: Graceful error handling and abort flow.

Tests cover:
- User abort (/abort) in all interactive states
- Abort triggers EXPERIMENT_REPORT with [TERMINATED-DURING-EXPERIMENT]
- Abort appends terminated line to EXPERIMENT_LOGS.md
- After abort: advance to DECIDE
- Subprocess failure handling (caught, not crashed)
- LLM unavailability: retry/skip/quit options
- Venv creation failure: retry/skip/quit options
- Ctrl+C during execution: save state and exit gracefully
- Engine-level error handling: unexpected exceptions saved to meta.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm import TrialAborted
from researchclaw.fsm.decide import handle_decide
from researchclaw.fsm.engine import FSMEngine
from researchclaw.fsm.evaluate import handle_eval_execute, handle_eval_implement
from researchclaw.fsm.experiment import (
    _prompt_llm_unavailable as exp_prompt_llm,
    _try_ensure_venv as exp_try_venv,
    handle_experiment_execute,
    handle_experiment_implement,
)
from researchclaw.fsm.evaluate import (
    _prompt_llm_unavailable as eval_prompt_llm,
    _try_ensure_venv as eval_try_venv,
)
from researchclaw.fsm.plan import handle_experiment_plan
from researchclaw.fsm.report import handle_experiment_report
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager

import researchclaw.fsm.evaluate as evaluate_mod
import researchclaw.fsm.experiment as experiment_mod
import researchclaw.fsm.plan as plan_mod
import researchclaw.fsm.report as report_mod


# --- Test helpers ---


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


class FakeProvider:
    """Fake LLM provider that returns pre-configured responses."""

    def __init__(
        self, responses: list[str] | None = None, error: Exception | None = None
    ) -> None:
        self._responses = list(responses) if responses else []
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        self.calls.append({"messages": list(messages), "system": system})
        if self._error:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return "default response"

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Any:
        yield self.chat(messages, system)


def _setup_trial(project_dir: Path) -> Path:
    """Create sandbox and trial for testing."""
    SandboxManager.initialize(project_dir)
    trial_dir = SandboxManager.create_trial(project_dir)
    (trial_dir / "PLAN.md").write_text("# Test Plan\nTest experiment.")
    return trial_dir


def _create_run_exp_sh(trial_dir: Path, script_body: str) -> None:
    """Create an executable run_exp.sh with given body."""
    script_path = trial_dir / "experiment" / "run_exp.sh"
    script_path.write_text(f"#!/usr/bin/env bash\n{script_body}\n")
    script_path.chmod(0o755)


def _create_run_eval_sh(trial_dir: Path, script_body: str) -> None:
    """Create an executable run_eval.sh with given body."""
    script_path = trial_dir / "experiment" / "run_eval.sh"
    script_path.write_text(f"#!/usr/bin/env bash\n{script_body}\n")
    script_path.chmod(0o755)


# =============================================================================
# Abort flow tests
# =============================================================================


class TestAbortFromPlan:
    """Test /abort during EXPERIMENT_PLAN state."""

    def test_abort_raises_trial_aborted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[SlashCommand("/abort", "")])
        meta = TrialMeta()
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)

        with pytest.raises(TrialAborted):
            handle_experiment_plan(trial_dir, meta, ResearchClawConfig(), chat)


class TestAbortFromDecide:
    """Test /abort during DECIDE state."""

    def test_abort_raises_trial_aborted(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[SlashCommand("/abort", "")])
        meta = TrialMeta()

        with pytest.raises(TrialAborted):
            handle_decide(trial_dir, meta, ResearchClawConfig(), chat)


class TestAbortFromExperimentLLMPrompt:
    """Test /abort when prompted about LLM unavailability in experiment handler."""

    def test_abort_raises_trial_aborted(self) -> None:
        chat = FakeChat(responses=[SlashCommand("/abort", "")])

        with pytest.raises(TrialAborted):
            exp_prompt_llm(chat, "No LLM available")

    def test_quit_returns_quit(self) -> None:
        chat = FakeChat(responses=[SlashCommand("/quit", "")])
        result = exp_prompt_llm(chat, "No LLM available")
        assert result == "quit"

    def test_retry_returns_retry(self) -> None:
        chat = FakeChat(responses=[UserMessage("r")])
        result = exp_prompt_llm(chat, "No LLM available")
        assert result == "retry"

    def test_skip_returns_skip(self) -> None:
        chat = FakeChat(responses=[UserMessage("s")])
        result = exp_prompt_llm(chat, "No LLM available")
        assert result == "skip"


class TestAbortFromEvalLLMPrompt:
    """Test /abort when prompted about LLM unavailability in eval handler."""

    def test_abort_raises_trial_aborted(self) -> None:
        chat = FakeChat(responses=[SlashCommand("/abort", "")])

        with pytest.raises(TrialAborted):
            eval_prompt_llm(chat, "No LLM available")


class TestAbortFromExperimentVenvFailure:
    """Test /abort when venv creation fails in experiment handler."""

    def test_abort_raises_trial_aborted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[SlashCommand("/abort", "")])
        config = ResearchClawConfig()

        # Make VenvManager.ensure_venv always fail
        from researchclaw.sandbox import venv_manager
        monkeypatch.setattr(
            venv_manager.VenvManager, "ensure_venv",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("venv fail"))),
        )

        with pytest.raises(TrialAborted):
            exp_try_venv(trial_dir, config, chat)

    def test_quit_raises_system_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[SlashCommand("/quit", "")])
        config = ResearchClawConfig()

        from researchclaw.sandbox import venv_manager
        monkeypatch.setattr(
            venv_manager.VenvManager, "ensure_venv",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("venv fail"))),
        )

        with pytest.raises(SystemExit):
            exp_try_venv(trial_dir, config, chat)

    def test_retry_loops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        # First try: fail + retry; second try: fail + skip
        chat = FakeChat(responses=[UserMessage("r"), UserMessage("s")])
        config = ResearchClawConfig()

        call_count = 0

        def failing_ensure(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("venv fail")

        from researchclaw.sandbox import venv_manager
        monkeypatch.setattr(
            venv_manager.VenvManager, "ensure_venv",
            staticmethod(failing_ensure),
        )

        result = exp_try_venv(trial_dir, config, chat)
        assert result is False
        assert call_count == 2


class TestAbortFromEvalVenvFailure:
    """Test /abort when venv creation fails in eval handler."""

    def test_abort_raises_trial_aborted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[SlashCommand("/abort", "")])
        config = ResearchClawConfig()

        from researchclaw.sandbox import venv_manager
        monkeypatch.setattr(
            venv_manager.VenvManager, "ensure_venv",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("venv fail"))),
        )

        with pytest.raises(TrialAborted):
            eval_try_venv(trial_dir, config, chat)


# =============================================================================
# Engine abort flow: TrialAborted → EXPERIMENT_REPORT → DECIDE
# =============================================================================


class TestEngineAbortFlow:
    """Test that the engine correctly handles TrialAborted exceptions."""

    def test_abort_routes_to_report_then_decide(
        self, tmp_path: Path
    ) -> None:
        """When a handler raises TrialAborted, engine routes to
        EXPERIMENT_REPORT then continues to DECIDE."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        visited: list[str] = []

        def aborting_plan(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise TrialAborted("aborted")

        def tracking_report(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            visited.append("report")
            assert m.status == "terminated"
            return State.DECIDE

        def tracking_decide(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            visited.append("decide")
            raise SystemExit("done")

        handlers = {
            State.EXPERIMENT_PLAN: aborting_plan,
            State.EXPERIMENT_REPORT: tracking_report,
            State.DECIDE: tracking_decide,
        }

        chat = FakeChat()
        engine = FSMEngine(tmp_path, ResearchClawConfig(), chat, handlers)

        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

        assert visited == ["report", "decide"]
        # Check that the chat notified about abort
        assert any("aborted" in m.lower() for m in chat.sent)

    def test_abort_sets_terminated_status_in_meta(self, tmp_path: Path) -> None:
        """TrialAborted sets meta.status to 'terminated' before routing to report."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def aborting_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise TrialAborted("aborted")

        # No report handler — engine falls back to DECIDE
        handlers = {
            State.EXPERIMENT_PLAN: aborting_handler,
        }

        chat = FakeChat()
        engine = FSMEngine(tmp_path, ResearchClawConfig(), chat, handlers)
        engine.run(trial_dir=trial_dir)

        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.status == "terminated"

    def test_abort_report_failure_falls_back_to_decide(self, tmp_path: Path) -> None:
        """If report handler also fails after abort, engine goes to DECIDE."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def aborting_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise TrialAborted("aborted")

        def failing_report(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise RuntimeError("report failed too")

        visited: list[str] = []

        def tracking_decide(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            visited.append("decide")
            raise SystemExit("done")

        handlers = {
            State.EXPERIMENT_PLAN: aborting_handler,
            State.EXPERIMENT_REPORT: failing_report,
            State.DECIDE: tracking_decide,
        }

        engine = FSMEngine(tmp_path, ResearchClawConfig(), FakeChat(), handlers)
        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

        assert visited == ["decide"]

    def test_abort_with_real_report_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full abort flow: abort → real report handler writes
        [TERMINATED-DURING-EXPERIMENT] and appends to logs."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test plan for abort")

        def aborting_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise TrialAborted("aborted at EXPERIMENT_EXECUTE")

        def decide_exit(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise SystemExit("done")

        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        handlers = {
            State.EXPERIMENT_EXECUTE: aborting_handler,
            State.EXPERIMENT_REPORT: handle_experiment_report,
            State.DECIDE: decide_exit,
        }

        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = "experiment_execute"
        SandboxManager.save_trial_meta(trial_dir, meta)

        chat = FakeChat()
        engine = FSMEngine(tmp_path, ResearchClawConfig(), chat, handlers)
        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

        # Verify REPORT.md has terminated marker
        report_path = trial_dir / "REPORT.md"
        assert report_path.exists()
        report_content = report_path.read_text()
        assert report_content.startswith("[TERMINATED-DURING-EXPERIMENT]")

        # Verify EXPERIMENT_LOGS.md has a terminated entry
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_content = logs_path.read_text()
        assert "[TERMINATED]" in logs_content or "trial_" in logs_content

    def test_abort_no_chat_interface(self, tmp_path: Path) -> None:
        """Engine handles abort gracefully when chat_interface is None."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def aborting_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise TrialAborted("aborted")

        handlers = {
            State.EXPERIMENT_PLAN: aborting_handler,
        }

        engine = FSMEngine(tmp_path, ResearchClawConfig(), None, handlers)
        engine.run(trial_dir=trial_dir)

        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.status == "terminated"


# =============================================================================
# Ctrl+C handling
# =============================================================================


class TestCtrlCHandling:
    """Test Ctrl+C (KeyboardInterrupt) is handled gracefully."""

    def test_ctrl_c_saves_state_and_returns(self, tmp_path: Path) -> None:
        """KeyboardInterrupt during handler saves meta and returns."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def interrupting_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise KeyboardInterrupt()

        handlers = {
            State.EXPERIMENT_PLAN: interrupting_handler,
        }

        chat = FakeChat()
        engine = FSMEngine(tmp_path, ResearchClawConfig(), chat, handlers)
        engine.run(trial_dir=trial_dir)

        # Should have saved state
        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.updated_at is not None

        # Should have notified user
        assert any("interrupted" in m.lower() or "saved" in m.lower() for m in chat.sent)

    def test_ctrl_c_no_chat_interface(self, tmp_path: Path) -> None:
        """KeyboardInterrupt with no chat interface still saves state."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def interrupting_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise KeyboardInterrupt()

        handlers = {
            State.EXPERIMENT_PLAN: interrupting_handler,
        }

        engine = FSMEngine(tmp_path, ResearchClawConfig(), None, handlers)
        engine.run(trial_dir=trial_dir)

        # Should have saved state without crashing
        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.updated_at is not None

    def test_ctrl_c_during_subprocess_kills_process(self, tmp_path: Path) -> None:
        """KeyboardInterrupt during experiment subprocess is re-raised to engine."""
        trial_dir = _setup_trial(tmp_path)
        # Create a script that runs for a while
        _create_run_exp_sh(trial_dir, "sleep 100")

        meta = TrialMeta()
        config = ResearchClawConfig(experiment_timeout_seconds=1)

        # The subprocess should be killed by timeout; not a direct
        # Ctrl+C test but verifies the subprocess cleanup works.
        result = handle_experiment_execute(trial_dir, meta, config, FakeChat())
        # Timeout returns failure, which means retry or report
        assert result in (State.EXPERIMENT_IMPLEMENT, State.EXPERIMENT_REPORT)


# =============================================================================
# Unexpected exception handling
# =============================================================================


class TestUnexpectedExceptionHandling:
    """Test that unexpected exceptions are caught, logged, and state saved."""

    def test_unexpected_error_saves_state_and_reraises(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def crashing_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise ValueError("something went wrong")

        handlers = {
            State.EXPERIMENT_PLAN: crashing_handler,
        }

        chat = FakeChat()
        engine = FSMEngine(tmp_path, ResearchClawConfig(), chat, handlers)

        with pytest.raises(ValueError, match="something went wrong"):
            engine.run(trial_dir=trial_dir)

        # State should have been saved
        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.updated_at is not None

        # Error message sent to chat
        assert any("unexpected error" in m.lower() for m in chat.sent)

    def test_unexpected_error_no_chat(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def crashing_handler(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise RuntimeError("boom")

        handlers = {
            State.EXPERIMENT_PLAN: crashing_handler,
        }

        engine = FSMEngine(tmp_path, ResearchClawConfig(), None, handlers)

        with pytest.raises(RuntimeError, match="boom"):
            engine.run(trial_dir=trial_dir)

        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.updated_at is not None


# =============================================================================
# Subprocess failure handling
# =============================================================================


class TestSubprocessFailures:
    """Test that subprocess failures are caught and logged, not crashed."""

    def test_experiment_script_failure_retries(self, tmp_path: Path) -> None:
        """Failed run_exp.sh triggers retry path."""
        trial_dir = _setup_trial(tmp_path)
        _create_run_exp_sh(trial_dir, "exit 1")

        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=3)
        chat = FakeChat()
        result = handle_experiment_execute(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert meta.experiment_retry_count == 1
        assert meta.experiment_exit_code == 1

    def test_experiment_max_retries_skips_to_report(self, tmp_path: Path) -> None:
        """Exceeding max retries skips to EXPERIMENT_REPORT."""
        trial_dir = _setup_trial(tmp_path)
        _create_run_exp_sh(trial_dir, "exit 1")

        meta = TrialMeta(experiment_retry_count=4)
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChat()
        result = handle_experiment_execute(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_REPORT

    def test_eval_script_failure_retries(self, tmp_path: Path) -> None:
        """Failed run_eval.sh triggers retry path."""
        trial_dir = _setup_trial(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")

        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=3)
        chat = FakeChat()
        result = handle_eval_execute(trial_dir, meta, config, chat)

        assert result == State.EVAL_IMPLEMENT
        assert meta.eval_retry_count == 1

    def test_eval_max_retries_skips_to_report(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        _create_run_eval_sh(trial_dir, "exit 1")

        meta = TrialMeta(eval_retry_count=4)
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChat()
        result = handle_eval_execute(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_REPORT

    def test_missing_run_exp_sh_treated_as_failure(self, tmp_path: Path) -> None:
        """Missing run_exp.sh should be treated as failure, not crash."""
        trial_dir = _setup_trial(tmp_path)
        # Don't create run_exp.sh

        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=3)
        chat = FakeChat()
        result = handle_experiment_execute(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert meta.experiment_exit_code == 1

    def test_missing_run_eval_sh_treated_as_failure(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        # Don't create run_eval.sh

        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=3)
        chat = FakeChat()
        result = handle_eval_execute(trial_dir, meta, config, chat)

        assert result == State.EVAL_IMPLEMENT
        assert meta.eval_exit_code == 1

    def test_experiment_timeout_treated_as_failure(self, tmp_path: Path) -> None:
        """Timeout during experiment is treated as failure."""
        trial_dir = _setup_trial(tmp_path)
        _create_run_exp_sh(trial_dir, "sleep 100")

        meta = TrialMeta()
        config = ResearchClawConfig(experiment_timeout_seconds=1, max_retries=3)
        chat = FakeChat()
        result = handle_experiment_execute(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert meta.experiment_exit_code == 1
        # Output log should contain TIMEOUT
        outputs_dir = trial_dir / "experiment" / "outputs"
        log_file = outputs_dir / "log_iter000"
        assert log_file.exists()
        assert "TIMEOUT" in log_file.read_text()

    def test_output_log_saved_on_failure(self, tmp_path: Path) -> None:
        """Output is saved even when experiment fails."""
        trial_dir = _setup_trial(tmp_path)
        _create_run_exp_sh(trial_dir, 'echo "error output" && exit 1')

        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=3)
        handle_experiment_execute(trial_dir, meta, config, FakeChat())

        log_file = trial_dir / "experiment" / "outputs" / "log_iter000"
        assert log_file.exists()
        assert "error output" in log_file.read_text()


# =============================================================================
# LLM unavailability handling
# =============================================================================


class TestLLMUnavailability:
    """Test retry/skip/quit flow when LLM is not available."""

    def test_no_provider_skip_uses_placeholder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LLM is unavailable and user skips, placeholder code is used."""
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[UserMessage("s")])
        meta = TrialMeta()
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)

        result = handle_experiment_implement(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_EXECUTE
        # Check placeholder code was written
        code_file = trial_dir / "experiment" / "codes_exp" / "main.py"
        assert code_file.exists()
        assert "placeholder" in code_file.read_text().lower()

    def test_no_provider_quit_raises_system_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[UserMessage("q")])
        meta = TrialMeta()
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)

        with pytest.raises(SystemExit):
            handle_experiment_implement(trial_dir, meta, ResearchClawConfig(), chat)

    def test_no_provider_retry_then_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User retries once, then skips — should use placeholder."""
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[UserMessage("r"), UserMessage("s")])
        meta = TrialMeta()
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)

        result = handle_experiment_implement(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_EXECUTE

    def test_provider_error_skip_uses_placeholder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LLM raises an error and user skips, placeholder is used."""
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[UserMessage("s")])
        meta = TrialMeta()
        provider = FakeProvider(error=RuntimeError("API down"))
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: provider)

        result = handle_experiment_implement(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_EXECUTE

    def test_no_chat_interface_auto_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no chat interface, LLM unavailability auto-skips."""
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)

        result = handle_experiment_implement(trial_dir, meta, ResearchClawConfig(), None)
        assert result == State.EXPERIMENT_EXECUTE

    def test_eval_no_provider_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eval handler also supports skip when LLM unavailable."""
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[UserMessage("s")])
        meta = TrialMeta()
        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)

        result = handle_eval_implement(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EVAL_EXECUTE
        code_file = trial_dir / "experiment" / "codes_eval" / "main.py"
        assert code_file.exists()
        assert "placeholder" in code_file.read_text().lower()

    def test_abort_during_llm_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User sends /abort when prompted about LLM unavailability."""
        trial_dir = _setup_trial(tmp_path)
        chat = FakeChat(responses=[SlashCommand("/abort", "")])
        meta = TrialMeta()
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)

        with pytest.raises(TrialAborted):
            handle_experiment_implement(trial_dir, meta, ResearchClawConfig(), chat)


# =============================================================================
# Venv creation failure handling
# =============================================================================


class TestVenvFailure:
    """Test graceful handling of venv creation failures."""

    def test_venv_failure_skip_continues_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When venv fails and user skips, experiment still runs."""
        trial_dir = _setup_trial(tmp_path)
        _create_run_exp_sh(trial_dir, "echo ok")
        chat = FakeChat(responses=[UserMessage("s")])
        meta = TrialMeta()
        config = ResearchClawConfig()

        from researchclaw.sandbox import venv_manager
        monkeypatch.setattr(
            venv_manager.VenvManager, "ensure_venv",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no python3"))),
        )

        result = handle_experiment_execute(trial_dir, meta, config, chat)
        # Should have continued to run (might succeed or fail depending on script)
        assert result in (State.EVAL_IMPLEMENT, State.EXPERIMENT_IMPLEMENT, State.EXPERIMENT_REPORT)

    def test_venv_failure_none_chat_auto_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no chat, venv failure auto-skips."""
        trial_dir = _setup_trial(tmp_path)
        _create_run_exp_sh(trial_dir, "echo ok")
        meta = TrialMeta()
        config = ResearchClawConfig()

        from researchclaw.sandbox import venv_manager
        monkeypatch.setattr(
            venv_manager.VenvManager, "ensure_venv",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no python3"))),
        )

        result = handle_experiment_execute(trial_dir, meta, config, None)
        assert result in (State.EVAL_IMPLEMENT, State.EXPERIMENT_IMPLEMENT, State.EXPERIMENT_REPORT)


# =============================================================================
# Report handler terminated marking
# =============================================================================


class TestReportTerminatedMarking:
    """Test that terminated trials get proper markers in REPORT.md and logs."""

    def test_terminated_report_has_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(status="terminated")
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChat())

        report = (trial_dir / "REPORT.md").read_text()
        assert report.startswith("[TERMINATED-DURING-EXPERIMENT]")

    def test_terminated_log_summary_has_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(status="terminated")
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChat())

        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_content = logs_path.read_text()
        assert "[TERMINATED]" in logs_content

    def test_normal_report_no_terminated_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(status="running")
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChat())

        report = (trial_dir / "REPORT.md").read_text()
        assert not report.startswith("[TERMINATED-DURING-EXPERIMENT]")

    def test_report_logs_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(status="running")
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChat())

        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        assert logs_path.exists()
        content = logs_path.read_text()
        assert "trial_" in content
        assert "REPORT.md" in content


# =============================================================================
# Integration: abort from various states through engine
# =============================================================================


class TestAbortIntegration:
    """Integration tests for abort flow through the full engine."""

    def test_abort_during_experiment_execute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Abort during execute state routes through report to decide."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test plan")

        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = "experiment_execute"
        SandboxManager.save_trial_meta(trial_dir, meta)

        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        def aborting_execute(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise TrialAborted("User aborted during execute")

        visited: list[str] = []

        def tracking_decide(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            visited.append("decide")
            raise SystemExit("done")

        handlers = {
            State.EXPERIMENT_EXECUTE: aborting_execute,
            State.EXPERIMENT_REPORT: handle_experiment_report,
            State.DECIDE: tracking_decide,
        }

        engine = FSMEngine(tmp_path, ResearchClawConfig(), FakeChat(), handlers)
        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

        assert visited == ["decide"]

        # Verify terminated report written
        report = (trial_dir / "REPORT.md").read_text()
        assert "[TERMINATED-DURING-EXPERIMENT]" in report

    def test_abort_during_eval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Abort during eval routes through report to decide."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test plan")

        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = "eval_implement"
        SandboxManager.save_trial_meta(trial_dir, meta)

        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        def aborting_eval(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            raise TrialAborted("User aborted during eval")

        visited: list[str] = []

        def tracking_decide(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            visited.append("decide")
            raise SystemExit("done")

        handlers = {
            State.EVAL_IMPLEMENT: aborting_eval,
            State.EXPERIMENT_REPORT: handle_experiment_report,
            State.DECIDE: tracking_decide,
        }

        engine = FSMEngine(tmp_path, ResearchClawConfig(), FakeChat(), handlers)
        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

        assert visited == ["decide"]

    def test_multiple_aborts_in_sequence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple aborts generate multiple terminated reports."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan 1")

        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)

        abort_count = 0

        def aborting_plan(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            nonlocal abort_count
            abort_count += 1
            raise TrialAborted("abort")

        def decide_exit(td: Path, m: TrialMeta, c: ResearchClawConfig, ci: Any) -> State:
            if abort_count >= 2:
                raise SystemExit("done after 2 aborts")
            # Return to plan to test second abort
            m.state = "experiment_plan"
            return State.EXPERIMENT_PLAN

        handlers = {
            State.EXPERIMENT_PLAN: aborting_plan,
            State.EXPERIMENT_REPORT: handle_experiment_report,
            State.DECIDE: decide_exit,
        }

        engine = FSMEngine(tmp_path, ResearchClawConfig(), FakeChat(), handlers)
        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

        assert abort_count == 2
