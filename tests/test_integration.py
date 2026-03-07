"""Integration tests for US-022: End-to-end CLI wiring and FSM loop."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import FakeChatInterface
from researchclaw.cli import _build_handlers, main
from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.engine import FSMEngine
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager

import researchclaw.cli as cli_mod
import researchclaw.fsm.evaluate as evaluate_mod
import researchclaw.fsm.experiment as experiment_mod
import researchclaw.fsm.plan as plan_mod
import researchclaw.fsm.report as report_mod


def _patch_all_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch all handler modules to return no LLM provider."""
    monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)
    monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)
    monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")
    monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)
    monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")
    monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)


class TestBuildHandlers:
    """Tests for _build_handlers helper."""

    def test_returns_all_handlers(self) -> None:
        handlers = _build_handlers()
        expected_states = {
            State.EXPERIMENT_PLAN,
            State.EXPERIMENT_IMPLEMENT,
            State.EXPERIMENT_EXECUTE,
            State.EVAL_IMPLEMENT,
            State.EVAL_EXECUTE,
            State.EXPERIMENT_REPORT,
            State.DECIDE,
            State.VIEW_SUMMARY,
            State.SETTINGS,
            State.MERGE_LOOP,
        }
        assert set(handlers.keys()) == expected_states

    def test_all_handlers_are_callable(self) -> None:
        handlers = _build_handlers()
        for state, handler in handlers.items():
            assert callable(handler), f"Handler for {state} is not callable"

    def test_merge_loop_in_handlers(self) -> None:
        """MERGE_LOOP is in the handler map as a stub that returns DECIDE."""
        handlers = _build_handlers()
        assert State.MERGE_LOOP in handlers


class TestFullFSMLoop:
    """Integration tests: full FSM loop with real handlers (no LLM)."""

    def test_plan_to_decide_full_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full FSM loop: PLAN -> IMPLEMENT -> EXECUTE -> EVAL_IMPL -> EVAL_EXEC -> REPORT -> DECIDE."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        _patch_all_providers(monkeypatch)

        # Create run scripts that succeed
        script_path = trial_dir / "experiment" / "run_exp.sh"
        script_path.write_text("#!/usr/bin/env bash\necho experiment_ok\n")
        script_path.chmod(0o755)

        # PLAN phase: approve immediately
        # After plan, implement will generate run_exp.sh (overwriting ours),
        # so we need to handle that. Instead, let's trace through with
        # a FakeChat that provides /approve for plan, then options for decide.
        chat = FakeChatInterface(responses=[
            SlashCommand("/approve", ""),  # approve plan
            UserMessage("5"),              # quit at DECIDE
        ])
        config = ResearchClawConfig(max_retries=1)
        handlers = _build_handlers()
        engine = FSMEngine(tmp_path, config, chat, handlers)

        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

        # Verify the loop visited PLAN → IMPLEMENT → ... → DECIDE
        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.state == "decide"

        # PLAN.md should exist
        assert (trial_dir / "PLAN.md").exists()

        # Experiment code should have been generated
        assert (trial_dir / "experiment" / "codes_exp" / "main.py").exists()

        # run_exp.sh should have been generated
        assert (trial_dir / "experiment" / "run_exp.sh").exists()

    def test_resume_from_mid_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resume FSM from a mid-state (EXPERIMENT_EXECUTE)."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        _patch_all_providers(monkeypatch)

        # Set state to EXPERIMENT_EXECUTE
        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = State.EXPERIMENT_EXECUTE.value
        meta.status = "running"
        SandboxManager.save_trial_meta(trial_dir, meta)

        # Create a successful run_exp.sh
        script_path = trial_dir / "experiment" / "run_exp.sh"
        script_path.write_text("#!/usr/bin/env bash\necho ok\n")
        script_path.chmod(0o755)

        # PLAN.md needed for eval implement
        (trial_dir / "PLAN.md").write_text("# Test Plan")

        chat = FakeChatInterface(responses=[
            UserMessage("5"),  # quit at DECIDE
        ])
        config = ResearchClawConfig(max_retries=1)
        handlers = _build_handlers()
        engine = FSMEngine(tmp_path, config, chat, handlers)

        with pytest.raises(SystemExit):
            engine.run()  # No trial_dir → resume from latest

        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.state == "decide"

    def test_resume_completed_trial_at_decide(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Completed trial that was set to DECIDE state resumes at DECIDE."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = State.DECIDE.value
        meta.status = "completed"
        SandboxManager.save_trial_meta(trial_dir, meta)

        chat = FakeChatInterface(responses=[
            UserMessage("5"),  # quit
        ])
        config = ResearchClawConfig()
        handlers = _build_handlers()
        engine = FSMEngine(tmp_path, config, chat, handlers)

        with pytest.raises(SystemExit):
            engine.run()

        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.state == "decide"

    def test_engine_creates_trial_on_first_run(
        self, tmp_path: Path
    ) -> None:
        """Engine creates a trial when none exist."""
        SandboxManager.initialize(tmp_path)

        # Empty handler map — engine will create trial then stop
        handlers: dict[State, object] = {}
        config = ResearchClawConfig()
        engine = FSMEngine(tmp_path, config, None, handlers)
        engine.run()

        latest = SandboxManager.get_latest_trial(tmp_path)
        assert latest is not None
        assert "_trial_" in latest.name


class TestCLIIntegration:
    """Tests for CLI wiring with the FSM."""

    @staticmethod
    def _patch_fsm(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
        """Patch CLI to use a fake FSM engine. Returns a list that tracks run() calls."""
        run_called: list[bool] = []
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)
        monkeypatch.setattr(cli_mod, "TerminalChat", lambda **kw: None)
        monkeypatch.setattr(cli_mod, "_build_handlers", lambda: {})

        class FakeEngine:
            def __init__(self, *a, **kw):
                pass
            def run(self):
                run_called.append(True)
                raise SystemExit("done")

        monkeypatch.setattr(cli_mod, "FSMEngine", FakeEngine)
        return run_called

    def test_cli_runs_fsm_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI creates FSMEngine and calls run()."""
        run_called = self._patch_fsm(monkeypatch)
        SandboxManager.initialize(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert len(run_called) == 1
        assert "Goodbye!" in result.output

    def test_cli_initializes_sandbox(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI initializes sandbox if not already initialized."""
        self._patch_fsm(monkeypatch)
        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Initializing sandbox..." in result.output
        assert SandboxManager.is_initialized(tmp_path)

    def test_cli_displays_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI displays version on startup."""
        self._patch_fsm(monkeypatch)
        SandboxManager.initialize(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "ResearchClaw v" in result.output

    def test_cli_first_run_no_trials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First run shows 'No trials yet' message."""
        self._patch_fsm(monkeypatch)
        SandboxManager.initialize(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "No trials yet." in result.output

    def test_cli_resume_running_trial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI shows resume message for running trials."""
        self._patch_fsm(monkeypatch)
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)
        meta = TrialMeta(trial_number=1, status="running", state="experiment_implement")
        SandboxManager.save_trial_meta(trial_dir, meta)

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Resuming trial" in result.output
        assert "experiment_implement" in result.output

    def test_cli_completed_trial_sets_decide(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI sets completed trial to DECIDE state."""
        self._patch_fsm(monkeypatch)
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)
        meta = TrialMeta(trial_number=1, status="completed", state="experiment_report")
        SandboxManager.save_trial_meta(trial_dir, meta)

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Entering DECIDE state" in result.output

        # Verify meta was updated
        loaded = SandboxManager.get_trial_meta(trial_dir)
        assert loaded.state == "decide"

    def test_cli_terminated_trial_sets_decide(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI sets terminated trial to DECIDE state."""
        self._patch_fsm(monkeypatch)
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)
        meta = TrialMeta(trial_number=1, status="terminated", state="experiment_execute")
        SandboxManager.save_trial_meta(trial_dir, meta)

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Entering DECIDE state" in result.output

    def test_status_subcommand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'researchclaw status' works as subcommand."""
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)
        SandboxManager.initialize(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["status"], obj={"project_dir": str(tmp_path)})
        assert "ResearchClaw Trials" in result.output

    def test_status_subcommand_no_sandbox(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'researchclaw status' shows message when no sandbox."""
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)
        runner = CliRunner()
        result = runner.invoke(main, ["status"], obj={"project_dir": str(tmp_path)})
        assert "No sandbox found" in result.output


class TestMainModule:
    """Tests for python -m researchclaw support."""

    def test_main_module_imports(self) -> None:
        """__main__.py imports correctly."""
        from researchclaw.__main__ import main as main_fn
        assert callable(main_fn)

    def test_version_option(self) -> None:
        """--version flag works."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert "researchclaw" in result.output
        assert result.exit_code == 0

    def test_help_option(self) -> None:
        """--help flag works."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "ResearchClaw" in result.output
        assert result.exit_code == 0


class TestViewSummaryAndSettings:
    """Integration tests for VIEW_SUMMARY and SETTINGS from DECIDE."""

    def test_decide_to_view_summary_and_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DECIDE → VIEW_SUMMARY → DECIDE → quit."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = State.DECIDE.value
        SandboxManager.save_trial_meta(trial_dir, meta)

        chat = FakeChatInterface(responses=[
            UserMessage("2"),       # View summary
            UserMessage("back"),    # Return from VIEW_SUMMARY
            UserMessage("5"),       # Quit
        ])
        config = ResearchClawConfig()
        handlers = _build_handlers()
        engine = FSMEngine(tmp_path, config, chat, handlers)

        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)

    def test_decide_to_settings_and_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DECIDE → SETTINGS → DECIDE → quit."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = State.DECIDE.value
        SandboxManager.save_trial_meta(trial_dir, meta)

        chat = FakeChatInterface(responses=[
            UserMessage("3"),       # Settings
            UserMessage("back"),    # Return from SETTINGS
            UserMessage("5"),       # Quit
        ])
        config = ResearchClawConfig()
        handlers = _build_handlers()
        engine = FSMEngine(tmp_path, config, chat, handlers)

        with pytest.raises(SystemExit):
            engine.run(trial_dir=trial_dir)
