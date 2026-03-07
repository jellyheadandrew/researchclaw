from __future__ import annotations

from pathlib import Path

from conftest import FakeChatInterface
from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.decide import handle_decide
from researchclaw.fsm.engine import FSMEngine
from researchclaw.fsm.evaluate import handle_eval_execute, handle_eval_implement
from researchclaw.fsm.experiment import (
    handle_experiment_execute,
    handle_experiment_implement,
)
from researchclaw.fsm.plan import handle_experiment_plan
from researchclaw.fsm.report import handle_experiment_report
from researchclaw.fsm.states import State
from researchclaw.fsm.settings import handle_settings
from researchclaw.fsm.view_summary import handle_view_summary
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager

import researchclaw.fsm.evaluate as evaluate_mod
import researchclaw.fsm.experiment as experiment_mod
import researchclaw.fsm.plan as plan_mod
import researchclaw.fsm.report as report_mod


def _all_stub_handlers() -> dict[State, object]:
    """Return the full handler map with all stub implementations."""
    return {
        State.EXPERIMENT_PLAN: handle_experiment_plan,
        State.EXPERIMENT_IMPLEMENT: handle_experiment_implement,
        State.EXPERIMENT_EXECUTE: handle_experiment_execute,
        State.EVAL_IMPLEMENT: handle_eval_implement,
        State.EVAL_EXECUTE: handle_eval_execute,
        State.EXPERIMENT_REPORT: handle_experiment_report,
        State.DECIDE: handle_decide,
    }


class TestFSMEngine:
    """Tests for the FSMEngine dispatch loop."""

    def test_engine_init(self, tmp_path: Path) -> None:
        """FSMEngine stores project_dir, config, chat_interface, and handlers."""
        config = ResearchClawConfig()
        chat = FakeChatInterface()
        handlers = _all_stub_handlers()

        engine = FSMEngine(tmp_path, config, chat, handlers)

        assert engine.project_dir == tmp_path
        assert engine.config is config
        assert engine.chat_interface is chat
        assert engine.handlers is handlers

    def test_full_cycle_plan_to_decide(self, tmp_path: Path) -> None:
        """FSM progresses EXPERIMENT_PLAN -> ... -> DECIDE with stub handlers.

        We use a modified DECIDE handler that raises SystemExit after one
        full cycle so the loop terminates.
        """
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        visited_states: list[str] = []

        def tracking_decide(
            trial_dir: Path,
            meta: TrialMeta,
            config: ResearchClawConfig,
            chat_interface: object,
        ) -> State:
            visited_states.append("decide")
            raise SystemExit("done")

        def tracking_handler(state_name: str, next_state: State):
            def handler(
                trial_dir: Path,
                meta: TrialMeta,
                config: ResearchClawConfig,
                chat_interface: object,
            ) -> State:
                visited_states.append(state_name)
                return next_state
            return handler

        handlers = {
            State.EXPERIMENT_PLAN: tracking_handler("experiment_plan", State.EXPERIMENT_IMPLEMENT),
            State.EXPERIMENT_IMPLEMENT: tracking_handler("experiment_implement", State.EXPERIMENT_EXECUTE),
            State.EXPERIMENT_EXECUTE: tracking_handler("experiment_execute", State.EVAL_IMPLEMENT),
            State.EVAL_IMPLEMENT: tracking_handler("eval_implement", State.EVAL_EXECUTE),
            State.EVAL_EXECUTE: tracking_handler("eval_execute", State.EXPERIMENT_REPORT),
            State.EXPERIMENT_REPORT: tracking_handler("experiment_report", State.DECIDE),
            State.DECIDE: tracking_decide,
        }

        config = ResearchClawConfig()
        chat = FakeChatInterface()
        engine = FSMEngine(tmp_path, config, chat, handlers)

        try:
            engine.run(trial_dir=trial_dir)
        except SystemExit:
            pass

        assert visited_states == [
            "experiment_plan",
            "experiment_implement",
            "experiment_execute",
            "eval_implement",
            "eval_execute",
            "experiment_report",
            "decide",
        ]

    def test_meta_updated_after_each_transition(self, tmp_path: Path) -> None:
        """meta.json is saved with updated state after each handler call."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        states_seen_in_meta: list[str] = []

        def plan_handler(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            return State.EXPERIMENT_IMPLEMENT

        def impl_handler(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            # At this point, meta.json should have been updated to experiment_implement
            saved_meta = SandboxManager.get_trial_meta(td)
            states_seen_in_meta.append(saved_meta.state)
            raise SystemExit("stop")

        handlers = {
            State.EXPERIMENT_PLAN: plan_handler,
            State.EXPERIMENT_IMPLEMENT: impl_handler,
        }

        config = ResearchClawConfig()
        engine = FSMEngine(tmp_path, config, None, handlers)

        try:
            engine.run(trial_dir=trial_dir)
        except SystemExit:
            pass

        # After plan_handler returned EXPERIMENT_IMPLEMENT, meta.json was saved
        # with state=experiment_implement before impl_handler was called
        assert states_seen_in_meta == ["experiment_implement"]

    def test_engine_stops_on_missing_handler(self, tmp_path: Path) -> None:
        """Engine stops if no handler is registered for the current state."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        # Only provide handler for EXPERIMENT_PLAN that transitions to
        # EXPERIMENT_IMPLEMENT, but no handler for EXPERIMENT_IMPLEMENT
        call_count = 0

        def plan_handler(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            nonlocal call_count
            call_count += 1
            return State.EXPERIMENT_IMPLEMENT

        handlers = {
            State.EXPERIMENT_PLAN: plan_handler,
        }

        config = ResearchClawConfig()
        engine = FSMEngine(tmp_path, config, None, handlers)
        engine.run(trial_dir=trial_dir)

        assert call_count == 1
        # meta.json should reflect the last transition
        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.state == "experiment_implement"

    def test_engine_creates_trial_if_none_exist(self, tmp_path: Path) -> None:
        """Engine creates a new trial if no trials exist in the sandbox."""
        SandboxManager.initialize(tmp_path)

        # No handler for EXPERIMENT_PLAN, so engine will stop immediately
        # after creating the trial
        handlers: dict[State, object] = {}
        config = ResearchClawConfig()
        engine = FSMEngine(tmp_path, config, None, handlers)
        engine.run()

        # A trial should have been created
        latest = SandboxManager.get_latest_trial(tmp_path)
        assert latest is not None

    def test_engine_uses_latest_trial_if_exists(self, tmp_path: Path) -> None:
        """Engine resumes from the latest trial when no trial_dir is given."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        # Set state to EVAL_IMPLEMENT to verify it resumes from there
        meta = SandboxManager.get_trial_meta(trial_dir)
        meta.state = State.EVAL_IMPLEMENT.value
        SandboxManager.save_trial_meta(trial_dir, meta)

        visited: list[str] = []

        def eval_impl_handler(
            td: Path, m: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            visited.append("eval_implement")
            raise SystemExit("stop")

        handlers = {
            State.EVAL_IMPLEMENT: eval_impl_handler,
        }

        config = ResearchClawConfig()
        engine = FSMEngine(tmp_path, config, None, handlers)

        try:
            engine.run()
        except SystemExit:
            pass

        assert visited == ["eval_implement"]


class TestStubHandlers:
    """Tests for individual stub handlers."""

    def test_plan_returns_implement(self, tmp_path: Path, monkeypatch: object) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[SlashCommand("/approve", "")])
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        result = handle_experiment_plan(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_IMPLEMENT

    def test_implement_returns_execute(self, tmp_path: Path, monkeypatch: object) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan")
        meta = TrialMeta()
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]
        result = handle_experiment_implement(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert result == State.EXPERIMENT_EXECUTE

    def test_execute_returns_eval_implement(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        # Create a successful run_exp.sh
        script_path = trial_dir / "experiment" / "run_exp.sh"
        script_path.write_text("#!/usr/bin/env bash\necho ok\n")
        script_path.chmod(0o755)
        meta = TrialMeta()
        result = handle_experiment_execute(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert result == State.EVAL_IMPLEMENT

    def test_eval_implement_returns_eval_execute(self, tmp_path: Path, monkeypatch: object) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan")
        meta = TrialMeta()
        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]
        result = handle_eval_implement(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert result == State.EVAL_EXECUTE

    def test_eval_execute_returns_report(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        # Create a successful run_eval.sh
        script_path = trial_dir / "experiment" / "run_eval.sh"
        script_path.write_text("#!/usr/bin/env bash\necho ok\n")
        script_path.chmod(0o755)
        meta = TrialMeta()
        result = handle_eval_execute(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert result == State.EXPERIMENT_REPORT

    def test_report_returns_decide(self, tmp_path: Path, monkeypatch: object) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        result = handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert result == State.DECIDE

    def test_decide_returns_plan(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("1")])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_PLAN

    def test_handlers_work_with_none_chat_interface(self, tmp_path: Path, monkeypatch: object) -> None:
        """Handlers should work when chat_interface is None."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan")
        meta = TrialMeta()
        config = ResearchClawConfig()
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        assert handle_experiment_plan(trial_dir, meta, config, None) == State.EXPERIMENT_IMPLEMENT
        assert handle_experiment_implement(trial_dir, meta, config, None) == State.EXPERIMENT_EXECUTE
        # Create run_exp.sh for execute handler (no longer a stub)
        script_path = trial_dir / "experiment" / "run_exp.sh"
        script_path.write_text("#!/usr/bin/env bash\necho ok\n")
        script_path.chmod(0o755)
        assert handle_experiment_execute(trial_dir, meta, config, None) == State.EVAL_IMPLEMENT
        assert handle_eval_implement(trial_dir, meta, config, None) == State.EVAL_EXECUTE
        # Create run_eval.sh for eval execute handler (no longer a stub)
        eval_script_path = trial_dir / "experiment" / "run_eval.sh"
        eval_script_path.write_text("#!/usr/bin/env bash\necho ok\n")
        eval_script_path.chmod(0o755)
        assert handle_eval_execute(trial_dir, meta, config, None) == State.EXPERIMENT_REPORT
        assert handle_experiment_report(trial_dir, meta, config, None) == State.DECIDE
        assert handle_decide(trial_dir, meta, config, None) == State.EXPERIMENT_PLAN

    def test_decide_sends_messages(self, tmp_path: Path) -> None:
        """DECIDE handler sends summary and options via chat_interface."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        chat = FakeChatInterface(responses=[UserMessage("1")])
        meta = TrialMeta()
        config = ResearchClawConfig()

        handle_decide(trial_dir, meta, config, chat)

        assert len(chat.messages) >= 2  # summary + options at minimum
        assert any("[DECIDE]" in m for m in chat.messages)
