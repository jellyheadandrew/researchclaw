from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.decide import _persist_autopilot, handle_decide
from researchclaw.fsm.engine import FSMEngine
from researchclaw.fsm.plan import handle_experiment_plan
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager

import researchclaw.fsm.plan as plan_mod


class FakeChatInterface:
    """Fake chat interface with pre-programmed responses."""

    def __init__(self, responses: list[ChatInput] | None = None) -> None:
        self.sent: list[str] = []
        self._responses = list(responses) if responses else []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def receive(self) -> ChatInput:
        if not self._responses:
            raise SystemExit("No more responses")
        return self._responses.pop(0)


# --- /autopilot with confirmation in DECIDE ---


class TestAutopilotEnableDecide:
    """Tests for /autopilot slash command with confirmation in DECIDE."""

    def test_autopilot_enable_confirmed(self, tmp_path: Path) -> None:
        """Confirmed /autopilot enables autopilot and starts new experiment."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("yes"),
        ])
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN
        assert config.autopilot is True
        assert meta.decision == "new_experiment"
        assert any("Autopilot enabled" in m for m in chat.sent)

    def test_autopilot_enable_confirmed_y(self, tmp_path: Path) -> None:
        """Shorthand 'y' also confirms."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("y"),
        ])
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN
        assert config.autopilot is True

    def test_autopilot_enable_declined(self, tmp_path: Path) -> None:
        """Declining /autopilot does NOT enable autopilot."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("no"),
            UserMessage("1"),  # Choose option to continue
        ])
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN
        assert config.autopilot is False
        assert any("not enabled" in m for m in chat.sent)

    def test_autopilot_enable_sends_confirmation_prompt(self, tmp_path: Path) -> None:
        """Confirmation prompt is shown before enabling autopilot."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("yes"),
        ])
        handle_decide(trial_dir, meta, config, chat)
        assert any("Enable autopilot" in m for m in chat.sent)

    def test_autopilot_enable_persists_to_config(self, tmp_path: Path) -> None:
        """Enabling autopilot persists the setting to project config."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("yes"),
        ])
        handle_decide(trial_dir, meta, config, chat)
        # Verify persisted
        project_config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        saved = ResearchClawConfig.load_from_yaml(project_config_path)
        assert saved.autopilot is True


# --- /autopilot-stop in DECIDE ---


class TestAutopilotStopDecide:
    """Tests for /autopilot-stop slash command in DECIDE."""

    def test_autopilot_stop_disables(self, tmp_path: Path) -> None:
        """Disabling autopilot via /autopilot-stop."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        # Note: autopilot=True means handler would auto-select, but
        # we need to handle the case where autopilot-stop is processed first.
        # Actually, with config.autopilot=True, decide handler auto-selects.
        # So test with config.autopilot=False to ensure /autopilot-stop works in interactive.
        config.autopilot = False  # Interactive mode so receive() is called
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot-stop", ""),
            UserMessage("1"),
        ])
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN
        assert config.autopilot is False
        assert any("Autopilot disabled" in m for m in chat.sent)

    def test_autopilot_stop_persists_to_config(self, tmp_path: Path) -> None:
        """Disabling autopilot persists the setting to project config."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        # Pre-set config with autopilot on in file
        project_config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        pre_config = ResearchClawConfig(autopilot=True)
        pre_config.save_to_yaml(project_config_path)

        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot-stop", ""),
            UserMessage("1"),
        ])
        handle_decide(trial_dir, meta, config, chat)

        saved = ResearchClawConfig.load_from_yaml(project_config_path)
        assert saved.autopilot is False


# --- /autopilot and /autopilot-stop in EXPERIMENT_PLAN ---


class TestAutopilotEnablePlan:
    """Tests for /autopilot in EXPERIMENT_PLAN state."""

    def test_autopilot_enable_in_plan_confirmed(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        """Enabling autopilot during planning switches to autopilot plan."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("yes"),
        ])
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        result = handle_experiment_plan(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_IMPLEMENT
        assert config.autopilot is True
        assert any("Autopilot enabled" in m for m in chat.sent)

    def test_autopilot_enable_in_plan_declined(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        """Declining autopilot during planning continues interactive mode."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("no"),
            SlashCommand("/approve", ""),
        ])
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        result = handle_experiment_plan(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_IMPLEMENT
        assert config.autopilot is False

    def test_autopilot_enable_in_plan_persists(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        """Enabling autopilot during planning persists to config."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("yes"),
        ])
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        handle_experiment_plan(trial_dir, meta, config, chat)
        project_config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        saved = ResearchClawConfig.load_from_yaml(project_config_path)
        assert saved.autopilot is True


class TestAutopilotStopPlan:
    """Tests for /autopilot-stop in EXPERIMENT_PLAN state."""

    def test_autopilot_stop_in_plan(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        """Disabling autopilot during planning continues interactive mode."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot-stop", ""),
            SlashCommand("/approve", ""),
        ])
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        result = handle_experiment_plan(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_IMPLEMENT
        assert config.autopilot is False
        assert any("Autopilot disabled" in m for m in chat.sent)


# --- Autopilot reasoning in meta.json at every FSM transition ---


class TestAutopilotDecisionReasoning:
    """Tests for autopilot logging decision reasoning at every state transition."""

    def test_autopilot_logs_reasoning_at_transition(self, tmp_path: Path) -> None:
        """In autopilot mode, engine logs reasoning to meta at each transition."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        reasonings_seen: list[str | None] = []

        def handler_a(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            meta.decision_reasoning = None  # Reset to test engine sets it
            return State.EXPERIMENT_IMPLEMENT

        def handler_b(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            # Check what reasoning was saved after handler_a
            saved = SandboxManager.get_trial_meta(td)
            reasonings_seen.append(saved.decision_reasoning)
            raise SystemExit("stop")

        handlers = {
            State.EXPERIMENT_PLAN: handler_a,
            State.EXPERIMENT_IMPLEMENT: handler_b,
        }

        config = ResearchClawConfig(autopilot=True)
        engine = FSMEngine(tmp_path, config, None, handlers)
        try:
            engine.run(trial_dir=trial_dir)
        except SystemExit:
            pass

        assert len(reasonings_seen) == 1
        assert reasonings_seen[0] is not None
        assert "[autopilot]" in reasonings_seen[0]

    def test_non_autopilot_does_not_log_reasoning(self, tmp_path: Path) -> None:
        """Non-autopilot mode does NOT auto-set decision_reasoning."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def handler_a(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            meta.decision_reasoning = None
            return State.EXPERIMENT_IMPLEMENT

        def handler_b(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            saved = SandboxManager.get_trial_meta(td)
            assert saved.decision_reasoning is None
            raise SystemExit("stop")

        handlers = {
            State.EXPERIMENT_PLAN: handler_a,
            State.EXPERIMENT_IMPLEMENT: handler_b,
        }

        config = ResearchClawConfig(autopilot=False)
        engine = FSMEngine(tmp_path, config, None, handlers)
        try:
            engine.run(trial_dir=trial_dir)
        except SystemExit:
            pass

    def test_handler_set_reasoning_preserved(self, tmp_path: Path) -> None:
        """If handler sets reasoning, engine doesn't overwrite it."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        def handler_a(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            meta.decision_reasoning = "Custom reasoning from handler"
            return State.EXPERIMENT_IMPLEMENT

        def handler_b(
            td: Path, meta: TrialMeta, cfg: ResearchClawConfig, ci: object,
        ) -> State:
            saved = SandboxManager.get_trial_meta(td)
            assert saved.decision_reasoning == "Custom reasoning from handler"
            raise SystemExit("stop")

        handlers = {
            State.EXPERIMENT_PLAN: handler_a,
            State.EXPERIMENT_IMPLEMENT: handler_b,
        }

        config = ResearchClawConfig(autopilot=True)
        engine = FSMEngine(tmp_path, config, None, handlers)
        try:
            engine.run(trial_dir=trial_dir)
        except SystemExit:
            pass


# --- Autopilot status in display ---


class TestAutopilotStatusDisplay:
    """Tests for autopilot status shown in DECIDE and CLI."""

    def test_decide_shows_autopilot_on(self, tmp_path: Path) -> None:
        """DECIDE handler shows 'Autopilot: ON' when enabled."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        chat = FakeChatInterface()
        handle_decide(trial_dir, meta, config, chat)
        assert any("Autopilot: ON" in m for m in chat.sent)

    def test_decide_shows_autopilot_off(self, tmp_path: Path) -> None:
        """DECIDE handler shows 'Autopilot: OFF' when disabled."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[UserMessage("1")])
        handle_decide(trial_dir, meta, config, chat)
        assert any("Autopilot: OFF" in m for m in chat.sent)


# --- Autopilot persistence in config ---


class TestAutopilotPersistence:
    """Tests for autopilot state persisting in config file."""

    def test_persist_autopilot_true(self, tmp_path: Path) -> None:
        """_persist_autopilot saves autopilot=True to config."""
        SandboxManager.initialize(tmp_path)
        config = ResearchClawConfig(autopilot=True)
        _persist_autopilot(config, tmp_path)
        project_config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        saved = ResearchClawConfig.load_from_yaml(project_config_path)
        assert saved.autopilot is True

    def test_persist_autopilot_false(self, tmp_path: Path) -> None:
        """_persist_autopilot saves autopilot=False to config."""
        SandboxManager.initialize(tmp_path)
        config = ResearchClawConfig(autopilot=False)
        _persist_autopilot(config, tmp_path)
        project_config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        saved = ResearchClawConfig.load_from_yaml(project_config_path)
        assert saved.autopilot is False

    def test_persist_autopilot_preserves_other_settings(self, tmp_path: Path) -> None:
        """Persisting autopilot doesn't lose other config settings."""
        SandboxManager.initialize(tmp_path)
        project_config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        # Save custom config
        custom = ResearchClawConfig(max_retries=99, autopilot=False)
        custom.save_to_yaml(project_config_path)

        # Now persist autopilot change
        config = ResearchClawConfig(autopilot=True)
        _persist_autopilot(config, tmp_path)

        saved = ResearchClawConfig.load_from_yaml(project_config_path)
        assert saved.autopilot is True
        # Other fields come from loaded config, so max_retries should be preserved
        assert saved.max_retries == 99

    def test_persist_autopilot_survives_restart(self, tmp_path: Path) -> None:
        """Autopilot state loaded back via load_merged_config."""
        SandboxManager.initialize(tmp_path)
        config = ResearchClawConfig(autopilot=True)
        _persist_autopilot(config, tmp_path)

        # Simulate restart by loading merged config
        loaded = ResearchClawConfig.load_merged_config(tmp_path)
        assert loaded.autopilot is True


# --- CLI autopilot display ---


class TestCLIAutopilotDisplay:
    """Tests for autopilot status in CLI output."""

    @staticmethod
    def _patch_fsm(monkeypatch: object) -> None:
        """Patch FSM engine to avoid real terminal interaction."""
        import researchclaw.cli as cli_mod
        monkeypatch.setattr(cli_mod, "TerminalChat", lambda **kw: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(cli_mod, "_build_handlers", lambda: {})  # type: ignore[attr-defined]
        class FakeEngine:
            def __init__(self, *a, **kw):
                pass
            def run(self):
                raise SystemExit("done")
        monkeypatch.setattr(cli_mod, "FSMEngine", FakeEngine)  # type: ignore[attr-defined]

    def test_cli_shows_autopilot_off(self, tmp_path: Path, monkeypatch: object) -> None:
        from click.testing import CliRunner
        from researchclaw.cli import main
        import researchclaw.cli as cli_mod

        self._patch_fsm(monkeypatch)
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)  # type: ignore[attr-defined]
        SandboxManager.initialize(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Autopilot: OFF" in result.output

    def test_cli_shows_autopilot_on(self, tmp_path: Path, monkeypatch: object) -> None:
        from click.testing import CliRunner
        from researchclaw.cli import main
        import researchclaw.cli as cli_mod

        self._patch_fsm(monkeypatch)
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)  # type: ignore[attr-defined]
        SandboxManager.initialize(tmp_path)

        # Set autopilot ON in project config
        project_config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        cfg = ResearchClawConfig(autopilot=True)
        cfg.save_to_yaml(project_config_path)

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Autopilot: ON" in result.output


# --- Autopilot in DECIDE auto-plan with reasoning in meta ---


class TestAutopilotDecideAutoAdvance:
    """Tests for autopilot auto-advancing from DECIDE with proper reasoning."""

    def test_autopilot_auto_advance_sets_reasoning(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        chat = FakeChatInterface()
        handle_decide(trial_dir, meta, config, chat)
        assert meta.decision_reasoning is not None
        assert "Autopilot" in meta.decision_reasoning

    def test_autopilot_plan_auto_generates(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        """In autopilot mode, plan handler auto-generates plan without user input."""
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        chat = FakeChatInterface(responses=[])  # No responses needed
        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        result = handle_experiment_plan(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_IMPLEMENT
        # PLAN.md should be written
        assert (trial_dir / "PLAN.md").exists()
        assert any("autopilot" in m.lower() for m in chat.sent)
