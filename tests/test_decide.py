from __future__ import annotations

from pathlib import Path

import pytest

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.decide import (
    DECISION_OPTIONS,
    OPTION_MAP,
    _build_trial_summary,
    _parse_user_choice,
    handle_decide,
)
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager


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


class TestParseUserChoice:
    """Tests for _parse_user_choice helper."""

    def test_direct_number(self) -> None:
        assert _parse_user_choice("1") == "1"
        assert _parse_user_choice("2") == "2"
        assert _parse_user_choice("3") == "3"
        assert _parse_user_choice("4") == "4"
        assert _parse_user_choice("5") == "5"

    def test_with_whitespace(self) -> None:
        assert _parse_user_choice("  1  ") == "1"
        assert _parse_user_choice("  3\n") == "3"

    def test_invalid_input(self) -> None:
        assert _parse_user_choice("") is None
        assert _parse_user_choice("hello") is None
        assert _parse_user_choice("0") is None
        assert _parse_user_choice("6") is None
        assert _parse_user_choice("99") is None

    def test_extracts_first_valid_digit(self) -> None:
        assert _parse_user_choice("option 1") == "1"
        assert _parse_user_choice("I want 2") == "2"

    def test_number_in_text(self) -> None:
        assert _parse_user_choice("choose 3 please") == "3"


class TestBuildTrialSummary:
    """Tests for _build_trial_summary helper."""

    def test_basic_summary(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        meta = TrialMeta(trial_number=1, status="completed")
        summary = _build_trial_summary(trial_dir, meta)
        assert "20260302_trial_001" in summary
        assert "completed" in summary

    def test_summary_with_exit_codes(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        meta = TrialMeta(
            trial_number=1,
            status="completed",
            experiment_exit_code=0,
            eval_exit_code=0,
        )
        summary = _build_trial_summary(trial_dir, meta)
        assert "Experiment exit code" in summary
        assert "Eval exit code" in summary

    def test_summary_without_exit_codes(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        meta = TrialMeta(trial_number=1, status="pending")
        summary = _build_trial_summary(trial_dir, meta)
        assert "Experiment exit code" not in summary
        assert "Eval exit code" not in summary

    def test_summary_with_report(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        (trial_dir / "REPORT.md").write_text("# Report\nThis is the trial report content.")
        meta = TrialMeta(trial_number=1, status="completed")
        summary = _build_trial_summary(trial_dir, meta)
        assert "Report preview" in summary
        assert "trial report content" in summary

    def test_summary_truncates_long_report(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        (trial_dir / "REPORT.md").write_text("x" * 1000)
        meta = TrialMeta(trial_number=1, status="completed")
        summary = _build_trial_summary(trial_dir, meta)
        assert "truncated" in summary

    def test_summary_with_no_report(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        meta = TrialMeta(trial_number=1, status="completed")
        summary = _build_trial_summary(trial_dir, meta)
        assert "Report preview" not in summary

    def test_summary_with_previous_decision(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        meta = TrialMeta(trial_number=1, status="completed", decision="1")
        summary = _build_trial_summary(trial_dir, meta)
        assert "Previous decision" in summary

    def test_summary_with_empty_report(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir()
        (trial_dir / "REPORT.md").write_text("")
        meta = TrialMeta(trial_number=1, status="completed")
        summary = _build_trial_summary(trial_dir, meta)
        assert "Report preview" not in summary


class TestOptionMap:
    """Tests for DECISION_OPTIONS and OPTION_MAP constants."""

    def test_option_map_has_five_entries(self) -> None:
        assert len(OPTION_MAP) == 5

    def test_option_1_is_experiment_plan(self) -> None:
        assert OPTION_MAP["1"] == State.EXPERIMENT_PLAN

    def test_option_2_is_view_summary(self) -> None:
        assert OPTION_MAP["2"] == State.VIEW_SUMMARY

    def test_option_3_is_settings(self) -> None:
        assert OPTION_MAP["3"] == State.SETTINGS

    def test_option_4_is_merge_loop(self) -> None:
        assert OPTION_MAP["4"] == State.MERGE_LOOP

    def test_option_5_is_quit(self) -> None:
        assert OPTION_MAP["5"] == "quit"

    def test_decision_options_text_contains_all_options(self) -> None:
        assert "(1)" in DECISION_OPTIONS
        assert "(2)" in DECISION_OPTIONS
        assert "(3)" in DECISION_OPTIONS
        assert "(4)" in DECISION_OPTIONS
        assert "(5)" in DECISION_OPTIONS
        assert "New experiment" in DECISION_OPTIONS
        assert "View summary" in DECISION_OPTIONS
        assert "Settings" in DECISION_OPTIONS
        assert "Merge" in DECISION_OPTIONS
        assert "Quit" in DECISION_OPTIONS


class TestHandleDecideInteractive:
    """Tests for handle_decide in interactive mode."""

    def test_choose_new_experiment(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("1")])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_PLAN
        assert meta.decision == "1"

    def test_choose_view_summary(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("2")])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.VIEW_SUMMARY

    def test_choose_settings(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("3")])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.SETTINGS

    def test_choose_merge_transitions_to_merge_loop(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("4")])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.MERGE_LOOP

    def test_choose_quit_raises_system_exit(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("5")])
        with pytest.raises(SystemExit):
            handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert meta.decision == "quit"

    def test_invalid_input_reprompts(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("hello"), UserMessage("1")])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_PLAN
        assert any("1-5" in m for m in chat.sent)

    def test_shows_trial_summary(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta(status="completed")
        chat = FakeChatInterface(responses=[UserMessage("1")])
        handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert any("[DECIDE]" in m for m in chat.sent)
        assert any("completed" in m for m in chat.sent)

    def test_shows_decision_options(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("1")])
        handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert any("What would you like to do" in m for m in chat.sent)

    def test_meta_decision_set(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("2")])
        handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert meta.decision == "2"
        assert meta.decision_reasoning is not None

    def test_meta_decision_reasoning_set(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("3")])
        handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert "option 3" in (meta.decision_reasoning or "")


class TestHandleDecideAutopilot:
    """Tests for handle_decide in autopilot mode."""

    def test_autopilot_returns_experiment_plan(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        chat = FakeChatInterface()
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN

    def test_autopilot_sets_decision(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        chat = FakeChatInterface()
        handle_decide(trial_dir, meta, config, chat)
        assert meta.decision == "new_experiment"
        assert "Autopilot" in (meta.decision_reasoning or "")

    def test_autopilot_sends_message(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        chat = FakeChatInterface()
        handle_decide(trial_dir, meta, config, chat)
        assert any("Autopilot" in m for m in chat.sent)

    def test_autopilot_skips_user_input(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        # No responses needed — autopilot doesn't call receive()
        chat = FakeChatInterface(responses=[])
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN

    def test_autopilot_with_none_chat(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=True)
        result = handle_decide(trial_dir, meta, config, None)
        assert result == State.EXPERIMENT_PLAN
        assert meta.decision == "new_experiment"


class TestHandleDecideSlashCommands:
    """Tests for slash command handling in DECIDE state."""

    def test_quit_slash_command(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[SlashCommand("/quit", "")])
        with pytest.raises(SystemExit):
            handle_decide(trial_dir, meta, ResearchClawConfig(), chat)

    def test_status_slash_command_continues(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            SlashCommand("/status", ""),
            UserMessage("1"),
        ])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_PLAN

    def test_autopilot_slash_command_enables_and_starts(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            SlashCommand("/autopilot", ""),
            UserMessage("yes"),  # Confirmation
        ])
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN
        assert config.autopilot is True
        assert meta.decision == "new_experiment"

    def test_unknown_slash_command_reprompts(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            SlashCommand("/approve", ""),
            UserMessage("1"),
        ])
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.EXPERIMENT_PLAN
        assert any("not available" in m for m in chat.sent)


class TestHandleDecideNoneChat:
    """Tests for handle_decide with None chat_interface."""

    def test_none_chat_returns_experiment_plan(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        result = handle_decide(trial_dir, meta, ResearchClawConfig(), None)
        assert result == State.EXPERIMENT_PLAN
        assert meta.decision == "new_experiment"
