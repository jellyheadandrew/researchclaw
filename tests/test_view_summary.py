from __future__ import annotations

from pathlib import Path

import pytest

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.states import State
from researchclaw.fsm.view_summary import (
    DEFAULT_DISPLAY_TRIALS,
    _format_trial_table,
    _get_unique_dates,
    _parse_experiment_logs,
    _browse_by_date,
    handle_view_summary,
)
from researchclaw.models import TrialMeta
from conftest import FakeChatInterface
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager


def _setup_sandbox_with_logs(project_dir: Path, log_lines: list[str]) -> Path:
    """Helper: initialize sandbox and write EXPERIMENT_LOGS.md with given lines."""
    SandboxManager.initialize(project_dir)
    logs_path = SandboxManager.sandbox_path(project_dir) / "EXPERIMENT_LOGS.md"
    logs_path.write_text("\n".join(log_lines) + "\n" if log_lines else "")
    trial_dir = SandboxManager.create_trial(project_dir)
    return trial_dir


SAMPLE_LOG_LINES = [
    "20260301 - trial_001: First experiment on transformers. Full Doc: [REPORT.md](experiments/20260301_trial_001/REPORT.md)",
    "20260301 - trial_002: Improved learning rate schedule. Full Doc: [REPORT.md](experiments/20260301_trial_002/REPORT.md)",
    "20260302 - trial_001: Added data augmentation. Full Doc: [REPORT.md](experiments/20260302_trial_001/REPORT.md)",
    "20260302 - trial_002: Switched to Adam optimizer. Full Doc: [REPORT.md](experiments/20260302_trial_002/REPORT.md)",
    "20260303 - trial_001: Batch normalization experiment. Full Doc: [REPORT.md](experiments/20260303_trial_001/REPORT.md)",
]


class TestParseExperimentLogs:
    """Tests for _parse_experiment_logs helper."""

    def test_empty_logs(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text("")
        entries = _parse_experiment_logs(tmp_path)
        assert entries == []

    def test_no_logs_file(self, tmp_path: Path) -> None:
        entries = _parse_experiment_logs(tmp_path)
        assert entries == []

    def test_single_entry(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text(
            "20260301 - trial_001: First experiment. Full Doc: [REPORT.md](experiments/20260301_trial_001/REPORT.md)\n"
        )
        entries = _parse_experiment_logs(tmp_path)
        assert len(entries) == 1
        assert entries[0]["date"] == "20260301"
        assert entries[0]["trial"] == "trial_001"
        assert "First experiment" in entries[0]["summary"]

    def test_multiple_entries(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text("\n".join(SAMPLE_LOG_LINES) + "\n")
        entries = _parse_experiment_logs(tmp_path)
        assert len(entries) == 5

    def test_preserves_order(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text("\n".join(SAMPLE_LOG_LINES) + "\n")
        entries = _parse_experiment_logs(tmp_path)
        assert entries[0]["date"] == "20260301"
        assert entries[0]["trial"] == "trial_001"
        assert entries[-1]["date"] == "20260303"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text(
            "\n\n20260301 - trial_001: First experiment. Full Doc: [REPORT.md](...)\n\n"
        )
        entries = _parse_experiment_logs(tmp_path)
        assert len(entries) == 1

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text(
            "This is not a valid log line\n"
            "20260301 - trial_001: Valid entry. Full Doc: [REPORT.md](...)\n"
        )
        entries = _parse_experiment_logs(tmp_path)
        assert len(entries) == 1
        assert entries[0]["trial"] == "trial_001"


class TestGetUniqueDates:
    """Tests for _get_unique_dates helper."""

    def test_empty_entries(self) -> None:
        assert _get_unique_dates([]) == []

    def test_single_date(self) -> None:
        entries = [{"date": "20260301", "trial": "trial_001", "summary": "a", "full_line": "..."}]
        assert _get_unique_dates(entries) == ["20260301"]

    def test_multiple_dates(self) -> None:
        entries = [
            {"date": "20260301", "trial": "trial_001", "summary": "a", "full_line": "..."},
            {"date": "20260301", "trial": "trial_002", "summary": "b", "full_line": "..."},
            {"date": "20260302", "trial": "trial_001", "summary": "c", "full_line": "..."},
        ]
        dates = _get_unique_dates(entries)
        assert dates == ["20260301", "20260302"]

    def test_preserves_order(self) -> None:
        entries = [
            {"date": "20260303", "trial": "trial_001", "summary": "a", "full_line": "..."},
            {"date": "20260301", "trial": "trial_001", "summary": "b", "full_line": "..."},
        ]
        dates = _get_unique_dates(entries)
        assert dates == ["20260303", "20260301"]


class TestFormatTrialTable:
    """Tests for _format_trial_table helper."""

    def test_empty_entries(self) -> None:
        assert _format_trial_table([]) == "No trials found."

    def test_single_entry(self) -> None:
        entries = [{"date": "20260301", "trial": "trial_001", "summary": "First experiment", "full_line": "..."}]
        result = _format_trial_table(entries)
        assert "[20260301]" in result
        assert "[trial_001]" in result
        assert "First experiment" in result

    def test_multiple_entries(self) -> None:
        entries = [
            {"date": "20260301", "trial": "trial_001", "summary": "First", "full_line": "..."},
            {"date": "20260302", "trial": "trial_001", "summary": "Second", "full_line": "..."},
        ]
        result = _format_trial_table(entries)
        lines = result.strip().splitlines()
        assert len(lines) == 2
        assert "[20260301]" in lines[0]
        assert "[20260302]" in lines[1]


class TestHandleViewSummaryNoLogs:
    """Tests for handle_view_summary when no logs exist."""

    def test_no_logs_returns_decide(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface()
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_no_logs_sends_message(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface()
        handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert any("No experiment logs" in m for m in chat.sent)

    def test_none_chat_returns_decide(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), None)
        assert result == State.DECIDE


class TestHandleViewSummaryWithLogs:
    """Tests for handle_view_summary with experiment logs."""

    def test_shows_recent_trials(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("back")])
        handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        # Should show all 5 trials (under default display_trials=10)
        assert any("20260301" in m and "trial_001" in m for m in chat.sent)
        assert any("20260303" in m for m in chat.sent)

    def test_respects_display_trials_config(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        config = ResearchClawConfig(display_trials=2)
        chat = FakeChatInterface(responses=[UserMessage("back")])
        handle_view_summary(trial_dir, meta, config, chat)
        # Should show only 2 most recent trials
        assert any("2 most recent" in m for m in chat.sent)
        # Most recent first — trial_001 from 20260303 and trial_002 from 20260302
        table_msg = chat.sent[1]  # table is in second message (after status header)
        assert "20260303" in table_msg
        assert "20260302" in table_msg

    def test_back_returns_decide(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("back")])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_exit_returns_decide(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("exit")])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_q_returns_decide(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("q")])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_quit_slash_command_raises_system_exit(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[SlashCommand("/quit", "")])
        with pytest.raises(SystemExit):
            handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)

    def test_unknown_slash_command_reprompts(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            SlashCommand("/approve", ""),
            UserMessage("back"),
        ])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE
        assert any("not available" in m for m in chat.sent)

    def test_shows_total_count_when_more_than_display(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        config = ResearchClawConfig(display_trials=2)
        chat = FakeChatInterface(responses=[UserMessage("back")])
        handle_view_summary(trial_dir, meta, config, chat)
        assert any("5 total trials" in m for m in chat.sent)

    def test_date_input_shows_trials_for_date(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            UserMessage("20260301"),
            UserMessage("back"),
        ])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE
        # Should show trials for 20260301
        assert any("20260301" in m and "trial_001" in m and "trial_002" in m for m in chat.sent)

    def test_invalid_date_shows_not_found(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            UserMessage("20261231"),
            UserMessage("back"),
        ])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE
        assert any("No trials found for date 20261231" in m for m in chat.sent)

    def test_invalid_input_reprompts(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            UserMessage("hello"),
            UserMessage("back"),
        ])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE
        assert any("Type 'back'" in m for m in chat.sent)

    def test_none_chat_returns_decide(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), None)
        assert result == State.DECIDE

    def test_most_recent_first_ordering(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("back")])
        handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        # Table message is second (after status header)
        table_msg = chat.sent[1]
        pos_303 = table_msg.find("20260303")
        pos_301 = table_msg.find("20260301")
        # 20260303 should appear before 20260301 in output (most recent first)
        assert pos_303 < pos_301


class TestBrowseByDate:
    """Tests for _browse_by_date helper."""

    def _make_entries(self) -> list[dict[str, str]]:
        return [
            {"date": "20260301", "trial": "trial_001", "summary": "First", "full_line": "..."},
            {"date": "20260301", "trial": "trial_002", "summary": "Second", "full_line": "..."},
            {"date": "20260302", "trial": "trial_001", "summary": "Third", "full_line": "..."},
        ]

    def test_shows_available_dates(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[UserMessage("back")])
        _browse_by_date(entries, chat)
        assert any("20260301" in m and "20260302" in m for m in chat.sent)
        assert any("2 trial(s)" in m for m in chat.sent)

    def test_select_by_number(self) -> None:
        entries = self._make_entries()
        # Dates reversed: (1) 20260302, (2) 20260301
        chat = FakeChatInterface(responses=[UserMessage("1")])
        _browse_by_date(entries, chat)
        assert any("20260302" in m and "trial_001" in m for m in chat.sent)

    def test_select_by_date(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[UserMessage("20260301")])
        _browse_by_date(entries, chat)
        assert any("20260301" in m and "trial_001" in m for m in chat.sent)

    def test_invalid_date_reprompts(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[
            UserMessage("20261231"),
            UserMessage("back"),
        ])
        _browse_by_date(entries, chat)
        assert any("No trials found" in m for m in chat.sent)

    def test_back_returns(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[UserMessage("back")])
        _browse_by_date(entries, chat)  # Should return without error

    def test_quit_slash_raises_system_exit(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[SlashCommand("/quit", "")])
        with pytest.raises(SystemExit):
            _browse_by_date(entries, chat)

    def test_unknown_slash_reprompts(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[
            SlashCommand("/approve", ""),
            UserMessage("back"),
        ])
        _browse_by_date(entries, chat)
        assert any("Type a date" in m for m in chat.sent)

    def test_invalid_input_reprompts(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[
            UserMessage("xyz"),
            UserMessage("back"),
        ])
        _browse_by_date(entries, chat)
        assert any("Enter a date" in m for m in chat.sent)

    def test_number_out_of_range_reprompts(self) -> None:
        entries = self._make_entries()
        chat = FakeChatInterface(responses=[
            UserMessage("99"),
            UserMessage("back"),
        ])
        _browse_by_date(entries, chat)
        # 99 is not a valid date (only 8 digits) and not in range, so reprompts
        assert any("Enter a date" in m for m in chat.sent)


class TestOlderNavigation:
    """Tests for 'older' navigation in handle_view_summary."""

    def test_older_shows_dates(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        config = ResearchClawConfig(display_trials=2)
        chat = FakeChatInterface(responses=[
            UserMessage("older"),
            UserMessage("1"),  # Select first date
            UserMessage("back"),
        ])
        result = handle_view_summary(trial_dir, meta, config, chat)
        assert result == State.DECIDE
        assert any("Available experiment dates" in m for m in chat.sent)

    def test_older_then_back(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, SAMPLE_LOG_LINES)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            UserMessage("older"),
            UserMessage("back"),  # Back from browse_by_date
            UserMessage("back"),  # Back from view_summary
        ])
        result = handle_view_summary(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE


class TestDisplayTrialsDefault:
    """Tests for DEFAULT_DISPLAY_TRIALS constant."""

    def test_default_is_10(self) -> None:
        assert DEFAULT_DISPLAY_TRIALS == 10

    def test_config_default_is_10(self) -> None:
        config = ResearchClawConfig()
        assert config.display_trials == 10
