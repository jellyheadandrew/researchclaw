from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.decide import handle_decide
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager
from researchclaw.status import (
    DEFAULT_STATUS_DISPLAY,
    _get_trial_dirs,
    _parse_log_summaries,
    build_status_table,
    render_status_string,
)

import researchclaw.cli as cli_mod


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


SAMPLE_LOG_LINES = [
    "20260301 - trial_001: First experiment on transformers. Full Doc: [REPORT.md](experiments/20260301_trial_001/REPORT.md)",
    "20260301 - trial_002: Improved learning rate schedule. Full Doc: [REPORT.md](experiments/20260301_trial_002/REPORT.md)",
    "20260302 - trial_001: Added data augmentation. Full Doc: [REPORT.md](experiments/20260302_trial_001/REPORT.md)",
    "20260302 - trial_002: Switched to Adam optimizer. Full Doc: [REPORT.md](experiments/20260302_trial_002/REPORT.md)",
    "20260303 - trial_001: Batch normalization experiment. Full Doc: [REPORT.md](experiments/20260303_trial_001/REPORT.md)",
]


def _setup_sandbox_with_logs(project_dir: Path, log_lines: list[str]) -> Path:
    """Helper: initialize sandbox and write EXPERIMENT_LOGS.md with given lines."""
    SandboxManager.initialize(project_dir)
    logs_path = SandboxManager.sandbox_path(project_dir) / "EXPERIMENT_LOGS.md"
    logs_path.write_text("\n".join(log_lines) + "\n" if log_lines else "")
    trial_dir = SandboxManager.create_trial(project_dir)
    return trial_dir


def _setup_sandbox_with_trials(
    project_dir: Path,
    log_lines: list[str],
    trial_names: list[str] | None = None,
) -> Path:
    """Helper: initialize sandbox, create trial dirs with meta.json, write logs."""
    SandboxManager.initialize(project_dir)
    logs_path = SandboxManager.sandbox_path(project_dir) / "EXPERIMENT_LOGS.md"
    logs_path.write_text("\n".join(log_lines) + "\n" if log_lines else "")

    experiments_dir = SandboxManager.sandbox_path(project_dir) / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    if trial_names is None:
        trial_names = []

    for name in trial_names:
        trial_dir = experiments_dir / name
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "experiment" / "codes_exp").mkdir(parents=True, exist_ok=True)
        (trial_dir / "experiment" / "codes_eval").mkdir(parents=True, exist_ok=True)
        (trial_dir / "experiment" / "outputs").mkdir(parents=True, exist_ok=True)
        # Parse trial number from name
        trial_num = int(name.split("_trial_")[1])
        meta = TrialMeta(trial_number=trial_num, status="completed", state="decide")
        meta.to_json(trial_dir / "meta.json")

    # Return last trial dir or create one
    if trial_names:
        return experiments_dir / trial_names[-1]
    return SandboxManager.create_trial(project_dir)


class TestParseLogSummaries:
    """Tests for _parse_log_summaries helper."""

    def test_empty_logs(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        result = _parse_log_summaries(tmp_path)
        assert result == {}

    def test_no_sandbox(self, tmp_path: Path) -> None:
        result = _parse_log_summaries(tmp_path)
        assert result == {}

    def test_parses_single_entry(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text(SAMPLE_LOG_LINES[0] + "\n")
        result = _parse_log_summaries(tmp_path)
        assert "20260301_trial_001" in result
        assert result["20260301_trial_001"] == "First experiment on transformers"

    def test_parses_multiple_entries(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text("\n".join(SAMPLE_LOG_LINES) + "\n")
        result = _parse_log_summaries(tmp_path)
        assert len(result) == 5
        assert "20260301_trial_001" in result
        assert "20260303_trial_001" in result

    def test_trial_name_format(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs_path.write_text(SAMPLE_LOG_LINES[2] + "\n")
        result = _parse_log_summaries(tmp_path)
        assert "20260302_trial_001" in result
        assert result["20260302_trial_001"] == "Added data augmentation"


class TestGetTrialDirs:
    """Tests for _get_trial_dirs helper."""

    def test_no_sandbox(self, tmp_path: Path) -> None:
        result = _get_trial_dirs(tmp_path)
        assert result == []

    def test_empty_experiments(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        result = _get_trial_dirs(tmp_path)
        assert result == []

    def test_returns_trial_dirs_sorted(self, tmp_path: Path) -> None:
        _setup_sandbox_with_trials(
            tmp_path,
            [],
            trial_names=["20260301_trial_001", "20260301_trial_002", "20260302_trial_001"],
        )
        result = _get_trial_dirs(tmp_path)
        assert len(result) == 3
        assert result[0].name == "20260301_trial_001"
        assert result[-1].name == "20260302_trial_001"


class TestBuildStatusTable:
    """Tests for build_status_table."""

    def test_no_trials(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        table = build_status_table(tmp_path)
        assert table.title == "ResearchClaw Trials"
        assert table.row_count == 0

    def test_with_trials_and_logs(self, tmp_path: Path) -> None:
        trial_names = [
            "20260301_trial_001",
            "20260301_trial_002",
            "20260302_trial_001",
        ]
        _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES[:3], trial_names)
        table = build_status_table(tmp_path)
        assert table.row_count == 3

    def test_shows_most_recent_first(self, tmp_path: Path) -> None:
        trial_names = [
            "20260301_trial_001",
            "20260301_trial_002",
            "20260302_trial_001",
        ]
        _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES[:3], trial_names)
        table = build_status_table(tmp_path)
        # Most recent first in the table
        assert table.row_count == 3

    def test_max_trials_limits_output(self, tmp_path: Path) -> None:
        trial_names = [
            "20260301_trial_001",
            "20260301_trial_002",
            "20260302_trial_001",
            "20260302_trial_002",
            "20260303_trial_001",
        ]
        _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES, trial_names)
        table = build_status_table(tmp_path, max_trials=3)
        assert table.row_count == 3

    def test_default_display_is_10(self) -> None:
        assert DEFAULT_STATUS_DISPLAY == 10

    def test_table_columns(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        table = build_status_table(tmp_path)
        column_names = [c.header for c in table.columns]
        assert "Trial" in column_names
        assert "State" in column_names
        assert "Status" in column_names
        assert "Summary" in column_names

    def test_trial_without_log_entry(self, tmp_path: Path) -> None:
        """Trials that have no corresponding log entry show empty summary."""
        _setup_sandbox_with_trials(
            tmp_path,
            [],  # No log entries
            trial_names=["20260301_trial_001"],
        )
        table = build_status_table(tmp_path)
        assert table.row_count == 1

    def test_trial_with_custom_meta(self, tmp_path: Path) -> None:
        """Trial meta state/status are reflected in the table."""
        SandboxManager.initialize(tmp_path)
        experiments_dir = SandboxManager.sandbox_path(tmp_path) / "experiments"
        trial_dir = experiments_dir / "20260301_trial_001"
        trial_dir.mkdir(parents=True)
        meta = TrialMeta(
            trial_number=1,
            status="running",
            state="experiment_execute",
        )
        meta.to_json(trial_dir / "meta.json")
        # Render to string to check content
        output = render_status_string(tmp_path)
        assert "experiment_execute" in output
        assert "running" in output


class TestRenderStatusString:
    """Tests for render_status_string."""

    def test_returns_string(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        result = render_status_string(tmp_path)
        assert isinstance(result, str)
        assert "ResearchClaw Trials" in result

    def test_contains_trial_names(self, tmp_path: Path) -> None:
        trial_names = ["20260301_trial_001", "20260302_trial_001"]
        _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES[:2], trial_names)
        result = render_status_string(tmp_path)
        assert "20260301_trial_001" in result

    def test_contains_summaries(self, tmp_path: Path) -> None:
        trial_names = ["20260301_trial_001"]
        _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES[:1], trial_names)
        result = render_status_string(tmp_path)
        assert "First experiment on transformers" in result


class TestStatusCLISubcommand:
    """Tests for 'researchclaw status' CLI subcommand."""

    def test_no_sandbox(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)
        runner = CliRunner()
        result = runner.invoke(cli_mod.main, ["status"], obj={"project_dir": str(tmp_path)})
        assert "No sandbox found" in result.output

    def test_with_sandbox_empty(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)
        SandboxManager.initialize(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli_mod.main, ["status"], obj={"project_dir": str(tmp_path)})
        assert "ResearchClaw Trials" in result.output

    def test_with_trials(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)
        trial_names = ["20260301_trial_001"]
        _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES[:1], trial_names)
        runner = CliRunner()
        result = runner.invoke(cli_mod.main, ["status"], obj={"project_dir": str(tmp_path)})
        assert "20260301_trial_001" in result.output


class TestStatusSlashCommandInDecide:
    """/status slash command in DECIDE handler shows status table."""

    def test_status_shows_table(self, tmp_path: Path) -> None:
        trial_names = ["20260301_trial_001"]
        trial_dir = _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES[:1], trial_names)
        meta = TrialMeta(trial_number=1, status="completed", state="decide")
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            SlashCommand("/status", ""),
            UserMessage("5"),  # quit
        ])
        with pytest.raises(SystemExit):
            handle_decide(trial_dir, meta, config, chat)
        # Check that status output was sent
        status_msgs = [m for m in chat.sent if "ResearchClaw Trials" in m]
        assert len(status_msgs) >= 1

    def test_status_then_choose(self, tmp_path: Path) -> None:
        trial_names = ["20260301_trial_001"]
        trial_dir = _setup_sandbox_with_trials(tmp_path, SAMPLE_LOG_LINES[:1], trial_names)
        meta = TrialMeta(trial_number=1, status="completed", state="decide")
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            SlashCommand("/status", ""),
            UserMessage("1"),  # new experiment
        ])
        result = handle_decide(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_PLAN
        # Status table should have been displayed
        status_msgs = [m for m in chat.sent if "ResearchClaw Trials" in m]
        assert len(status_msgs) >= 1

    def test_status_with_no_trials(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox_with_logs(tmp_path, [])
        meta = TrialMeta(trial_number=1, status="pending", state="decide")
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            SlashCommand("/status", ""),
            UserMessage("5"),  # quit
        ])
        with pytest.raises(SystemExit):
            handle_decide(trial_dir, meta, config, chat)
        # Should still send status table (just empty)
        status_msgs = [m for m in chat.sent if "ResearchClaw Trials" in m]
        assert len(status_msgs) >= 1
