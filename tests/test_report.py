"""Tests for the EXPERIMENT_REPORT state handler."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand
from researchclaw.sandbox import SandboxManager

import researchclaw.fsm.report as report_mod
from researchclaw.fsm.report import (
    SUMMARY_AGENT_SYSTEM,
    SUMMARY_LOG_SYSTEM,
    _append_experiment_log,
    _gather_prior_reports,
    _gather_trial_content,
    _generate_log_summary_fallback,
    _generate_log_summary_llm,
    _generate_report_fallback,
    _generate_report_llm,
    _write_report,
    handle_experiment_report,
)


class FakeChatInterface:
    """Fake chat interface for testing."""

    def __init__(self, responses: list[ChatInput] | None = None) -> None:
        self.messages: list[str] = []
        self._responses = list(responses) if responses else []

    def send(self, message: str) -> None:
        self.messages.append(message)

    def receive(self) -> ChatInput:
        if not self._responses:
            raise SystemExit("No more responses")
        return self._responses.pop(0)


class FakeProvider:
    """Fake LLM provider for testing."""

    def __init__(
        self,
        responses: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        self.calls.append({"messages": list(messages), "system": system})
        if self._error is not None:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return "LLM response"


def _setup_trial(tmp_path: Path) -> Path:
    """Create a sandbox and trial for testing."""
    SandboxManager.initialize(tmp_path)
    trial_dir = SandboxManager.create_trial(tmp_path)
    return trial_dir


def _populate_trial(trial_dir: Path) -> None:
    """Populate a trial directory with sample content."""
    (trial_dir / "PLAN.md").write_text("# Test Plan\n\nTest experiment.")
    (trial_dir / "experiment" / "codes_exp" / "main.py").write_text(
        "print('hello experiment')"
    )
    (trial_dir / "experiment" / "outputs" / "log_iter000").write_text(
        "hello experiment\n"
    )
    (trial_dir / "experiment" / "codes_eval" / "main.py").write_text(
        "print('hello eval')"
    )
    (trial_dir / "experiment" / "outputs" / "eval_log_iter000").write_text(
        "eval result: ok\n"
    )


# --- Tests for _gather_trial_content ---

class TestGatherTrialContent:

    def test_empty_trial(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        content = _gather_trial_content(trial_dir)
        assert content["plan"] == "No plan available."
        assert content["experiment_code"] == "No experiment code."
        assert content["visualizations"] == "No visualizations."

    def test_with_plan(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        (trial_dir / "PLAN.md").write_text("# My Plan\nDo stuff.")
        content = _gather_trial_content(trial_dir)
        assert "My Plan" in content["plan"]

    def test_with_experiment_code(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        (trial_dir / "experiment" / "codes_exp" / "main.py").write_text("print('hi')")
        content = _gather_trial_content(trial_dir)
        assert "main.py" in content["experiment_code"]
        assert "print('hi')" in content["experiment_code"]

    def test_with_experiment_outputs(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        (trial_dir / "experiment" / "outputs" / "log_iter000").write_text("output here")
        content = _gather_trial_content(trial_dir)
        assert "log_iter000" in content["experiment_outputs"]
        assert "output here" in content["experiment_outputs"]

    def test_with_eval_code(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        (trial_dir / "experiment" / "codes_eval" / "main.py").write_text("eval code")
        content = _gather_trial_content(trial_dir)
        assert "eval code" in content["eval_code"]

    def test_with_eval_outputs(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        (trial_dir / "experiment" / "outputs" / "eval_log_iter000").write_text("eval out")
        content = _gather_trial_content(trial_dir)
        assert "eval out" in content["eval_outputs"]

    def test_with_visualizations(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        (trial_dir / "result.png").write_text("fake png")
        (trial_dir / "video.mp4").write_text("fake mp4")
        content = _gather_trial_content(trial_dir)
        assert "result.png" in content["visualizations"]
        assert "video.mp4" in content["visualizations"]

    def test_ignores_non_viz_files(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        (trial_dir / "meta.json").write_text("{}")  # Already exists, but just ensure
        content = _gather_trial_content(trial_dir)
        assert "meta.json" not in content["visualizations"]

    def test_truncates_long_output(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        long_output = "x" * 10000
        (trial_dir / "experiment" / "outputs" / "log_iter000").write_text(long_output)
        content = _gather_trial_content(trial_dir)
        assert "[... truncated]" in content["experiment_outputs"]

    def test_no_outputs_dir(self, tmp_path: Path) -> None:
        """When outputs dir doesn't exist, returns default strings."""
        trial_dir = _setup_trial(tmp_path)
        # Remove the outputs dir
        import shutil
        outputs_dir = trial_dir / "experiment" / "outputs"
        shutil.rmtree(outputs_dir)
        content = _gather_trial_content(trial_dir)
        assert content["experiment_outputs"] == "No experiment outputs."
        assert content["eval_outputs"] == "No evaluation outputs."

    def test_empty_codes_dir(self, tmp_path: Path) -> None:
        """Empty codes dirs return default strings."""
        trial_dir = _setup_trial(tmp_path)
        content = _gather_trial_content(trial_dir)
        # codes_exp and codes_eval exist but are empty
        assert content["experiment_code"] == "No experiment code."
        assert content["eval_code"] == "No evaluation code."


# --- Tests for _gather_prior_reports ---

class TestGatherPriorReports:

    def test_no_prior_trials(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        project_dir = trial_dir.parent.parent.parent
        result = _gather_prior_reports(project_dir, trial_dir.name)
        assert result == "No prior trials."

    def test_with_prior_trial_report(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial1 = SandboxManager.create_trial(tmp_path)
        (trial1 / "REPORT.md").write_text("# Report 1\nGood results.")
        trial2 = SandboxManager.create_trial(tmp_path)
        result = _gather_prior_reports(tmp_path, trial2.name)
        assert "Report 1" in result
        assert trial1.name in result

    def test_excludes_current_trial(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial1 = SandboxManager.create_trial(tmp_path)
        (trial1 / "REPORT.md").write_text("Report for trial 1")
        result = _gather_prior_reports(tmp_path, trial1.name)
        assert result == "No prior trials."

    def test_truncates_long_prior_report(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial1 = SandboxManager.create_trial(tmp_path)
        (trial1 / "REPORT.md").write_text("x" * 5000)
        trial2 = SandboxManager.create_trial(tmp_path)
        result = _gather_prior_reports(tmp_path, trial2.name)
        assert "[... truncated]" in result

    def test_no_experiments_dir(self, tmp_path: Path) -> None:
        result = _gather_prior_reports(tmp_path, "fake_trial")
        assert result == "No prior trials."

    def test_prior_trials_without_reports(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        SandboxManager.create_trial(tmp_path)
        trial2 = SandboxManager.create_trial(tmp_path)
        result = _gather_prior_reports(tmp_path, trial2.name)
        assert result == "No prior trial reports."


# --- Tests for _write_report ---

class TestWriteReport:

    def test_writes_report(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        _write_report(trial_dir, "# Report\nResults here.", terminated=False)
        content = (trial_dir / "REPORT.md").read_text()
        assert content == "# Report\nResults here."

    def test_terminated_report_has_marker(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        _write_report(trial_dir, "# Report\nResults here.", terminated=True)
        content = (trial_dir / "REPORT.md").read_text()
        assert content.startswith("[TERMINATED-DURING-EXPERIMENT]")
        assert "Results here." in content

    def test_non_terminated_no_marker(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        _write_report(trial_dir, "# Report", terminated=False)
        content = (trial_dir / "REPORT.md").read_text()
        assert "[TERMINATED-DURING-EXPERIMENT]" not in content


# --- Tests for _append_experiment_log ---

class TestAppendExperimentLog:

    def test_appends_entry(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        _append_experiment_log(tmp_path, trial_dir, "Good results found")
        logs = (SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md").read_text()
        assert "Good results found" in logs
        assert trial_dir.name in logs
        assert "REPORT.md" in logs

    def test_log_format(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)
        _append_experiment_log(tmp_path, trial_dir, "Summary text")
        logs = (SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md").read_text()
        # Format: {YYYYMMDD} - trial_{N:03}: {summary}. Full Doc: [REPORT.md](...)
        parts = trial_dir.name.split("_trial_")
        date_str = parts[0]
        trial_num = parts[1]
        assert f"{date_str} - trial_{trial_num}" in logs
        assert f"[REPORT.md](experiments/{trial_dir.name}/REPORT.md)" in logs

    def test_multiple_appends(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial1 = SandboxManager.create_trial(tmp_path)
        trial2 = SandboxManager.create_trial(tmp_path)
        _append_experiment_log(tmp_path, trial1, "First trial summary")
        _append_experiment_log(tmp_path, trial2, "Second trial summary")
        logs = (SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md").read_text()
        assert "First trial summary" in logs
        assert "Second trial summary" in logs
        lines = [l for l in logs.strip().splitlines() if l.strip()]
        assert len(lines) == 2


# --- Tests for _generate_report_fallback ---

class TestGenerateReportFallback:

    def test_contains_trial_info(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(trial_number=1, experiment_exit_code=0, eval_exit_code=0)
        content = {"plan": "# Plan", "experiment_code": "code", "experiment_outputs": "output",
                   "eval_code": "eval", "eval_outputs": "eval out", "visualizations": "plot.png"}
        report = _generate_report_fallback(trial_dir, meta, content)
        assert trial_dir.name in report
        assert "Trial Number" in report
        assert "Plan" in report

    def test_contains_outputs(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        content = {"plan": "plan", "experiment_code": "code",
                   "experiment_outputs": "EXP OUTPUT HERE",
                   "eval_code": "eval code", "eval_outputs": "EVAL OUTPUT HERE",
                   "visualizations": "No visualizations."}
        report = _generate_report_fallback(trial_dir, meta, content)
        assert "EXP OUTPUT HERE" in report
        assert "EVAL OUTPUT HERE" in report

    def test_contains_future_directions(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        content = {"plan": "", "experiment_code": "", "experiment_outputs": "",
                   "eval_code": "", "eval_outputs": "", "visualizations": ""}
        report = _generate_report_fallback(trial_dir, meta, content)
        assert "Future Directions" in report


# --- Tests for _generate_log_summary_fallback ---

class TestGenerateLogSummaryFallback:

    def test_includes_plan_snippet(self) -> None:
        meta = TrialMeta(experiment_exit_code=0, eval_exit_code=0)
        content = {"plan": "# Test Plan\nDo cool stuff."}
        summary = _generate_log_summary_fallback(meta, content)
        assert "Test Plan" in summary

    def test_includes_exit_codes(self) -> None:
        meta = TrialMeta(experiment_exit_code=0, eval_exit_code=1)
        content = {"plan": "plan"}
        summary = _generate_log_summary_fallback(meta, content)
        assert "exit=0" in summary
        assert "exit=1" in summary

    def test_truncates_long_plan(self) -> None:
        meta = TrialMeta()
        content = {"plan": "x" * 300}
        summary = _generate_log_summary_fallback(meta, content)
        assert len(summary) < 400  # 150 chars plan + exit info


# --- Tests for _generate_report_llm ---

class TestGenerateReportLLM:

    def test_calls_provider(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(trial_number=1)
        provider = FakeProvider(responses=["# LLM Report\nGreat results."])
        content = {"plan": "plan", "experiment_code": "code",
                   "experiment_outputs": "output", "eval_code": "eval",
                   "eval_outputs": "eval out", "visualizations": "none"}
        result = _generate_report_llm(trial_dir, meta, provider, content, "No prior trials.")
        assert result == "# LLM Report\nGreat results."
        assert len(provider.calls) == 1

    def test_system_prompt_contains_trial_info(self, tmp_path: Path) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(trial_number=3)
        provider = FakeProvider(responses=["report"])
        content = {"plan": "MY PLAN", "experiment_code": "MY CODE",
                   "experiment_outputs": "MY OUTPUT", "eval_code": "MY EVAL",
                   "eval_outputs": "MY EVAL OUT", "visualizations": "plot.png"}
        _generate_report_llm(trial_dir, meta, provider, content, "Prior stuff")
        system = provider.calls[0]["system"]
        assert "MY PLAN" in system
        assert "MY CODE" in system
        assert "MY OUTPUT" in system
        assert "Prior stuff" in system
        assert trial_dir.name in system


# --- Tests for _generate_log_summary_llm ---

class TestGenerateLogSummaryLLM:

    def test_calls_provider(self) -> None:
        provider = FakeProvider(responses=["Short summary of results."])
        result = _generate_log_summary_llm(provider, "# Long Report\nblah blah")
        assert result == "Short summary of results."
        assert len(provider.calls) == 1

    def test_truncates_to_3_lines(self) -> None:
        provider = FakeProvider(responses=["Line 1\nLine 2\nLine 3\nLine 4\nLine 5"])
        result = _generate_log_summary_llm(provider, "report")
        # Should only include first 3 non-empty lines joined by space
        assert result == "Line 1 Line 2 Line 3"

    def test_skips_empty_lines(self) -> None:
        provider = FakeProvider(responses=["\n\nLine 1\n\nLine 2\n\n"])
        result = _generate_log_summary_llm(provider, "report")
        assert result == "Line 1 Line 2"

    def test_system_prompt(self) -> None:
        provider = FakeProvider(responses=["summary"])
        _generate_log_summary_llm(provider, "report content")
        system = provider.calls[0]["system"]
        assert "2-3 lines" in system


# --- Tests for handle_experiment_report ---

class TestHandleExperimentReport:

    def test_returns_decide(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        result = handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert result == State.DECIDE

    def test_writes_report_md(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        _populate_trial(trial_dir)
        meta = TrialMeta()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert (trial_dir / "REPORT.md").exists()
        content = (trial_dir / "REPORT.md").read_text()
        assert "Trial Report" in content

    def test_updates_experiment_logs(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        logs_path = SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md"
        logs = logs_path.read_text()
        assert trial_dir.name in logs
        assert "REPORT.md" in logs

    def test_sends_messages(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), chat)
        assert any("EXPERIMENT_REPORT" in m for m in chat.messages)
        assert any("REPORT.md written" in m for m in chat.messages)
        assert any("EXPERIMENT_LOGS.md updated" in m for m in chat.messages)

    def test_none_chat_interface(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        result = handle_experiment_report(trial_dir, meta, ResearchClawConfig(), None)
        assert result == State.DECIDE
        assert (trial_dir / "REPORT.md").exists()

    def test_terminated_trial(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(status="terminated")
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        content = (trial_dir / "REPORT.md").read_text()
        assert content.startswith("[TERMINATED-DURING-EXPERIMENT]")

    def test_non_terminated_trial(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(status="running")
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        content = (trial_dir / "REPORT.md").read_text()
        assert not content.startswith("[TERMINATED-DURING-EXPERIMENT]")

    def test_with_llm_provider(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        _populate_trial(trial_dir)
        meta = TrialMeta()
        provider = FakeProvider(responses=[
            "# LLM Generated Report\nGreat findings here.",  # Report
            "Trial tested plan. Results were good.",  # Log summary
        ])
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: provider)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        content = (trial_dir / "REPORT.md").read_text()
        assert "LLM Generated Report" in content
        logs = (SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md").read_text()
        assert "Trial tested plan" in logs

    def test_llm_error_falls_back(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        provider = FakeProvider(error=RuntimeError("LLM down"))
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: provider)
        chat = FakeChatInterface()
        result = handle_experiment_report(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE
        # Should have fallen back
        content = (trial_dir / "REPORT.md").read_text()
        assert "Trial Report" in content
        assert any("LLM error" in m for m in chat.messages)

    def test_sets_status_completed(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta(status="running")
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert meta.status == "completed"

    def test_report_contains_experiment_outputs(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        _populate_trial(trial_dir)
        meta = TrialMeta()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        content = (trial_dir / "REPORT.md").read_text()
        assert "Experiment Outputs" in content

    def test_do_not_interrupt_message(self, tmp_path: Path, monkeypatch: Any) -> None:
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: None)
        handle_experiment_report(trial_dir, meta, ResearchClawConfig(), chat)
        assert any("do not interrupt" in m for m in chat.messages)

    def test_with_prior_trials(self, tmp_path: Path, monkeypatch: Any) -> None:
        SandboxManager.initialize(tmp_path)
        trial1 = SandboxManager.create_trial(tmp_path)
        (trial1 / "REPORT.md").write_text("# Prior Report\nPrior findings.")
        trial2 = SandboxManager.create_trial(tmp_path)
        meta = TrialMeta()
        provider = FakeProvider(responses=[
            "# Report with comparison\nImproved over prior.",
            "Summary with comparison.",
        ])
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: provider)
        handle_experiment_report(trial2, meta, ResearchClawConfig(), FakeChatInterface())
        # The provider should have been called with prior report context in system prompt
        system = provider.calls[0]["system"]
        assert "Prior Report" in system or "Prior findings" in system

    def test_log_summary_llm_error_falls_back(self, tmp_path: Path, monkeypatch: Any) -> None:
        """When LLM fails on log summary but succeeds on report, falls back for summary."""
        trial_dir = _setup_trial(tmp_path)
        meta = TrialMeta()
        call_count = 0

        class ErrorOnSecondCall:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self._call_count = 0

            def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
                self._call_count += 1
                self.calls.append({"messages": list(messages), "system": system})
                if self._call_count == 1:
                    return "# Report Content"
                raise RuntimeError("Summary LLM fail")

        provider = ErrorOnSecondCall()
        monkeypatch.setattr(report_mod, "_get_provider_safe", lambda cfg: provider)
        result = handle_experiment_report(trial_dir, meta, ResearchClawConfig(), FakeChatInterface())
        assert result == State.DECIDE
        # Report should be from LLM
        content = (trial_dir / "REPORT.md").read_text()
        assert "Report Content" in content
        # Log summary should be fallback
        logs = (SandboxManager.sandbox_path(tmp_path) / "EXPERIMENT_LOGS.md").read_text()
        assert "exit=" in logs


# --- Tests for system prompt content ---

class TestSystemPrompts:

    def test_summary_agent_system_has_placeholders(self) -> None:
        assert "{trial_name}" in SUMMARY_AGENT_SYSTEM
        assert "{plan_content}" in SUMMARY_AGENT_SYSTEM
        assert "{experiment_code}" in SUMMARY_AGENT_SYSTEM
        assert "{experiment_outputs}" in SUMMARY_AGENT_SYSTEM
        assert "{eval_code}" in SUMMARY_AGENT_SYSTEM
        assert "{eval_outputs}" in SUMMARY_AGENT_SYSTEM
        assert "{prior_reports}" in SUMMARY_AGENT_SYSTEM
        assert "{visualizations}" in SUMMARY_AGENT_SYSTEM

    def test_summary_agent_instructions(self) -> None:
        assert "What was done" in SUMMARY_AGENT_SYSTEM
        assert "Results summary" in SUMMARY_AGENT_SYSTEM
        assert "Comparison with prior trials" in SUMMARY_AGENT_SYSTEM
        assert "Future directions" in SUMMARY_AGENT_SYSTEM

    def test_summary_log_system(self) -> None:
        assert "2-3 lines" in SUMMARY_LOG_SYSTEM
        assert "concise" in SUMMARY_LOG_SYSTEM
