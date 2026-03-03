from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.plan import (
    HISTORIAN_AGENT_SYSTEM,
    PLANNING_AGENT_SYSTEM,
    _build_historian_context,
    _default_plan,
    _gather_trial_history,
    _get_search_context,
    _write_plan,
    handle_experiment_plan,
)
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager

import researchclaw.fsm.plan as plan_mod


# --- Fake ChatInterface for testing ---

class FakeChat:
    """Fake chat interface with pre-programmed responses."""

    def __init__(self, responses: list[ChatInput]) -> None:
        self.responses = list(responses)
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def send_image(self, path: str, caption: str | None = None) -> None:
        pass

    def receive(self) -> ChatInput:
        if not self.responses:
            raise SystemExit("No more responses")
        return self.responses.pop(0)


# --- Fake LLM Provider ---

class FakeProvider:
    """Fake LLM provider that returns pre-configured responses."""

    def __init__(self, responses: list[str] | None = None, error: Exception | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        self.calls.append({"messages": list(messages), "system": system})
        if self._error:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return "Mock LLM response\n\n**If you approve plan, please type /approve. If not, please iterate.**"

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Any:
        yield self.chat(messages, system)


def _setup_sandbox(project_dir: Path) -> Path:
    """Initialize sandbox and create a trial, return trial dir."""
    SandboxManager.initialize(project_dir)
    return SandboxManager.create_trial(project_dir)


# --- Tests for _gather_trial_history ---

class TestGatherTrialHistory:
    def test_empty_when_no_experiments(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        result = _gather_trial_history(tmp_path)
        assert result == ""

    def test_empty_when_no_reports(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        result = _gather_trial_history(tmp_path)
        assert result == ""

    def test_includes_report_content(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "REPORT.md").write_text("# Trial 1 Report\nGood results.")
        result = _gather_trial_history(tmp_path)
        assert "Trial 1 Report" in result
        assert "Good results" in result
        assert trial_dir.name in result

    def test_includes_experiment_logs(self, tmp_path: Path) -> None:
        _setup_sandbox(tmp_path)
        sandbox = SandboxManager.sandbox_path(tmp_path)
        (sandbox / "EXPERIMENT_LOGS.md").write_text("20260302 - trial_001: first trial.")
        result = _gather_trial_history(tmp_path)
        assert "20260302 - trial_001" in result
        assert "Experiment Logs" in result

    def test_multiple_trials(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        t1 = SandboxManager.create_trial(tmp_path)
        t2 = SandboxManager.create_trial(tmp_path)
        (t1 / "REPORT.md").write_text("Report A")
        (t2 / "REPORT.md").write_text("Report B")
        result = _gather_trial_history(tmp_path)
        assert "Report A" in result
        assert "Report B" in result

    def test_skips_empty_reports(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "REPORT.md").write_text("")
        result = _gather_trial_history(tmp_path)
        assert result == ""

    def test_no_experiments_dir(self, tmp_path: Path) -> None:
        result = _gather_trial_history(tmp_path)
        assert result == ""


# --- Tests for _build_historian_context ---

class TestBuildHistorianContext:
    def test_no_history_returns_message(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        result = _build_historian_context(tmp_path)
        assert result == "No prior trials."

    def test_short_history_returned_directly(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "REPORT.md").write_text("Short report.")
        result = _build_historian_context(tmp_path)
        assert "Short report" in result

    def test_long_history_summarized_by_llm(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        long_report = "A" * 4000
        (trial_dir / "REPORT.md").write_text(long_report)
        provider = FakeProvider(responses=["Summarized history"])
        result = _build_historian_context(tmp_path, provider=provider)
        assert result == "Summarized history"
        assert len(provider.calls) == 1
        assert HISTORIAN_AGENT_SYSTEM in provider.calls[0]["system"]

    def test_long_history_truncated_without_provider(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        long_report = "A" * 4000
        (trial_dir / "REPORT.md").write_text(long_report)
        result = _build_historian_context(tmp_path, provider=None)
        assert "[... truncated]" in result
        assert len(result) < 4000

    def test_long_history_truncated_on_llm_error(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        long_report = "A" * 4000
        (trial_dir / "REPORT.md").write_text(long_report)
        provider = FakeProvider(error=RuntimeError("LLM failed"))
        result = _build_historian_context(tmp_path, provider=provider)
        assert "[... truncated]" in result


# --- Tests for _get_search_context ---

class TestGetSearchContext:
    def test_returns_empty_string(self) -> None:
        assert _get_search_context() == ""


# --- Tests for _write_plan ---

class TestWritePlan:
    def test_writes_plan_file(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, "# My Plan\nDo stuff.")
        assert (tmp_path / "PLAN.md").read_text() == "# My Plan\nDo stuff."


# --- Tests for _default_plan ---

class TestDefaultPlan:
    def test_empty_messages(self) -> None:
        result = _default_plan([])
        assert "No plan content" in result
        assert "Approved without discussion" in result

    def test_with_messages(self) -> None:
        messages = [
            {"role": "user", "content": "test hypothesis"},
            {"role": "assistant", "content": "good idea"},
        ]
        result = _default_plan(messages)
        assert "test hypothesis" in result
        assert "good idea" in result
        assert "**User**:" in result
        assert "**Assistant**:" in result


# --- Tests for handle_experiment_plan (interactive mode) ---

class TestInteractivePlan:
    def test_approve_immediately(self, tmp_path: Path, monkeypatch: object) -> None:
        """User types /approve immediately — writes default plan, returns EXPERIMENT_IMPLEMENT."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([SlashCommand("/approve", "")])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert (trial_dir / "PLAN.md").exists()
        assert meta.plan_approved_at is not None
        assert meta.status == "running"
        assert any("Plan approved" in m for m in chat.sent)

    def test_chat_then_approve(self, tmp_path: Path, monkeypatch: object) -> None:
        """User sends a message, gets LLM response, then approves."""
        trial_dir = _setup_sandbox(tmp_path)
        provider = FakeProvider(responses=["# Experiment Plan\nTest the hypothesis."])
        chat = FakeChat([
            UserMessage("I want to test image classification"),
            SlashCommand("/approve", ""),
        ])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        plan_content = (trial_dir / "PLAN.md").read_text()
        assert "Test the hypothesis" in plan_content
        assert len(provider.calls) == 1
        assert provider.calls[0]["messages"][0]["content"] == "I want to test image classification"

    def test_multiple_chat_rounds(self, tmp_path: Path, monkeypatch: object) -> None:
        """User has multiple exchanges before approving."""
        trial_dir = _setup_sandbox(tmp_path)
        provider = FakeProvider(responses=[
            "First response about classification",
            "# Final Plan\nRevised approach.",
        ])
        chat = FakeChat([
            UserMessage("image classification"),
            UserMessage("let's revise"),
            SlashCommand("/approve", ""),
        ])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        plan_content = (trial_dir / "PLAN.md").read_text()
        assert "Revised approach" in plan_content
        assert len(provider.calls) == 2
        # Second call should include the full conversation history: user, assistant, user
        # Note: by the time the second chat() is called, the first response has also been appended
        assert len(provider.calls[1]["messages"]) == 3  # user, assistant, user

    def test_quit_during_planning(self, tmp_path: Path, monkeypatch: object) -> None:
        """User types /quit — raises SystemExit."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([SlashCommand("/quit", "")])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        try:
            handle_experiment_plan(trial_dir, meta, config, chat)
            assert False, "Expected SystemExit"
        except SystemExit:
            pass

    def test_unknown_slash_command(self, tmp_path: Path, monkeypatch: object) -> None:
        """Unknown slash command during planning shows info message."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([
            SlashCommand("/status", ""),
            SlashCommand("/approve", ""),
        ])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert any("not available during planning" in m for m in chat.sent)

    def test_no_provider_still_works(self, tmp_path: Path, monkeypatch: object) -> None:
        """Without an LLM provider, user can still chat and approve."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([
            UserMessage("my plan idea"),
            SlashCommand("/approve", ""),
        ])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert any("No LLM provider" in m for m in chat.sent)

    def test_llm_error_handled(self, tmp_path: Path, monkeypatch: object) -> None:
        """LLM error during chat is handled gracefully."""
        trial_dir = _setup_sandbox(tmp_path)
        provider = FakeProvider(error=RuntimeError("Connection failed"))
        chat = FakeChat([
            UserMessage("test idea"),
            SlashCommand("/approve", ""),
        ])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert any("LLM error" in m for m in chat.sent)

    def test_no_chat_interface(self, tmp_path: Path, monkeypatch: object) -> None:
        """With None chat interface, writes empty plan and advances."""
        trial_dir = _setup_sandbox(tmp_path)
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, None)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert (trial_dir / "PLAN.md").exists()
        assert meta.plan_approved_at is not None

    def test_first_trial_greeting(self, tmp_path: Path, monkeypatch: object) -> None:
        """First trial shows 'first trial' greeting."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([SlashCommand("/approve", "")])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        handle_experiment_plan(trial_dir, meta, config, chat)

        assert any("first trial" in m for m in chat.sent)

    def test_prior_trials_greeting(self, tmp_path: Path, monkeypatch: object) -> None:
        """With prior trial history, shows 'prior trials' greeting."""
        SandboxManager.initialize(tmp_path)
        t1 = SandboxManager.create_trial(tmp_path)
        (t1 / "REPORT.md").write_text("Prior trial report.")
        t2 = SandboxManager.create_trial(tmp_path)

        chat = FakeChat([SlashCommand("/approve", "")])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        handle_experiment_plan(t2, meta, config, chat)

        assert any("prior trials" in m for m in chat.sent)

    def test_messages_accumulated(self, tmp_path: Path, monkeypatch: object) -> None:
        """Messages are accumulated as list of dicts with role/content."""
        trial_dir = _setup_sandbox(tmp_path)
        call_messages: list[list[dict[str, str]]] = []

        class TrackingProvider:
            def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
                call_messages.append(list(messages))
                return "Response\n\n**If you approve plan, please type /approve. If not, please iterate.**"

        chat = FakeChat([
            UserMessage("first msg"),
            UserMessage("second msg"),
            SlashCommand("/approve", ""),
        ])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: TrackingProvider())  # type: ignore[attr-defined]

        handle_experiment_plan(trial_dir, meta, config, chat)

        # First call: 1 user message
        assert len(call_messages[0]) == 1
        assert call_messages[0][0]["role"] == "user"
        # Second call: user, assistant, user
        assert len(call_messages[1]) == 3
        assert call_messages[1][0]["role"] == "user"
        assert call_messages[1][1]["role"] == "assistant"
        assert call_messages[1][2]["role"] == "user"

    def test_approve_bold_prompt_in_greeting(self, tmp_path: Path, monkeypatch: object) -> None:
        """Greeting includes bold approve instruction."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([SlashCommand("/approve", "")])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        handle_experiment_plan(trial_dir, meta, config, chat)

        assert any("/approve" in m and "iterate" in m for m in chat.sent)


# --- Tests for handle_experiment_plan (autopilot mode) ---

class TestAutopilotPlan:
    def test_autopilot_with_provider(self, tmp_path: Path, monkeypatch: object) -> None:
        """Autopilot mode with LLM generates plan automatically."""
        trial_dir = _setup_sandbox(tmp_path)
        provider = FakeProvider(responses=["# Auto Plan\nDo the experiment."])
        chat = FakeChat([])
        config = ResearchClawConfig(autopilot=True)
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        plan_content = (trial_dir / "PLAN.md").read_text()
        assert "Auto Plan" in plan_content
        assert meta.plan_approved_at is not None
        assert meta.status == "running"
        assert any("autopilot" in m for m in chat.sent)

    def test_autopilot_without_provider(self, tmp_path: Path, monkeypatch: object) -> None:
        """Autopilot without LLM writes fallback plan."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([])
        config = ResearchClawConfig(autopilot=True)
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        plan_content = (trial_dir / "PLAN.md").read_text()
        assert "no LLM available" in plan_content

    def test_autopilot_llm_error(self, tmp_path: Path, monkeypatch: object) -> None:
        """Autopilot with LLM error writes fallback plan."""
        trial_dir = _setup_sandbox(tmp_path)
        provider = FakeProvider(error=RuntimeError("API error"))
        chat = FakeChat([])
        config = ResearchClawConfig(autopilot=True)
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_IMPLEMENT
        plan_content = (trial_dir / "PLAN.md").read_text()
        assert "LLM error" in plan_content

    def test_autopilot_no_chat_interface(self, tmp_path: Path, monkeypatch: object) -> None:
        """Autopilot with None chat interface still works."""
        trial_dir = _setup_sandbox(tmp_path)
        config = ResearchClawConfig(autopilot=True)
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_experiment_plan(trial_dir, meta, config, None)

        assert result == State.EXPERIMENT_IMPLEMENT
        assert (trial_dir / "PLAN.md").exists()

    def test_autopilot_writes_plan_to_correct_path(self, tmp_path: Path, monkeypatch: object) -> None:
        """Autopilot writes PLAN.md in trial directory."""
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat([])
        config = ResearchClawConfig(autopilot=True)
        meta = TrialMeta()

        monkeypatch.setattr(plan_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        handle_experiment_plan(trial_dir, meta, config, chat)

        plan_path = trial_dir / "PLAN.md"
        assert plan_path.exists()
        assert plan_path.read_text().startswith("# Experiment Plan")


# --- Tests for system prompts ---

class TestSystemPrompts:
    def test_planning_agent_system_has_placeholders(self) -> None:
        assert "{historian_context}" in PLANNING_AGENT_SYSTEM
        assert "{search_context}" in PLANNING_AGENT_SYSTEM

    def test_planning_agent_system_ends_with_approve(self) -> None:
        assert "/approve" in PLANNING_AGENT_SYSTEM

    def test_historian_system_mentions_budget(self) -> None:
        assert "3000" in HISTORIAN_AGENT_SYSTEM

    def test_historian_system_mentions_threshold(self) -> None:
        assert "5" in HISTORIAN_AGENT_SYSTEM
