"""
Tests for the ResearchClaw agent state machine.

Tests cover:
  - State transitions and guards
  - Slash command dispatch (/status, /start, /approve, etc.)
  - LLM conversation routing (all states)
  - RESEARCH_TRIAL_SUMMARY.md lifecycle
  - State restoration from .trials.jsonl
  - Agentic tool-use loop (mocked)
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from researchclaw.agent import AgentState, ResearchClaw, _SLASH_COMMANDS
from researchclaw.config import Config
from researchclaw.llm import LLMResponse, ToolCall
from researchclaw.models import TrialInfo, TrialStatus
from researchclaw.watcher import ExperimentEvent, ExperimentStatus


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_dir(tmp_path):
    """Set up a realistic base directory."""
    (tmp_path / "github_codes").mkdir()
    (tmp_path / "github_codes" / "train.py").write_text("# train")
    (tmp_path / "github_codes" / "model.py").write_text("# model")
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "experiment_reports").mkdir()
    (tmp_path / "reference").mkdir()
    (tmp_path / ".trials.jsonl").write_text("")
    (tmp_path / "MEMORY.md").write_text("# Agent Memory\n\n(empty)\n")
    (tmp_path / "RESEARCH_TRIAL_SUMMARY.md").write_text(
        "# Research Trial Summary\n\nLast updated: (none)\n\n## Trial History\n\n(No trials yet.)\n"
    )
    (tmp_path / "config.yaml").write_text("base_dir: " + str(tmp_path))
    return tmp_path


@pytest.fixture
def config(base_dir):
    """Create a Config pointing at tmp base_dir."""
    return Config(
        base_dir=str(base_dir),
        messenger_type="stdio",
        llm_provider="anthropic",
    )


@pytest.fixture
def agent(config, base_dir):
    """Create a ResearchClaw agent with mocked external dependencies.

    The mock LLM has supports_tool_use=False by default, so the text-marker
    (legacy) path is tested. Set agent.llm.supports_tool_use = True to test
    the agentic loop.
    """
    with patch("researchclaw.agent.get_llm_provider") as mock_llm_factory, \
         patch("researchclaw.agent.get_messenger") as mock_msg_factory, \
         patch("researchclaw.agent.Watcher") as mock_watcher_cls:

        # Mock LLM — text-marker path by default
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "OK"
        mock_llm.complete_with_context.return_value = "I suggest we try X."
        mock_llm.supports_tool_use = False
        mock_llm_factory.return_value = mock_llm

        # Mock Messenger
        mock_messenger = MagicMock()
        mock_messenger.receive.return_value = None
        mock_messenger.confirm.return_value = True
        mock_msg_factory.return_value = mock_messenger

        # Mock Watcher
        mock_watcher = MagicMock()
        mock_watcher.check_gpu.return_value = None
        mock_watcher_cls.return_value = mock_watcher

        a = ResearchClaw(config)
        yield a


# ──────────────────────────────────────────────────────────────────────────────
# TestAgentStateEnum
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentStateEnum:
    def test_has_four_states(self):
        assert len(AgentState) == 4

    def test_state_values(self):
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RESEARCH.value == "research"
        assert AgentState.EXECUTE.value == "execute"
        assert AgentState.AWAITING_APPROVAL.value == "awaiting_approval"


# ──────────────────────────────────────────────────────────────────────────────
# TestInitialState
# ──────────────────────────────────────────────────────────────────────────────

class TestInitialState:
    def test_starts_in_idle(self, agent):
        assert agent.state == AgentState.IDLE

    def test_no_current_trial(self, agent):
        assert agent.current_trial is None

    def test_no_active_proc(self, agent):
        assert agent.active_proc is None


# ──────────────────────────────────────────────────────────────────────────────
# TestSlashCommands
# ──────────────────────────────────────────────────────────────────────────────

class TestSlashCommands:
    """Test that slash commands are dispatched correctly."""

    def test_slash_status(self, agent):
        agent._handle_message("/status")
        agent.messenger.send.assert_called()
        last_call = agent.messenger.send.call_args[0][0]
        assert "IDLE" in last_call

    def test_slash_summary(self, agent):
        agent._handle_message("/summary")
        agent.messenger.send.assert_called()

    def test_slash_push(self, agent):
        with patch.object(agent.git_mgr, "push", return_value="ok"):
            agent._handle_message("/push")
            agent.messenger.send.assert_called()

    def test_slash_start_transitions_to_research(self, agent):
        agent._handle_message("/start testing new LR")
        assert agent.state == AgentState.RESEARCH
        assert agent.current_trial is not None
        assert agent.current_trial.status == TrialStatus.ACTIVE

    def test_slash_kill_in_execute(self, agent):
        # Enter RESEARCH then EXECUTE
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test")
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 99999
        agent.active_proc = mock_proc
        with agent._trial_lock:
            agent._transition_to(AgentState.EXECUTE, "test")

        agent.messenger.confirm.return_value = True
        agent._handle_message("/kill")
        assert agent.state == AgentState.RESEARCH

    def test_slash_approve(self, agent):
        # Enter AWAITING_APPROVAL
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test")
        with agent._trial_lock:
            agent.sandbox_mgr.mark_review(agent.current_trial)
            agent._transition_to(AgentState.AWAITING_APPROVAL, "test")

        with patch.object(agent.git_mgr, "get_diff", return_value="diff"), \
             patch.object(agent.git_mgr, "authorize_merge"), \
             patch.object(agent.git_mgr, "merge_trial", return_value="abc1234"), \
             patch.object(agent.summarizer, "generate_trial_summary_entry", return_value="### entry"):
            agent.messenger.confirm.side_effect = [True, False]  # merge yes, push no
            agent._handle_message("/approve")
            assert agent.state == AgentState.IDLE

    def test_slash_reject(self, agent):
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test")
        with agent._trial_lock:
            agent.sandbox_mgr.mark_review(agent.current_trial)
            agent._transition_to(AgentState.AWAITING_APPROVAL, "test")

        with patch.object(agent.summarizer, "generate_trial_summary_entry", return_value="### entry"):
            agent._handle_message("/reject")
            assert agent.state == AgentState.IDLE

    def test_slash_continue(self, agent):
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test")
        with agent._trial_lock:
            agent.sandbox_mgr.mark_review(agent.current_trial)
            agent._transition_to(AgentState.AWAITING_APPROVAL, "test")

        agent._handle_message("/continue")
        assert agent.state == AgentState.RESEARCH

    def test_all_slash_commands_registered(self):
        """Ensure all expected slash commands exist."""
        expected = {"/status", "/summary", "/push", "/start", "/kill",
                    "/approve", "/reject", "/continue"}
        assert set(_SLASH_COMMANDS.keys()) == expected


# ──────────────────────────────────────────────────────────────────────────────
# TestIdleState — LLM routing (no keyword hijack)
# ──────────────────────────────────────────────────────────────────────────────

class TestIdleState:
    def test_start_trial_via_slash_command(self, agent):
        agent._handle_message("/start with goal X")
        assert agent.state == AgentState.RESEARCH
        assert agent.current_trial is not None
        assert agent.current_trial.status == TrialStatus.ACTIVE

    def test_natural_language_goes_to_llm_in_idle(self, agent):
        """Natural language messages should go to LLM, not return canned response."""
        agent.llm.complete_with_context.return_value = "I can help with that!"
        agent._handle_message("hello there")
        # LLM should have been called
        agent.llm.complete_with_context.assert_called()
        # Response should be sent to researcher
        agent.messenger.send.assert_called()
        last_call = agent.messenger.send.call_args[0][0]
        assert "I can help with that!" in last_call

    def test_status_keyword_goes_to_llm_not_canned(self, agent):
        """'status' without slash prefix should go to LLM, not trigger status handler."""
        agent.llm.complete_with_context.return_value = "Let me check the status for you."
        agent._handle_message("what's the status of the model convergence?")
        agent.llm.complete_with_context.assert_called()

    def test_update_keyword_goes_to_llm_not_canned(self, agent):
        """'update' in natural language should go to LLM, not trigger status."""
        agent.llm.complete_with_context.return_value = "I'll update the learning rate."
        agent._handle_message("update the learning rate")
        agent.llm.complete_with_context.assert_called()

    def test_slash_status_still_works(self, agent):
        agent._handle_message("/status")
        agent.messenger.send.assert_called()
        last_call = agent.messenger.send.call_args[0][0]
        assert "IDLE" in last_call


# ──────────────────────────────────────────────────────────────────────────────
# TestResearchState
# ──────────────────────────────────────────────────────────────────────────────

class TestResearchState:
    def _enter_research(self, agent):
        """Helper: transition agent to RESEARCH state."""
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start new trial")
        assert agent.state == AgentState.RESEARCH

    def test_reject_via_slash_command(self, agent):
        self._enter_research(agent)
        agent._handle_message("/reject")
        assert agent.state == AgentState.IDLE

    def test_llm_conversation_stays_in_research(self, agent):
        self._enter_research(agent)
        agent.llm.complete_with_context.return_value = "I suggest trying a lower LR."
        agent._handle_message("try reducing the learning rate")
        assert agent.state == AgentState.RESEARCH

    def test_code_change_stays_in_research(self, agent):
        self._enter_research(agent)
        agent.llm.complete_with_context.return_value = (
            "CODE_CHANGE: train.py\n# modified training script"
        )
        agent._handle_message("modify the training script")
        assert agent.state == AgentState.RESEARCH

    def test_run_command_transitions_to_execute(self, agent):
        self._enter_research(agent)
        agent.llm.complete_with_context.return_value = "RUN_COMMAND: python train.py"
        agent.messenger.confirm.return_value = True

        with patch.object(agent.runner, "run_async") as mock_run:
            mock_proc = MagicMock(spec=subprocess.Popen)
            mock_proc.pid = 12345
            mock_run.return_value = mock_proc

            # Patch threading to not actually start a thread
            with patch("researchclaw.agent.threading.Thread") as mock_thread:
                agent._handle_message("run the training")
                assert agent.state == AgentState.EXECUTE
                assert agent.active_proc is mock_proc

    def test_start_trial_blocked_in_research(self, agent):
        self._enter_research(agent)
        agent.messenger.send.reset_mock()
        agent._handle_message("/start a new trial")
        # Should stay in RESEARCH, not create a second trial
        assert agent.state == AgentState.RESEARCH
        # One of the sent messages should mention there's already an active trial
        all_calls = [call[0][0] for call in agent.messenger.send.call_args_list]
        combined = " ".join(all_calls).lower()
        assert "already" in combined or "active trial" in combined or "finish" in combined

    def test_report_via_text_marker(self, agent):
        self._enter_research(agent)
        agent.llm.complete_with_context.return_value = "REPORT"
        with patch.object(agent.git_mgr, "get_full_diff", return_value="diff here"), \
             patch.object(agent.summarizer, "generate_report", return_value="# Report\nresults here"):
            agent._handle_message("generate report")
            assert agent.state == AgentState.AWAITING_APPROVAL

    def test_natural_language_with_keywords_goes_to_llm(self, agent):
        """Messages containing keywords like 'abort' should go to LLM, not trigger actions."""
        self._enter_research(agent)
        agent.llm.complete_with_context.return_value = "I can try a different approach."
        agent._handle_message("abort the current approach and try something new")
        # Should go to LLM, not trigger reject
        assert agent.state == AgentState.RESEARCH
        agent.llm.complete_with_context.assert_called()


# ──────────────────────────────────────────────────────────────────────────────
# TestExecuteState
# ──────────────────────────────────────────────────────────────────────────────

class TestExecuteState:
    def _enter_execute(self, agent):
        """Helper: transition agent to EXECUTE state."""
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start new trial")

        # Manually transition to EXECUTE
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 99999
        agent.active_proc = mock_proc
        with agent._trial_lock:
            agent._transition_to(AgentState.EXECUTE, "test")
        assert agent.state == AgentState.EXECUTE

    def test_natural_language_goes_to_llm_in_execute(self, agent):
        """In EXECUTE state, natural language goes to LLM (can discuss experiment)."""
        self._enter_execute(agent)
        agent.llm.complete_with_context.return_value = "The experiment is running. Looks good so far."
        agent._handle_message("how is it going?")
        agent.llm.complete_with_context.assert_called()
        # Should still be in EXECUTE
        assert agent.state == AgentState.EXECUTE

    def test_slash_kill_transitions_to_research(self, agent):
        self._enter_execute(agent)
        agent.messenger.confirm.return_value = True
        agent._handle_message("/kill")
        assert agent.state == AgentState.RESEARCH
        assert agent.active_proc is None

    def test_kill_cancelled_stays_in_execute(self, agent):
        self._enter_execute(agent)
        agent.messenger.confirm.return_value = False
        agent._handle_message("/kill")
        assert agent.state == AgentState.EXECUTE

    def test_slash_status_works_in_execute(self, agent):
        self._enter_execute(agent)
        agent._handle_message("/status")
        agent.messenger.send.assert_called()

    def test_start_trial_blocked_in_execute(self, agent):
        self._enter_execute(agent)
        agent._handle_message("/start new trial")
        # Should still be in EXECUTE
        assert agent.state == AgentState.EXECUTE


# ──────────────────────────────────────────────────────────────────────────────
# TestAwaitingApprovalState
# ──────────────────────────────────────────────────────────────────────────────

class TestAwaitingApprovalState:
    def _enter_awaiting_approval(self, agent):
        """Helper: transition agent to AWAITING_APPROVAL state."""
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start new trial")

        # Manually transition to AWAITING_APPROVAL
        with agent._trial_lock:
            agent.sandbox_mgr.mark_review(agent.current_trial)
            agent._transition_to(AgentState.AWAITING_APPROVAL, "test")
        assert agent.state == AgentState.AWAITING_APPROVAL

    def test_approve_transitions_to_idle(self, agent):
        self._enter_awaiting_approval(agent)
        agent.messenger.confirm.return_value = True
        with patch.object(agent.git_mgr, "get_diff", return_value="diff"), \
             patch.object(agent.git_mgr, "authorize_merge"), \
             patch.object(agent.git_mgr, "merge_trial", return_value="abc1234"), \
             patch.object(agent.summarizer, "generate_trial_summary_entry", return_value="### entry"):
            # First confirm is for merge, second is for push
            agent.messenger.confirm.side_effect = [True, False]
            agent._handle_message("/approve")
            assert agent.state == AgentState.IDLE

    def test_reject_transitions_to_idle(self, agent):
        self._enter_awaiting_approval(agent)
        with patch.object(agent.summarizer, "generate_trial_summary_entry", return_value="### entry"):
            agent._handle_message("/reject")
            assert agent.state == AgentState.IDLE

    def test_continue_transitions_to_research(self, agent):
        self._enter_awaiting_approval(agent)
        agent._handle_message("/continue")
        assert agent.state == AgentState.RESEARCH
        assert agent.current_trial.status == TrialStatus.ACTIVE

    def test_natural_language_goes_to_llm(self, agent):
        """In AWAITING_APPROVAL, natural language goes to LLM (can discuss results)."""
        self._enter_awaiting_approval(agent)
        agent.llm.complete_with_context.return_value = "The results look promising. I'd suggest approving."
        agent._handle_message("what do you think about these results?")
        agent.llm.complete_with_context.assert_called()
        # Should still be in AWAITING_APPROVAL
        assert agent.state == AgentState.AWAITING_APPROVAL

    def test_start_trial_blocked(self, agent):
        self._enter_awaiting_approval(agent)
        agent._handle_message("/start a new trial")
        # Should stay in AWAITING_APPROVAL
        assert agent.state == AgentState.AWAITING_APPROVAL

    def test_approve_blocked_if_not_in_approval_state(self, agent):
        """Approving in IDLE should not cause errors."""
        assert agent.state == AgentState.IDLE
        agent._handle_message("/approve")
        assert agent.state == AgentState.IDLE


# ──────────────────────────────────────────────────────────────────────────────
# TestTextMarkerActions (legacy text-marker path)
# ──────────────────────────────────────────────────────────────────────────────

class TestTextMarkerActions:
    """Test new lifecycle action markers (START_TRIAL, APPROVE_TRIAL, etc.)."""

    def test_start_trial_marker_in_idle(self, agent):
        agent.llm.complete_with_context.return_value = "START_TRIAL: test learning rates"
        agent._handle_message("let's start experimenting")
        assert agent.state == AgentState.RESEARCH

    def test_approve_trial_marker(self, agent):
        # Enter AWAITING_APPROVAL
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test")
        with agent._trial_lock:
            agent.sandbox_mgr.mark_review(agent.current_trial)
            agent._transition_to(AgentState.AWAITING_APPROVAL, "test")

        agent.llm.complete_with_context.return_value = "Results look great. APPROVE_TRIAL"
        with patch.object(agent.git_mgr, "get_diff", return_value="diff"), \
             patch.object(agent.git_mgr, "authorize_merge"), \
             patch.object(agent.git_mgr, "merge_trial", return_value="abc1234"), \
             patch.object(agent.summarizer, "generate_trial_summary_entry", return_value="### entry"):
            agent.messenger.confirm.side_effect = [True, False]
            agent._handle_message("approve this")
            assert agent.state == AgentState.IDLE

    def test_push_to_remote_marker(self, agent):
        agent.llm.complete_with_context.return_value = "PUSH_TO_REMOTE"
        with patch.object(agent.git_mgr, "push", return_value="ok"):
            agent._handle_message("push the code")
            agent.messenger.send.assert_called()

    def test_memory_update_marker(self, agent):
        # Enter RESEARCH so LLM is called
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test")

        agent.llm.complete_with_context.return_value = (
            "MEMORY_UPDATE:\n- The project uses PyTorch\n- LR=0.001 works best"
        )
        agent._handle_message("remember this")
        memory = (agent.base_dir / "MEMORY.md").read_text()
        assert "PyTorch" in memory


# ──────────────────────────────────────────────────────────────────────────────
# TestAgenticLoop
# ──────────────────────────────────────────────────────────────────────────────

class TestAgenticLoop:
    """Test the agentic tool-use loop (supports_tool_use=True)."""

    def _enable_tool_use(self, agent):
        """Switch the mock LLM to tool-use mode."""
        agent.llm.supports_tool_use = True

    def test_pure_text_response_sent_to_researcher(self, agent):
        self._enable_tool_use(agent)
        agent.llm.complete_with_tools.return_value = LLMResponse(
            text="Hello! How can I help?", tool_calls=[], stop_reason="end_turn"
        )
        agent._handle_message("hello")
        agent.messenger.send.assert_called()
        last_call = agent.messenger.send.call_args[0][0]
        assert "Hello! How can I help?" in last_call

    def test_read_file_tool_call(self, agent, base_dir):
        """LLM calls read_file, gets content back, then responds."""
        self._enable_tool_use(agent)

        # First call: LLM requests read_file tool
        # Second call: LLM responds with text after getting file content
        agent.llm.complete_with_tools.side_effect = [
            LLMResponse(
                text="Let me check that file.",
                tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "github_codes/train.py"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="The training script contains basic training code.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]
        agent._handle_message("what's in train.py?")
        # Should have called complete_with_tools twice
        assert agent.llm.complete_with_tools.call_count == 2
        # Last message sent should be the final response
        last_call = agent.messenger.send.call_args[0][0]
        assert "training script" in last_call.lower()

    def test_write_file_blocked_outside_research(self, agent):
        """write_file tool should be blocked when not in RESEARCH state."""
        self._enable_tool_use(agent)
        # Agent is in IDLE — write_file should return error
        agent.llm.complete_with_tools.side_effect = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc_1", name="write_file", arguments={"path": "test.py", "content": "# test"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="I can't write files right now.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]
        agent._handle_message("write a file")
        # The tool result should contain an error
        history = agent.conversation_history
        tool_results = [m for m in history if m.get("role") == "tool_result"]
        assert any("ERROR" in r.get("content", "") for r in tool_results)

    def test_propose_action_start_trial(self, agent):
        """LLM can propose starting a trial via propose_action tool."""
        self._enable_tool_use(agent)
        agent.llm.complete_with_tools.side_effect = [
            LLMResponse(
                text="Let's start a trial!",
                tool_calls=[ToolCall(id="tc_1", name="propose_action",
                                    arguments={"action": "start_trial", "detail": "test LR"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="Trial started. What should we try?",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]
        agent._handle_message("let's experiment")
        assert agent.state == AgentState.RESEARCH
        assert agent.current_trial is not None

    def test_max_iterations_stop(self, agent):
        """Agentic loop should stop after max_iterations."""
        self._enable_tool_use(agent)
        agent.config.agent_max_iterations = 2

        # Always return tool calls — should be capped at 2 iterations
        agent.llm.complete_with_tools.return_value = LLMResponse(
            text="Still working...",
            tool_calls=[ToolCall(id="tc_1", name="list_directory", arguments={"path": "github_codes"})],
            stop_reason="tool_use",
        )
        agent._handle_message("explore the codebase")
        assert agent.llm.complete_with_tools.call_count == 2
        # Should have sent the "max iterations" message
        all_calls = [call[0][0] for call in agent.messenger.send.call_args_list]
        assert any("max iterations" in c.lower() for c in all_calls)

    def test_available_tools_vary_by_state(self, agent):
        """RESEARCH state should have write/run tools; other states should not."""
        # IDLE — no write/run tools
        tools = agent._get_available_tools()
        tool_names = {t["name"] for t in tools}
        assert "write_file" not in tool_names
        assert "run_command" not in tool_names
        assert "read_file" in tool_names

        # Enter RESEARCH
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test")
        tools = agent._get_available_tools()
        tool_names = {t["name"] for t in tools}
        assert "write_file" in tool_names
        assert "run_command" in tool_names
        assert "read_file" in tool_names


# ──────────────────────────────────────────────────────────────────────────────
# TestStateRestoration
# ──────────────────────────────────────────────────────────────────────────────

class TestStateRestoration:
    def test_restores_active_trial_to_research(self, config, base_dir):
        """If .trials.jsonl has an active trial, agent should start in RESEARCH."""
        import json
        trial_data = {
            "date": "20260226", "number": 1, "status": "active",
            "started_at": "2026-02-26T10:00:00", "goal": "test",
        }
        (base_dir / ".trials.jsonl").write_text(json.dumps(trial_data) + "\n")
        # Create the sandbox directory
        sandbox = base_dir / "sandbox" / "20260226" / "trial_001"
        sandbox.mkdir(parents=True)

        with patch("researchclaw.agent.get_llm_provider") as mock_llm_factory, \
             patch("researchclaw.agent.get_messenger") as mock_msg_factory, \
             patch("researchclaw.agent.Watcher"):
            mock_llm = MagicMock()
            mock_llm.supports_tool_use = False
            mock_llm_factory.return_value = mock_llm
            mock_msg_factory.return_value = MagicMock()

            a = ResearchClaw(config)
            assert a.state == AgentState.RESEARCH
            assert a.current_trial is not None
            assert a.current_trial.status == TrialStatus.ACTIVE

    def test_restores_review_trial_to_awaiting_approval(self, config, base_dir):
        """If .trials.jsonl has a review trial, agent should start in AWAITING_APPROVAL."""
        import json
        trial_data = {
            "date": "20260226", "number": 1, "status": "review",
            "started_at": "2026-02-26T10:00:00", "goal": "test",
        }
        (base_dir / ".trials.jsonl").write_text(json.dumps(trial_data) + "\n")
        sandbox = base_dir / "sandbox" / "20260226" / "trial_001"
        sandbox.mkdir(parents=True)

        with patch("researchclaw.agent.get_llm_provider") as mock_llm_factory, \
             patch("researchclaw.agent.get_messenger") as mock_msg_factory, \
             patch("researchclaw.agent.Watcher"):
            mock_llm = MagicMock()
            mock_llm.supports_tool_use = False
            mock_llm_factory.return_value = mock_llm
            mock_msg_factory.return_value = MagicMock()

            a = ResearchClaw(config)
            assert a.state == AgentState.AWAITING_APPROVAL
            assert a.current_trial is not None

    def test_starts_idle_with_no_trials(self, agent):
        assert agent.state == AgentState.IDLE


# ──────────────────────────────────────────────────────────────────────────────
# TestTrialSummaryLifecycle
# ──────────────────────────────────────────────────────────────────────────────

class TestTrialSummaryLifecycle:
    def test_summary_file_exists_after_init(self, base_dir):
        summary = base_dir / "RESEARCH_TRIAL_SUMMARY.md"
        assert summary.exists()

    def test_summary_updated_on_approve(self, agent):
        # Start and progress to AWAITING_APPROVAL
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test LR")

        with agent._trial_lock:
            agent.sandbox_mgr.mark_review(agent.current_trial)
            agent._transition_to(AgentState.AWAITING_APPROVAL, "test")

        with patch.object(agent.git_mgr, "get_diff", return_value="diff"), \
             patch.object(agent.git_mgr, "authorize_merge"), \
             patch.object(agent.git_mgr, "merge_trial", return_value="abc123"), \
             patch.object(agent.summarizer, "generate_trial_summary_entry",
                         return_value="### trial_001 (20260226) -- APPROVED\n**Goal:** test\n"):
            agent.messenger.confirm.side_effect = [True, False]  # approve, don't push
            agent._handle_message("/approve")

        summary = (agent.base_dir / "RESEARCH_TRIAL_SUMMARY.md").read_text()
        assert "trial_001" in summary

    def test_summary_updated_on_reject(self, agent):
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start test LR")

        with agent._trial_lock:
            agent.sandbox_mgr.mark_review(agent.current_trial)
            agent._transition_to(AgentState.AWAITING_APPROVAL, "test")

        with patch.object(agent.summarizer, "generate_trial_summary_entry",
                         return_value="### trial_001 (20260226) -- REJECTED\n**Goal:** test\n"):
            agent._handle_message("/reject")

        summary = (agent.base_dir / "RESEARCH_TRIAL_SUMMARY.md").read_text()
        assert "trial_001" in summary

    def test_summary_readable_via_send(self, agent):
        agent._send_trial_summary()
        agent.messenger.send.assert_called()
        last_call = agent.messenger.send.call_args[0][0]
        assert "Research Trial Summary" in last_call


# ──────────────────────────────────────────────────────────────────────────────
# TestWatcherTransition
# ──────────────────────────────────────────────────────────────────────────────

class TestWatcherTransition:
    def test_finished_transitions_to_awaiting_approval(self, agent):
        """When the watcher detects FINISHED, state should go to AWAITING_APPROVAL."""
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start new trial")

        trial = agent.current_trial
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 55555

        # Simulate watcher yielding a FINISHED event
        finished_status = MagicMock(spec=ExperimentStatus)
        finished_status.event = ExperimentEvent.FINISHED
        finished_status.trial_name = trial.trial_name
        finished_status.duration = "0m 05s"
        finished_status.gpu_info = None
        finished_status.new_files = []
        finished_status.log_tail = "done"
        finished_status.returncode = 0

        with patch.object(agent.watcher, "watch", return_value=[finished_status]), \
             patch.object(agent.summarizer, "format_status_message", return_value="done"), \
             patch.object(agent.git_mgr, "get_full_diff", return_value=""), \
             patch.object(agent.summarizer, "generate_report", return_value="# Report"):
            agent._watch_experiment(mock_proc, trial)

        assert agent.state == AgentState.AWAITING_APPROVAL
        assert agent.current_trial.status == TrialStatus.REVIEW

    def test_crashed_transitions_to_awaiting_approval(self, agent):
        """When the watcher detects CRASHED, state should still go to AWAITING_APPROVAL."""
        agent.llm.complete_with_context.return_value = "What should I try?"
        agent._handle_message("/start new trial")

        trial = agent.current_trial
        mock_proc = MagicMock(spec=subprocess.Popen)

        crashed_status = MagicMock(spec=ExperimentStatus)
        crashed_status.event = ExperimentEvent.CRASHED
        crashed_status.trial_name = trial.trial_name
        crashed_status.duration = "0m 02s"
        crashed_status.returncode = 1
        crashed_status.log_tail = "error"

        with patch.object(agent.watcher, "watch", return_value=[crashed_status]), \
             patch.object(agent.summarizer, "format_status_message", return_value="crashed"):
            agent._watch_experiment(mock_proc, trial)

        assert agent.state == AgentState.AWAITING_APPROVAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
