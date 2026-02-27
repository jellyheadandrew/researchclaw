"""
agent.py — ResearchClaw main loop.

The agent runs as a long-lived daemon process with four states:
  IDLE               — no active trial, waiting for researcher to start work
  RESEARCH           — trial in progress, agent proposes code changes
  EXECUTE            — experiment subprocess running, agent monitors
  AWAITING_APPROVAL  — experiment done, REPORT.md generated, awaiting decision

Conversation style: casual, concise, research-focused. Like a capable labmate.
The agent always shows exact commands before running them. No surprises.

LLM responses are tagged with action markers:
  CODE_CHANGE: <path>\n<new content>
  RUN_COMMAND: <shell command>
  QUESTION: <text>
  REPORT: (trigger report generation)
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path

from .access_control import PathValidator
from .config import Config, load_config
from .env_manager import EnvManager
from .git_manager import GitManager
from .llm import AGENT_TOOLS, LLMProvider, LLMResponse, ToolCall, get_llm_provider
from .messenger import Messenger, get_messenger
from .models import TrialInfo, TrialStatus
from .runner import Runner
from .sandbox_manager import SandboxManager
from .summarizer import Summarizer
from .utils import setup_logging
from .watcher import ExperimentEvent, Watcher

logger = logging.getLogger("researchclaw.agent")


class AgentState(Enum):
    IDLE = "idle"
    RESEARCH = "research"
    EXECUTE = "execute"
    AWAITING_APPROVAL = "awaiting_approval"


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are ResearchClaw, an AI research collaborator running on the researcher's GPU server.

Your personality:
- You are a careful, competent research engineer.
- You speak concisely. No fluff. Researchers hate fluff.
- You always show the exact command before running it. Never surprise the researcher.
- When you're unsure, say so. "I'm not sure if this config is right — does this look correct?"
- You track what trial you're on and what the research goal is.

Your constraints:
- You can READ all code in github_codes/, all past trials, and reference/.
- reference/ contains user-supplied context: papers, API docs, external codebases, notes.
  Read it proactively when interpreting researcher directions — it may explain terminology,
  constraints, or prior work that informs what to try.
- You can only WRITE inside the current trial's sandbox and report directories.
- You NEVER modify github_codes/ directly. Changes go through sandbox → approval → merge.
- You ALWAYS ask for confirmation before executing any command.
- You NEVER run destructive commands (rm -rf, sudo, etc.) without explicit approval.

Your memory (persists across restarts):
{agent_memory}

Update your memory when you learn project-specific facts that are worth keeping across trials:
code conventions, researcher preferences, recurring patterns, what approaches failed and why.
Do NOT store per-trial notes here — those belong in REPORT.md.
Keep it concise: short bullet points, no repetition. Overwrite the whole file each time.

{state_instructions}

Current state:
- Base directory: {base_dir}
- Agent state: {agent_state}
- Current trial: {current_trial}
- GitHub repo: {repo_info}
- GPU: {gpu_info}

Prior trial summary (avoid repeating failed approaches):
{trial_summary}

Your workflow:
1. Researcher says "start a new trial" → you create a sandbox, copy the codebase, ask what to try.
2. You propose code changes, show diffs, get approval.
3. You run the experiment, monitor it, report results.
4. When done, you generate REPORT.md and ask: approve (merge to main), reject, or continue.

You can use tools to read files, list directories, search for patterns, write files,
run commands, and propose lifecycle actions. Use tools proactively to explore the
codebase before proposing changes. When the researcher asks about code, read the
relevant files first.

The researcher also has these slash commands available for quick actions:
  /status, /summary, /push, /start <goal>, /kill, /approve, /reject, /continue

{text_marker_instructions}
"""

# Injected into system prompt only for providers without native tool use.
# Providers with tool_use get structured tool definitions instead.
TEXT_MARKER_INSTRUCTIONS = """\
When you want to take an action, use these markers in your response:
- To modify a file: CODE_CHANGE: <relative/path/in/sandbox>\\n<full new content>
- To run a command: RUN_COMMAND: <command>
- To ask a question: QUESTION: <your question>
- When experiment is done and you've analyzed results: REPORT
- To update your persistent memory (no approval needed): MEMORY_UPDATE:\\n<full new content of MEMORY.md>
- To start a trial: START_TRIAL: <research goal>
- To propose approval: APPROVE_TRIAL
- To propose rejection: REJECT_TRIAL
- To continue iterating: CONTINUE_TRIAL
- To push to remote: PUSH_TO_REMOTE
"""

# State-specific instructions injected into the system prompt
STATE_INSTRUCTIONS = {
    AgentState.IDLE: (
        "You are in IDLE mode. No trial is active.\n"
        "You can have a conversation with the researcher — discuss research plans,\n"
        "answer questions about the codebase, or help with setup.\n"
        "When the researcher is ready to experiment, propose starting a trial\n"
        "(use the propose_action tool with action='start_trial', or START_TRIAL marker).\n"
        "You can also read files from github_codes/ and reference/ to help discuss the project."
    ),
    AgentState.RESEARCH: (
        "You are in RESEARCH mode. You should:\n"
        "- Read the trial summary below to avoid repeating past experiments.\n"
        "- Read reference/ for relevant context (papers, docs, prior work).\n"
        "- Explore the codebase using read_file and search_files tools before proposing changes.\n"
        "- Propose code changes or commands to run.\n"
        "- Do NOT run long experiments without discussing the plan first.\n"
        "- When you run a command, it will start the experiment and you won't be able\n"
        "  to make more code changes until it finishes."
    ),
    AgentState.EXECUTE: (
        "An experiment is currently running. Do NOT propose code changes or new commands.\n"
        "You can still have a conversation — answer questions, discuss what to try next,\n"
        "read files, and help the researcher think about the experiment.\n"
        "The watcher will notify when the experiment finishes."
    ),
    AgentState.AWAITING_APPROVAL: (
        "The experiment has finished. A REPORT.md has been generated.\n"
        "You can discuss the results with the researcher and help them decide.\n"
        "The options are: approve (merge to main), reject (discard), or continue (iterate).\n"
        "Use the propose_action tool or the appropriate marker to propose the next step.\n"
        "You can also read files (sandbox, reports, github_codes) to help analyze results."
    ),
}


# ── Slash command dispatch ────────────────────────────────────────────────────

# Mapping from slash-command prefix to (method_name, takes_msg_arg).
# Only exact prefix matches are recognized — no substring matching.
_SLASH_COMMANDS: dict[str, tuple[str, bool]] = {
    "/status":   ("_send_status", False),
    "/summary":  ("_send_trial_summary", False),
    "/push":     ("_push_to_github", False),
    "/start":    ("_start_trial", True),
    "/kill":     ("_kill_experiment", False),
    "/approve":  ("_approve_trial", True),
    "/reject":   ("_reject_trial", False),
    "/continue": ("_continue_trial", False),
}


class ResearchClaw:
    """Main ResearchClaw agent."""

    def __init__(self, config: Config):
        self.config = config
        self.base_dir = Path(config.base_dir).resolve()

        # Core components
        self.validator = PathValidator(str(self.base_dir))
        self.sandbox_mgr = SandboxManager(
            str(self.base_dir),
            self.validator,
            ignore_patterns=tuple(config.sandbox_copy_ignore),
        )
        self.env_mgr = EnvManager(
            project_root=str(Path(__file__).resolve().parent.parent),
            backend=config.env_backend,
        )
        self.llm = get_llm_provider(config)
        self.messenger = get_messenger(config)
        self.runner = Runner(
            str(self.base_dir),
            self.validator,
            env_manager=self.env_mgr,
            venv_path=config.runner_venv_path,
            conda_env=config.runner_default_env,
        )
        self.watcher = Watcher(
            str(self.base_dir),
            poll_interval=config.watcher_poll_interval,
            heartbeat_timeout=config.watcher_heartbeat_timeout,
            status_update_interval=config.watcher_status_update_interval,
        )
        self.summarizer = Summarizer(
            self.llm, self.validator, str(self.base_dir),
            log_tail_lines=config.report_log_tail,
        )
        self.git_mgr = GitManager(str(self.base_dir))

        # State
        self.state = AgentState.IDLE
        self.current_trial: TrialInfo | None = None
        self.active_proc: subprocess.Popen | None = None
        self.conversation_history: list[dict] = []

        # Thread safety: protects self.state and current_trial.status mutations.
        # _watch_experiment() runs in a daemon thread and transitions state.
        # The main loop reads state in _handle_message().
        self._trial_lock = threading.Lock()

        # Restore active trial from .trials.jsonl on startup
        self._restore_state()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: AgentState, reason: str = "") -> None:
        """Transition to a new state. Must be called under _trial_lock or by the main thread."""
        old_state = self.state
        self.state = new_state
        logger.info(
            "State transition: %s -> %s (%s)",
            old_state.value, new_state.value, reason,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the agent event loop. Blocks until interrupted."""
        logger.info("ResearchClaw starting (base_dir=%s)", self.base_dir)

        # Bootstrap managed experiment environment (creates env_000 if needed)
        try:
            self.env_mgr.bootstrap()
        except Exception as e:
            logger.warning("EnvManager bootstrap failed: %s", e)
        self.messenger.send(
            f"ResearchClaw online. [{self.state.value.upper()}]\n"
            + (
                "No active trial. Say 'start a new trial' to begin."
                if self.state == AgentState.IDLE
                else f"Resuming {self.current_trial.trial_name} "
                     f"(state: {self.state.value}, trial status: {self.current_trial.status})."
                if self.current_trial
                else "No active trial. Say 'start a new trial' to begin."
            )
        )

        while True:
            # Poll for incoming messages (non-blocking with short timeout)
            msg = self.messenger.receive(timeout=float(self.config.watcher_poll_interval))
            if msg:
                try:
                    self._handle_message(msg.strip())
                except Exception as e:
                    logger.exception("Error handling message: %s", e)
                    self.messenger.send(f"Internal error: {e}\nSee logs for details.")

    # ------------------------------------------------------------------
    # Message routing — slash commands + LLM for everything else
    # ------------------------------------------------------------------

    def _handle_message(self, msg: str) -> None:
        """Route an incoming researcher message.

        Slash commands (/status, /push, etc.) are handled directly.
        Everything else goes to the LLM for natural language understanding.
        """
        # 1. Check for explicit slash commands (exact prefix match)
        first_token = msg.strip().split()[0].lower() if msg.strip() else ""
        if first_token in _SLASH_COMMANDS:
            method_name, takes_msg = _SLASH_COMMANDS[first_token]
            handler = getattr(self, method_name)
            if takes_msg:
                handler(msg)
            else:
                handler()
            return

        # 2. Everything else → LLM conversation (works in ALL states)
        self._llm_conversation(msg)

    # ------------------------------------------------------------------
    # Trial lifecycle
    # ------------------------------------------------------------------

    def _start_trial(self, msg: str) -> None:
        """Create a new trial sandbox, transition to RESEARCH state."""
        if self.state != AgentState.IDLE:
            if self.current_trial:
                self.messenger.send(
                    f"[{self.state.value.upper()}] There's already an active trial: "
                    f"{self.current_trial.trial_name}.\n"
                    "Finish the current trial first (approve/reject), then start a new one."
                )
            else:
                self.messenger.send(
                    f"[{self.state.value.upper()}] Cannot start a new trial in this state."
                )
            return

        # Extract research goal from message
        goal = msg
        for prefix in ("start a new trial", "new trial", "start trial", "let's try", "let's"):
            if goal.lower().startswith(prefix):
                goal = goal[len(prefix):].strip().lstrip(",").strip()
                break

        self.messenger.send("Creating new trial sandbox...")
        try:
            trial = self.sandbox_mgr.create_trial(goal=goal)
        except FileNotFoundError as e:
            self.messenger.send(f"Error: {e}")
            return

        # Link trial to the current managed environment
        if self.env_mgr.current_env_id >= 0:
            trial.env_id = self.env_mgr.current_env_id

        self.current_trial = trial
        self.conversation_history = []

        with self._trial_lock:
            self._transition_to(AgentState.RESEARCH, f"started {trial.trial_name}")

        self.messenger.send(
            f"[RESEARCH] Started {trial.trial_name} for {trial.date}.\n"
            f"Copied codebase to `{trial.sandbox_path}`.\n"
            f"Goal: {goal or '(not specified)'}\n\n"
            "What should I try? I'll read the codebase and propose changes."
        )

        # Auto-start LLM conversation to propose changes
        if goal:
            self._llm_conversation(f"Research goal: {goal}. Please explore the codebase and propose how to implement this.")

    def _approve_trial(self, msg: str) -> None:
        """Merge approved trial into github_codes/ and finalize."""
        if self.state != AgentState.AWAITING_APPROVAL:
            self.messenger.send(
                f"[{self.state.value.upper()}] Cannot approve right now — "
                "trial must be in AWAITING_APPROVAL state."
            )
            return

        if not self.current_trial:
            self.messenger.send("No active trial to approve.")
            return

        # Show diff and require explicit confirmation before touching github_codes/
        diff = self.git_mgr.get_diff(self.current_trial)
        confirmed = self.messenger.confirm(
            f"Merge {self.current_trial.trial_name} into github_codes/?\n\n"
            f"Changes:\n{diff}"
        )
        if not confirmed:
            self.messenger.send("Merge cancelled. Trial remains in AWAITING_APPROVAL state.")
            return

        # Extract commit message from user's message, or generate one
        commit_msg = self.current_trial.goal or f"Trial {self.current_trial.trial_name} changes"

        self.messenger.send(f"Merging {self.current_trial.trial_name} into github_codes/...")
        try:
            self.git_mgr.authorize_merge()
            commit_hash = self.git_mgr.merge_trial(self.current_trial, commit_msg)
        except Exception as e:
            self.messenger.send(f"Merge failed: {e}")
            return

        with self._trial_lock:
            self.sandbox_mgr.finalize_trial(self.current_trial, "approved")
            self._transition_to(AgentState.IDLE, f"{self.current_trial.trial_name} approved")

        self.messenger.send(
            f"✅ {self.current_trial.trial_name} approved and merged.\n"
            f"Commit: `{commit_hash}`"
        )

        # Update trial summary
        self._update_trial_summary(commit_hash=commit_hash)

        # Ask researcher before pushing
        push_confirmed = self.messenger.confirm("Push to remote now?")
        if push_confirmed:
            self.messenger.send("Pushing to remote...")
            try:
                push_output = self.git_mgr.push()
                self.messenger.send(
                    f"✅ Pushed.\n"
                    f"{push_output.strip() or 'ok'}\n\n"
                    "[IDLE] Start a new trial when ready."
                )
            except Exception as push_err:
                logger.warning("Push failed after merge: %s", push_err)
                self.messenger.send(
                    f"⚠️ Push failed: {push_err}\n"
                    "Say 'push to github' to retry when ready."
                )
        else:
            self.messenger.send(
                "[IDLE] Not pushed. Say 'push to github' when ready, or start a new trial."
            )

    def _reject_trial(self) -> None:
        """Reject trial — preserve sandbox for reference but do not merge."""
        if self.state not in (AgentState.RESEARCH, AgentState.AWAITING_APPROVAL):
            self.messenger.send(
                f"[{self.state.value.upper()}] Cannot reject — "
                "must be in RESEARCH or AWAITING_APPROVAL state."
            )
            return

        if not self.current_trial:
            self.messenger.send("No active trial to reject.")
            return

        trial_name = self.current_trial.trial_name
        sandbox = self.current_trial.sandbox_path

        with self._trial_lock:
            self.sandbox_mgr.finalize_trial(self.current_trial, "rejected")
            self._transition_to(AgentState.IDLE, f"{trial_name} rejected")

        self.messenger.send(
            f"❌ {trial_name} rejected.\n"
            f"Code not merged. Sandbox preserved at `{sandbox}` for reference.\n\n"
            "[IDLE] Start a new trial when ready."
        )

        # Update trial summary
        self._update_trial_summary()

    def _continue_trial(self) -> None:
        """Go back to RESEARCH to iterate on the current trial."""
        if self.state != AgentState.AWAITING_APPROVAL:
            self.messenger.send(
                f"[{self.state.value.upper()}] Cannot continue — "
                "must be in AWAITING_APPROVAL state."
            )
            return

        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        with self._trial_lock:
            self.sandbox_mgr.reactivate_trial(self.current_trial)
            self._transition_to(AgentState.RESEARCH, "continue iterating")

        # Append a system note to conversation history about the state change
        self.conversation_history.append({
            "role": "user",
            "content": (
                "[System note: The researcher chose to continue iterating on this trial. "
                "The previous experiment results are in REPORT.md. "
                "Propose next changes based on those results.]"
            ),
        })

        self.messenger.send(
            f"[RESEARCH] Continuing {self.current_trial.trial_name}. Sandbox is writable again.\n"
            "What changes should I make next?"
        )

    def _kill_experiment(self) -> None:
        """Kill the running experiment and transition back to RESEARCH."""
        if self.state != AgentState.EXECUTE:
            self.messenger.send("No experiment is running.")
            return

        if not self.active_proc:
            self.messenger.send("No experiment process found.")
            return

        confirmed = self.messenger.confirm(
            f"Kill experiment (PID {self.active_proc.pid})?"
        )
        if not confirmed:
            self.messenger.send("Experiment continues running.")
            return

        try:
            self.active_proc.terminate()
            self.active_proc.wait(timeout=10)
        except Exception as e:
            logger.warning("Error killing process: %s", e)
            try:
                self.active_proc.kill()
            except Exception:
                pass

        self.active_proc = None

        with self._trial_lock:
            # Trial stays "active" — we just go back to RESEARCH
            self._transition_to(AgentState.RESEARCH, "experiment killed by user")

        self.messenger.send(
            f"[RESEARCH] Experiment killed. Back in research mode.\n"
            "What changes should I make?"
        )

    def _push_to_github(self) -> None:
        """Push to remote — only on explicit researcher request."""
        self.messenger.send("Pushing to origin...")
        try:
            output = self.git_mgr.push()
            self.messenger.send(f"✅ Pushed.\n```\n{output.strip()}\n```")
        except Exception as e:
            self.messenger.send(f"Push failed: {e}")

    def _send_status(self) -> None:
        """Send current state, trial info, and git status."""
        state_str = f"Agent state: {self.state.value.upper()}"

        if self.current_trial:
            trial_info = (
                f"Trial: {self.current_trial.trial_name}\n"
                f"Trial status: {self.current_trial.status}\n"
                f"Started: {self.current_trial.started_at}\n"
                f"Goal: {self.current_trial.goal or '(not specified)'}"
            )
        else:
            trial_info = "No active trial."

        if self.active_proc and self.state == AgentState.EXECUTE:
            proc_info = f"Running experiment: PID {self.active_proc.pid}"
        else:
            proc_info = ""

        try:
            git_status = self.git_mgr.status()
        except Exception:
            git_status = "(git status unavailable)"

        parts = [state_str, trial_info]
        if proc_info:
            parts.append(proc_info)
        parts.append(git_status)
        self.messenger.send("\n\n".join(parts))

    # ------------------------------------------------------------------
    # RESEARCH_TRIAL_SUMMARY.md management
    # ------------------------------------------------------------------

    def _update_trial_summary(self, commit_hash: str | None = None) -> None:
        """Generate and prepend an entry to RESEARCH_TRIAL_SUMMARY.md after trial finalization."""
        if not self.current_trial:
            return

        summary_path = self.base_dir / "RESEARCH_TRIAL_SUMMARY.md"

        try:
            entry = self.summarizer.generate_trial_summary_entry(
                self.current_trial, commit_hash=commit_hash,
            )
        except Exception as e:
            logger.warning("Failed to generate trial summary entry: %s", e)
            return

        try:
            if summary_path.exists():
                existing = summary_path.read_text(errors="replace")
            else:
                existing = "# Research Trial Summary\n\nLast updated: (none)\n\n## Trial History\n\n"

            # Update the "Last updated" line
            now = datetime.now().isoformat()
            existing = re.sub(
                r"Last updated:.*",
                f"Last updated: {now}",
                existing,
                count=1,
            )

            # Remove the "(No trials yet.)" placeholder if present
            existing = existing.replace("(No trials yet.)\n", "")

            # Insert new entry right after "## Trial History\n\n"
            marker = "## Trial History\n\n"
            if marker in existing:
                idx = existing.index(marker) + len(marker)
                updated = existing[:idx] + entry + "\n" + existing[idx:]
            else:
                updated = existing + "\n" + entry + "\n"

            summary_path.write_text(updated)
            logger.info("RESEARCH_TRIAL_SUMMARY.md updated for %s", self.current_trial.trial_name)
        except Exception as e:
            logger.warning("Failed to write RESEARCH_TRIAL_SUMMARY.md: %s", e)

    def _send_trial_summary(self) -> None:
        """Read and send RESEARCH_TRIAL_SUMMARY.md to the researcher."""
        summary_path = self.base_dir / "RESEARCH_TRIAL_SUMMARY.md"
        if summary_path.exists():
            content = summary_path.read_text(errors="replace")
            self.messenger.send(content[:4000])
        else:
            self.messenger.send("No trial summary yet. Start and complete a trial first.")

    # ------------------------------------------------------------------
    # LLM conversation (works in all states)
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current context."""
        try:
            gpu_info = self.watcher.check_gpu()
            gpu_str = (
                f"GPU {gpu_info.get('utilization', 0):.0f}%, "
                f"{gpu_info.get('memory_used_mb', 0):.0f}/{gpu_info.get('memory_total_mb', 0):.0f}MB"
                if gpu_info else "N/A"
            )
        except Exception:
            gpu_str = "N/A"

        try:
            repo_info = self.git_mgr.status()
        except Exception:
            repo_info = "(unavailable)"

        memory_path = self.base_dir / "MEMORY.md"
        agent_memory = memory_path.read_text(errors="replace").strip() if memory_path.exists() else "(empty)"

        summary_path = self.base_dir / "RESEARCH_TRIAL_SUMMARY.md"
        trial_summary = summary_path.read_text(errors="replace").strip() if summary_path.exists() else "(no prior trials)"

        trial_str = (
            f"{self.current_trial.trial_name} (status: {self.current_trial.status})"
            if self.current_trial
            else "(none)"
        )

        # Only include text marker instructions for providers without tool use
        text_markers = "" if self.llm.supports_tool_use else TEXT_MARKER_INSTRUCTIONS

        return SYSTEM_PROMPT.format(
            base_dir=str(self.base_dir),
            agent_state=self.state.value,
            current_trial=trial_str,
            repo_info=repo_info,
            gpu_info=gpu_str,
            agent_memory=agent_memory,
            state_instructions=STATE_INSTRUCTIONS.get(self.state, ""),
            trial_summary=trial_summary,
            text_marker_instructions=text_markers,
        )

    def _get_available_tools(self) -> list[dict]:
        """Return tool definitions filtered by current state."""
        # Read-only tools available in all states
        read_tools = {"read_file", "list_directory", "search_files", "propose_action"}
        # Write tools only in RESEARCH state
        write_tools = {"write_file", "run_command"}

        available = []
        for tool in AGENT_TOOLS:
            if tool["name"] in read_tools:
                available.append(tool)
            elif tool["name"] in write_tools and self.state == AgentState.RESEARCH:
                available.append(tool)
        return available

    def _llm_conversation(self, user_message: str) -> None:
        """Feed message to LLM and handle the response. Works in all states."""
        system = self._build_system_prompt()

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        if self.llm.supports_tool_use:
            self._agentic_loop(system)
        else:
            self._text_marker_conversation(system)

    def _text_marker_conversation(self, system_prompt: str) -> None:
        """Single-call LLM conversation with text-based action markers (legacy path)."""
        try:
            response = self.llm.complete_with_context(
                system_prompt=system_prompt,
                messages=self.conversation_history,
                max_tokens=4096,
            )
        except Exception as e:
            self.messenger.send(f"LLM error: {e}")
            return

        self.conversation_history.append({"role": "assistant", "content": response})
        self._handle_llm_response(response)

    def _agentic_loop(self, system_prompt: str) -> None:
        """Iterative tool-use loop: LLM calls tools, we execute, feed results back."""
        max_iter = self.config.agent_max_iterations
        tools = self._get_available_tools()

        for _ in range(max_iter):
            try:
                llm_resp = self.llm.complete_with_tools(
                    system_prompt=system_prompt,
                    messages=self.conversation_history,
                    tools=tools,
                    max_tokens=4096,
                )
            except Exception as e:
                self.messenger.send(f"LLM error: {e}")
                return

            # No tool calls — pure text response, send and stop
            if not llm_resp.tool_calls:
                if llm_resp.text:
                    # Check for MEMORY_UPDATE in text even from tool-use providers
                    self._check_memory_update(llm_resp.text)
                    self.messenger.send(llm_resp.text)
                self.conversation_history.append({"role": "assistant", "content": llm_resp.text})
                break

            # Send any text part to the researcher
            if llm_resp.text:
                self._check_memory_update(llm_resp.text)
                self.messenger.send(llm_resp.text)

            # Record assistant message with tool calls
            self.conversation_history.append({
                "role": "assistant",
                "text": llm_resp.text,
                "content": llm_resp.text,
                "tool_calls": llm_resp.tool_calls,
            })

            # Execute each tool call
            for tc in llm_resp.tool_calls:
                result = self._execute_tool(tc)
                self.conversation_history.append({
                    "role": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            # If a tool triggered a state transition that should stop the loop
            # (e.g., run_command → EXECUTE, or propose_action → state change)
            if self.state == AgentState.EXECUTE:
                break

            # Refresh tools in case state changed (e.g., trial started)
            tools = self._get_available_tools()
        else:
            self.messenger.send(
                "(Reached max iterations — stopping. Send another message to continue.)"
            )

    def _check_memory_update(self, text: str) -> None:
        """Extract and apply MEMORY_UPDATE from text if present."""
        mem_match = re.search(
            r"MEMORY_UPDATE:\n(.*?)(?=\nRUN_COMMAND:|\nCODE_CHANGE:|\Z)",
            text, re.DOTALL,
        )
        if mem_match:
            self._update_memory(mem_match.group(1).rstrip())

    # ------------------------------------------------------------------
    # Tool executors (for agentic loop)
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result string."""
        name = tool_call.name
        args = tool_call.arguments
        try:
            if name == "read_file":
                return self._tool_read_file(args.get("path", ""))
            elif name == "list_directory":
                return self._tool_list_directory(args.get("path", ""))
            elif name == "search_files":
                return self._tool_search_files(args.get("pattern", ""), args.get("path", "github_codes"))
            elif name == "write_file":
                return self._tool_write_file(args.get("path", ""), args.get("content", ""))
            elif name == "run_command":
                return self._tool_run_command(args.get("command", ""))
            elif name == "propose_action":
                return self._tool_propose_action(args.get("action", ""), args.get("detail", ""))
            else:
                return f"Unknown tool: {name}"
        except PermissionError as e:
            return f"BLOCKED: {e}"
        except Exception as e:
            logger.warning("Tool %s error: %s", name, e)
            return f"Error: {e}"

    def _tool_read_file(self, rel_path: str) -> str:
        """Read a file, validated through PathValidator."""
        if not rel_path:
            return "Error: path is required"
        abs_path = self.base_dir / rel_path
        self.validator.validate_read(str(abs_path))
        if not abs_path.exists():
            return f"File not found: {rel_path}"
        if abs_path.is_dir():
            return f"Path is a directory. Use list_directory instead: {rel_path}"
        content = abs_path.read_text(errors="replace")
        if len(content) > 10000:
            return content[:10000] + f"\n... (truncated, {len(content)} chars total)"
        return content

    def _tool_list_directory(self, rel_path: str) -> str:
        """List files in a directory."""
        if not rel_path:
            return "Error: path is required"
        abs_path = self.base_dir / rel_path
        self.validator.validate_read(str(abs_path))
        if not abs_path.exists():
            return f"Directory not found: {rel_path}"
        if not abs_path.is_dir():
            return f"Not a directory: {rel_path}"
        entries = sorted(abs_path.iterdir())
        lines = []
        for e in entries[:200]:  # cap at 200 entries
            suffix = "/" if e.is_dir() else ""
            lines.append(f"{e.name}{suffix}")
        result = "\n".join(lines)
        if len(entries) > 200:
            result += f"\n... ({len(entries)} entries total)"
        return result

    def _tool_search_files(self, pattern: str, rel_path: str) -> str:
        """Search for a regex pattern in files under a directory."""
        if not pattern:
            return "Error: pattern is required"
        abs_path = self.base_dir / rel_path
        self.validator.validate_read(str(abs_path))
        if not abs_path.exists():
            return f"Directory not found: {rel_path}"

        import subprocess as sp
        try:
            result = sp.run(
                ["grep", "-rn", "--include=*.py", "--include=*.yaml", "--include=*.yml",
                 "--include=*.json", "--include=*.md", "--include=*.txt", "--include=*.cfg",
                 "--include=*.toml", "--include=*.sh", "-E", pattern, str(abs_path)],
                capture_output=True, text=True, timeout=15,
            )
            output = result.stdout.strip()
            if not output:
                return f"No matches found for pattern '{pattern}' in {rel_path}"
            # Truncate
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output
        except sp.TimeoutExpired:
            return "Search timed out (>15s). Try a more specific pattern or path."
        except Exception as e:
            return f"Search error: {e}"

    def _tool_write_file(self, rel_path: str, content: str) -> str:
        """Write file in sandbox — requires RESEARCH state, asks confirmation."""
        if self.state != AgentState.RESEARCH or not self.current_trial:
            return "ERROR: Can only write files in RESEARCH state with an active trial."
        if not rel_path:
            return "Error: path is required"

        abs_path = self.base_dir / self.current_trial.sandbox_path / rel_path
        self.validator.validate_write(str(abs_path))

        # Show diff to researcher
        if abs_path.exists():
            old_lines = abs_path.read_text(errors="replace").splitlines(keepends=True)
            new_lines = content.splitlines(keepends=True)
            import difflib
            diff = "".join(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
            ))
            preview = diff[:2000] if diff else "(no changes)"
        else:
            preview = f"(new file: {len(content)} chars)"

        confirmed = self.messenger.confirm(
            f"Write to `{rel_path}`?\n```diff\n{preview}\n```"
        )
        if not confirmed:
            return "Write cancelled by researcher."

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content)
        return f"Written: {rel_path} ({len(content)} chars)"

    def _tool_run_command(self, cmd: str) -> str:
        """Run command — delegates to _prompt_and_run()."""
        if self.state != AgentState.RESEARCH:
            return "ERROR: Can only run commands in RESEARCH state."
        if not cmd:
            return "Error: command is required"
        self._prompt_and_run(cmd)
        return f"Command submitted: {cmd}"

    def _tool_propose_action(self, action: str, detail: str) -> str:
        """Propose a lifecycle action."""
        if action == "start_trial":
            if self.state != AgentState.IDLE:
                return f"Cannot start trial in {self.state.value} state."
            self._start_trial(detail or "")
            return "Trial start initiated."
        elif action == "approve":
            if self.state != AgentState.AWAITING_APPROVAL:
                return f"Cannot approve in {self.state.value} state."
            self._approve_trial(detail or "approve")
            return "Approval initiated."
        elif action == "reject":
            if self.state not in (AgentState.RESEARCH, AgentState.AWAITING_APPROVAL):
                return f"Cannot reject in {self.state.value} state."
            self._reject_trial()
            return "Rejection initiated."
        elif action == "continue":
            if self.state != AgentState.AWAITING_APPROVAL:
                return f"Cannot continue in {self.state.value} state."
            self._continue_trial()
            return "Continue initiated."
        elif action == "push":
            self._push_to_github()
            return "Push initiated."
        else:
            return f"Unknown action: {action}"

    def _handle_llm_response(self, response: str) -> None:
        """Parse LLM response for action markers and execute or prompt for approval.

        This is the legacy text-marker path for providers without native tool use.
        """
        # MEMORY_UPDATE is handled silently first — no researcher approval needed.
        mem_match = re.search(
            r"MEMORY_UPDATE:\n(.*?)(?=\nRUN_COMMAND:|\nCODE_CHANGE:|\Z)",
            response, re.DOTALL,
        )
        if mem_match:
            self._update_memory(mem_match.group(1).rstrip())
            response = (response[:mem_match.start()] + response[mem_match.end():]).strip()
            if not response:
                return

        # Check for lifecycle action markers (new)
        if "START_TRIAL:" in response:
            match = re.search(r"START_TRIAL:\s*(.+?)(?:\n|$)", response)
            if match and self.state == AgentState.IDLE:
                display = response[:match.start()].strip()
                if display:
                    self.messenger.send(display)
                self._start_trial(match.group(1).strip())
                return

        if "APPROVE_TRIAL" in response and self.state == AgentState.AWAITING_APPROVAL:
            display = response.replace("APPROVE_TRIAL", "").strip()
            if display:
                self.messenger.send(display)
            self._approve_trial("approve")
            return

        if "REJECT_TRIAL" in response and self.state in (AgentState.RESEARCH, AgentState.AWAITING_APPROVAL):
            display = response.replace("REJECT_TRIAL", "").strip()
            if display:
                self.messenger.send(display)
            self._reject_trial()
            return

        if "CONTINUE_TRIAL" in response and self.state == AgentState.AWAITING_APPROVAL:
            display = response.replace("CONTINUE_TRIAL", "").strip()
            if display:
                self.messenger.send(display)
            self._continue_trial()
            return

        if "PUSH_TO_REMOTE" in response:
            display = response.replace("PUSH_TO_REMOTE", "").strip()
            if display:
                self.messenger.send(display)
            self._push_to_github()
            return

        # Check for REPORT trigger
        if "REPORT" in response and not any(m in response for m in ("RUN_COMMAND:", "CODE_CHANGE:")):
            self._generate_and_send_report()
            return

        # Check for RUN_COMMAND — only allowed in RESEARCH state
        run_match = re.search(r"RUN_COMMAND:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        if run_match:
            if self.state != AgentState.RESEARCH:
                self.messenger.send(
                    f"[{self.state.value.upper()}] Cannot run commands in this state."
                )
                return
            cmd = run_match.group(1).strip().strip("`")
            # Strip everything after RUN_COMMAND from displayed response
            display = response[:run_match.start()].strip()
            if display:
                self.messenger.send(display)
            self._prompt_and_run(cmd)
            return

        # Check for CODE_CHANGE — only allowed in RESEARCH state
        code_match = re.search(
            r"CODE_CHANGE:\s*(\S+)\n(.*?)(?=\nCODE_CHANGE:|\nRUN_COMMAND:|\Z)",
            response, re.DOTALL,
        )
        if code_match:
            if self.state != AgentState.RESEARCH:
                self.messenger.send(
                    f"[{self.state.value.upper()}] Cannot apply code changes in this state."
                )
                return
            rel_path = code_match.group(1)
            new_content = code_match.group(2)
            display = response[:code_match.start()].strip()
            if display:
                self.messenger.send(display)
            self._prompt_and_apply_change(rel_path, new_content)
            return

        # Pure text response — just send it
        self.messenger.send(response)

    def _update_memory(self, content: str) -> None:
        """Overwrite MEMORY.md with new content. No researcher approval required."""
        memory_path = self.base_dir / "MEMORY.md"
        try:
            self.validator.validate_write(str(memory_path))
            memory_path.write_text(content + "\n")
            logger.info("Agent memory updated (%d chars)", len(content))
        except Exception as e:
            logger.warning("Failed to write MEMORY.md: %s", e)

    def _prompt_and_run(self, cmd: str) -> None:
        """Show the command to the researcher and run it if they approve."""
        # Validate paths before asking
        try:
            self.validator.validate_shell_command(cmd)
        except PermissionError as e:
            self.messenger.send(f"BLOCKED — command would write outside sandbox:\n```\n{cmd}\n```\nViolation: {e}")
            return

        # Guard: no running experiment already
        if self.active_proc is not None:
            self.messenger.send("An experiment is already running. Kill it first or wait for it to finish.")
            return

        # Intercept environment-mutating commands (pip/conda install).
        # These are handled synchronously by EnvManager (fork env + apply),
        # not routed through the Runner/Watcher async pipeline.
        if self.env_mgr.is_env_mutation(cmd):
            confirmed = self.messenger.confirm(
                f"This command modifies the environment:\n```\n{cmd}\n```\n"
                f"A new environment (env_{self.env_mgr.current_env_id + 1:03d}) "
                f"will be created. Proceed?"
            )
            if not confirmed:
                self.messenger.send("Skipped.")
                return
            try:
                trial_name = self.current_trial.trial_name if self.current_trial else ""
                self.env_mgr.apply_mutation(cmd, trial_name)
                if self.current_trial:
                    self.current_trial.env_id = self.env_mgr.current_env_id
                    self.current_trial.command_history.append(cmd)
                self.messenger.send(
                    f"Environment updated -> env_{self.env_mgr.current_env_id:03d}\n"
                    f"Package operation complete."
                )
            except Exception as e:
                self.messenger.send(f"Environment operation failed: {e}")
            return

        confirmed = self.messenger.confirm(f"Run this command?\n```\n{cmd}\n```")
        if not confirmed:
            self.messenger.send("Skipped.")
            return

        if self.current_trial:
            self.current_trial.command_history.append(cmd)

        self.messenger.send(f"Starting... PID pending.")
        try:
            proc = self.runner.run_async(cmd, self.current_trial)
            self.active_proc = proc

            # Transition to EXECUTE
            with self._trial_lock:
                self._transition_to(AgentState.EXECUTE, f"experiment started (PID {proc.pid})")

            self.messenger.send(
                f"[EXECUTE] Started. PID {proc.pid}. I'll ping you when it's done."
            )

            # Watch in background thread
            thread = threading.Thread(
                target=self._watch_experiment,
                args=(proc, self.current_trial),
                daemon=True,
            )
            thread.start()
        except Exception as e:
            self.messenger.send(f"Failed to start: {e}")

    def _prompt_and_apply_change(self, rel_path: str, new_content: str) -> None:
        """Show the proposed file change and apply it if approved."""
        if not self.current_trial:
            return

        abs_path = self.base_dir / self.current_trial.sandbox_path / rel_path

        # Show existing content vs new content as diff
        if abs_path.exists():
            old_lines = abs_path.read_text(errors="replace").splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            import difflib
            diff = "".join(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            ))
        else:
            diff = f"(new file)\n+++ {rel_path}\n" + "\n".join(f"+ {l}" for l in new_content.splitlines())

        confirmed = self.messenger.confirm(f"Apply this change to `{rel_path}`?\n```diff\n{diff[:2000]}\n```")
        if not confirmed:
            self.messenger.send("Change not applied.")
            return

        try:
            validated = self.validator.validate_write(str(abs_path))
            validated.parent.mkdir(parents=True, exist_ok=True)
            validated.write_text(new_content)
            self.messenger.send(f"Applied: `{rel_path}`")
        except PermissionError as e:
            self.messenger.send(f"BLOCKED: {e}")

    def _watch_experiment(self, proc: subprocess.Popen, trial: TrialInfo) -> None:
        """Background thread: watch experiment and send status updates."""
        try:
            for status in self.watcher.watch(proc, trial):
                msg = self.summarizer.format_status_message(status)
                self.messenger.send(msg)

                if status.event in (ExperimentEvent.FINISHED, ExperimentEvent.CRASHED):
                    self.active_proc = None

                    if status.event == ExperimentEvent.FINISHED:
                        self._generate_and_send_report()
                    else:
                        # CRASHED — still transition to AWAITING_APPROVAL so user can decide
                        with self._trial_lock:
                            self.sandbox_mgr.mark_review(trial)
                            self._transition_to(
                                AgentState.AWAITING_APPROVAL,
                                f"experiment {status.event.value}",
                            )
                        self.messenger.send(
                            f"[AWAITING_APPROVAL] Experiment crashed.\n"
                            "What would you like to do?\n"
                            "(a) approve — merge current code to main anyway\n"
                            "(b) reject — discard (keep sandbox for reference)\n"
                            "(c) continue — keep iterating on this trial"
                        )
                    break
        except Exception as e:
            logger.exception("Watcher error: %s", e)
            self.messenger.send(f"Watcher error: {e}")

    def _generate_and_send_report(self) -> None:
        """Generate REPORT.md and send a summary to the researcher."""
        if not self.current_trial:
            return

        try:
            diff = self.git_mgr.get_full_diff(self.current_trial)
            report = self.summarizer.generate_report(self.current_trial, diff)

            # Send a brief summary (not the full report)
            summary_lines = report.split("\n")[:40]
            summary = "\n".join(summary_lines)

            with self._trial_lock:
                self.sandbox_mgr.mark_review(self.current_trial)
                self._transition_to(
                    AgentState.AWAITING_APPROVAL,
                    "report generated",
                )

            self.messenger.send(
                f"REPORT.md generated for {self.current_trial.trial_name}.\n\n"
                f"```\n{summary}\n...\n```\n\n"
                "[AWAITING_APPROVAL] What would you like to do?\n"
                "(a) approve — merge to main\n"
                "(b) reject — discard (keep sandbox for reference)\n"
                "(c) continue — keep iterating on this trial"
            )
        except Exception as e:
            logger.exception("Report generation error: %s", e)
            self.messenger.send(f"Report generation failed: {e}\nTrial results are in the sandbox.")

    # ------------------------------------------------------------------
    # State restoration
    # ------------------------------------------------------------------

    def _restore_state(self) -> None:
        """On startup, check .trials.jsonl for any active or review trial to resume."""
        try:
            # Check for active trial first (was in RESEARCH or EXECUTE)
            active = self.sandbox_mgr.get_active_trial()
            if active:
                self.current_trial = active
                self.validator.set_trial(active)
                self._transition_to(AgentState.RESEARCH, "restored active trial")
                logger.info("Restored active trial: %s", active.trial_name)
                return

            # Check for trial in review (was in AWAITING_APPROVAL)
            review = self.sandbox_mgr.get_review_trial()
            if review:
                self.current_trial = review
                self.validator.set_trial(review)
                self._transition_to(AgentState.AWAITING_APPROVAL, "restored trial in review")
                logger.info("Restored review trial: %s", review.trial_name)
                return

        except Exception as e:
            logger.warning("Could not restore trial state: %s", e)


def main() -> None:
    """Entry point — load config and run the agent."""
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    config = load_config(config_path)
    setup_logging(config.base_dir)

    agent = ResearchClaw(config)
    agent.run()


if __name__ == "__main__":
    main()
