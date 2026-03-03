from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm import TrialAborted
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager


def _persist_autopilot(config: ResearchClawConfig, project_dir: Path) -> None:
    """Persist the current autopilot setting to the project config file."""
    project_config_path = (
        SandboxManager.sandbox_path(project_dir)
        / "project_settings"
        / "researchclaw.yaml"
    )
    if project_config_path.exists():
        saved = ResearchClawConfig.load_from_yaml(project_config_path)
    else:
        saved = ResearchClawConfig()
    saved.autopilot = config.autopilot
    saved.save_to_yaml(project_config_path)


# --- System prompts for agents ---

PLANNING_AGENT_SYSTEM = """\
You are the Planning Agent for ResearchClaw, a research experiment orchestrator.

Your role is to brainstorm experiment direction with the user and help draft an experiment plan.

Context about prior trials (if any):
{historian_context}

{search_context}

Guidelines:
- Help the user define a clear experiment: hypothesis, methodology, expected outcomes
- Consider what was learned from prior trials (if any)
- Be concise and actionable
- End every message with:
**If you approve plan, please type /approve. If not, please iterate.**
"""

HISTORIAN_AGENT_SYSTEM = """\
You are the Historian Agent for ResearchClaw. Summarize the following trial history \
concisely (max 3000 tokens). Focus on key findings, patterns, and lessons learned.

For ≤5 trials: provide per-trial summaries.
For >5 trials: summarize overall themes/trends, then detail only the 3 most recent trials.
"""

AUTOPILOT_PLAN_SYSTEM = """\
You are the Planning Agent for ResearchClaw in autopilot mode.

Context about prior trials (if any):
{historian_context}

Based on the trial history above, generate a complete experiment plan. Include:
1. Hypothesis
2. Methodology
3. Expected outcomes
4. Success criteria

Be concise and actionable. Output the plan in markdown format.
"""


# --- Historian agent ---

def _gather_trial_history(project_dir: Path) -> str:
    """Gather prior trial reports and logs for historian context.

    Returns raw text of prior trial history that can be summarized by the
    historian agent (or used directly if short enough).
    """
    sandbox = SandboxManager.sandbox_path(project_dir)
    experiments_dir = sandbox / "experiments"

    if not experiments_dir.is_dir():
        return ""

    trial_dirs = sorted(
        d for d in experiments_dir.iterdir()
        if d.is_dir() and "_trial_" in d.name
    )

    if not trial_dirs:
        return ""

    # Collect REPORT.md content from each trial
    parts: list[str] = []
    for td in trial_dirs:
        report_path = td / "REPORT.md"
        if report_path.exists():
            content = report_path.read_text().strip()
            if content:
                parts.append(f"### {td.name}\n{content}")

    # Also include EXPERIMENT_LOGS.md if it exists and has content
    logs_path = sandbox / "EXPERIMENT_LOGS.md"
    if logs_path.exists():
        logs_content = logs_path.read_text().strip()
        if logs_content:
            parts.append(f"### Experiment Logs\n{logs_content}")

    return "\n\n".join(parts)


def _build_historian_context(
    project_dir: Path,
    provider: Any | None = None,
) -> str:
    """Build historian context from prior trial history.

    If an LLM provider is available and history is long, uses the historian
    agent to summarize. Otherwise returns raw history (truncated if needed).
    """
    raw_history = _gather_trial_history(project_dir)
    if not raw_history:
        return "No prior trials."

    # If history is short enough, use it directly
    if len(raw_history) <= 3000:
        return raw_history

    # Try to summarize with LLM if available
    if provider is not None:
        try:
            summary = provider.chat(
                messages=[{"role": "user", "content": raw_history}],
                system=HISTORIAN_AGENT_SYSTEM,
            )
            return summary
        except Exception:
            pass

    # Fallback: truncate to ~3000 chars
    return raw_history[:3000] + "\n\n[... truncated]"


def _get_search_context() -> str:
    """Search agent stub — returns empty context in v1."""
    return ""


# --- Plan handler ---

def handle_experiment_plan(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the EXPERIMENT_PLAN state.

    In interactive mode: runs a chat loop with the planning agent.
    In autopilot mode: auto-generates a plan without user input.

    The user can:
    - Chat with the planning agent to refine the plan
    - Type /approve to approve the plan and advance to EXPERIMENT_IMPLEMENT
    - Type /quit to exit

    Returns:
        State.EXPERIMENT_IMPLEMENT on /approve
        State.DECIDE if user aborts planning
    """
    # Get LLM provider (may be None if not configured)
    provider = _get_provider_safe(config)

    # Build context
    project_dir = trial_dir.parent.parent  # experiments/{trial} -> sandbox -> project
    # Actually: trial_dir is under sandbox_researchclaw/experiments/{trial_name}
    # project_dir should be 3 levels up: trial_dir.parent = experiments/, .parent = sandbox, .parent = project
    project_dir = trial_dir.parent.parent.parent
    historian_context = _build_historian_context(project_dir, provider)
    search_context = _get_search_context()

    if config.autopilot:
        return _handle_autopilot_plan(
            trial_dir, meta, config, chat_interface, provider,
            historian_context, search_context,
        )

    return _handle_interactive_plan(
        trial_dir, meta, config, chat_interface, provider,
        historian_context, search_context,
    )


def _get_provider_safe(config: ResearchClawConfig) -> Any | None:
    """Try to get an LLM provider, return None if unavailable."""
    try:
        from researchclaw.llm.provider import get_provider
        return get_provider(config)
    except Exception:
        return None


def _handle_autopilot_plan(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
    provider: Any | None,
    historian_context: str,
    search_context: str,
) -> State:
    """Auto-generate a plan in autopilot mode."""
    if chat_interface is not None:
        chat_interface.send("[autopilot] Generating experiment plan...")

    if provider is not None:
        system = AUTOPILOT_PLAN_SYSTEM.format(
            historian_context=historian_context,
        )
        try:
            plan_content = provider.chat(
                messages=[{
                    "role": "user",
                    "content": "Generate an experiment plan based on the context above.",
                }],
                system=system,
            )
        except Exception as e:
            plan_content = (
                f"# Experiment Plan\n\n"
                f"*Auto-generated plan (LLM error: {e})*\n\n"
                f"## Context\n{historian_context}\n"
            )
    else:
        plan_content = (
            f"# Experiment Plan\n\n"
            f"*Auto-generated plan (no LLM available)*\n\n"
            f"## Context\n{historian_context}\n"
        )

    # Write PLAN.md
    _write_plan(trial_dir, plan_content)

    # Update meta
    meta.plan_approved_at = datetime.now(timezone.utc).isoformat()
    meta.status = "running"

    if chat_interface is not None:
        chat_interface.send(f"Plan written to {trial_dir.name}/PLAN.md")

    return State.EXPERIMENT_IMPLEMENT


def _handle_interactive_plan(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
    provider: Any | None,
    historian_context: str,
    search_context: str,
) -> State:
    """Run interactive planning chat loop."""
    if chat_interface is None:
        # No chat interface — skip to implement with empty plan
        _write_plan(trial_dir, "# Experiment Plan\n\n*No chat interface available.*\n")
        meta.plan_approved_at = datetime.now(timezone.utc).isoformat()
        meta.status = "running"
        return State.EXPERIMENT_IMPLEMENT

    # Build system prompt
    search_section = f"Search context:\n{search_context}" if search_context else ""
    system = PLANNING_AGENT_SYSTEM.format(
        historian_context=historian_context,
        search_context=search_section,
    )

    # Conversation accumulator
    messages: list[dict[str, str]] = []

    # Initial greeting
    if historian_context and historian_context != "No prior trials.":
        chat_interface.send(
            f"[EXPERIMENT_PLAN] Starting planning for {trial_dir.name}.\n"
            f"I have context from prior trials. Let's plan the next experiment.\n\n"
            f"**If you approve plan, please type /approve. If not, please iterate.**"
        )
    else:
        chat_interface.send(
            f"[EXPERIMENT_PLAN] Starting planning for {trial_dir.name}.\n"
            f"This is the first trial. What experiment would you like to run?\n\n"
            f"**If you approve plan, please type /approve. If not, please iterate.**"
        )

    # Track the last assistant response as the plan to write
    last_assistant_response: str = ""

    # Chat loop
    while True:
        user_input: ChatInput = chat_interface.receive()

        if isinstance(user_input, SlashCommand):
            if user_input.name == "/approve":
                # Write plan
                plan_content = last_assistant_response or _default_plan(messages)
                _write_plan(trial_dir, plan_content)
                meta.plan_approved_at = datetime.now(timezone.utc).isoformat()
                meta.status = "running"
                chat_interface.send(
                    f"Plan approved and written to {trial_dir.name}/PLAN.md"
                )
                return State.EXPERIMENT_IMPLEMENT

            elif user_input.name == "/quit":
                raise SystemExit("User quit during planning")

            elif user_input.name == "/abort":
                raise TrialAborted("User aborted trial during planning")

            elif user_input.name == "/autopilot":
                # Confirmation prompt
                chat_interface.send(
                    "Enable autopilot? The system will run experiments "
                    "unattended. Type 'yes' to confirm."
                )
                confirm = chat_interface.receive()
                confirm_text = ""
                if isinstance(confirm, UserMessage):
                    confirm_text = confirm.text.strip().lower()
                elif isinstance(confirm, SlashCommand):
                    confirm_text = confirm.name
                if confirm_text not in ("yes", "y"):
                    chat_interface.send("Autopilot not enabled.")
                    continue
                config.autopilot = True
                project_dir_for_persist = trial_dir.parent.parent.parent
                _persist_autopilot(config, project_dir_for_persist)
                chat_interface.send(
                    "Autopilot enabled. Auto-generating plan..."
                )
                return _handle_autopilot_plan(
                    trial_dir, meta, config, chat_interface, provider,
                    historian_context, search_context,
                )

            elif user_input.name == "/autopilot-stop":
                config.autopilot = False
                project_dir_for_persist = trial_dir.parent.parent.parent
                _persist_autopilot(config, project_dir_for_persist)
                chat_interface.send("Autopilot disabled.")
                continue

            else:
                # Other slash commands — inform user
                chat_interface.send(
                    f"Command {user_input.name} is not available during planning. "
                    f"Use /approve to approve the plan or /quit to exit."
                )
                continue

        # Regular user message
        user_text = user_input.text
        messages.append({"role": "user", "content": user_text})

        # Get LLM response
        if provider is not None:
            try:
                response = provider.chat(messages=messages, system=system)
            except Exception as e:
                response = f"*LLM error: {e}*\n\n**If you approve plan, please type /approve. If not, please iterate.**"
        else:
            response = (
                f"*No LLM provider available. Please describe your plan and type /approve when ready.*\n\n"
                f"**If you approve plan, please type /approve. If not, please iterate.**"
            )

        messages.append({"role": "assistant", "content": response})
        last_assistant_response = response
        chat_interface.send(response)


def _write_plan(trial_dir: Path, content: str) -> None:
    """Write PLAN.md to the trial directory."""
    plan_path = trial_dir / "PLAN.md"
    plan_path.write_text(content)


def _default_plan(messages: list[dict[str, str]]) -> str:
    """Build a default plan from conversation messages when no LLM response is available."""
    if not messages:
        return "# Experiment Plan\n\n*No plan content. Approved without discussion.*\n"

    parts = ["# Experiment Plan\n"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"**User**: {content}\n")
        elif role == "assistant":
            parts.append(f"**Assistant**: {content}\n")
    return "\n".join(parts)
