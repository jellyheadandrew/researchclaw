from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm import TrialAborted
from researchclaw.fsm.states import State
from researchclaw.fsm._shared import persist_autopilot
from researchclaw.models import TrialMeta
from researchclaw.sandbox import SandboxManager

# Backward-compatibility alias — tests import the private name via
# ``from researchclaw.fsm.decide import _persist_autopilot``.
_persist_autopilot = persist_autopilot


# --- Decision options ---

DECISION_OPTIONS = """\
What would you like to do next?

  (1) New experiment  — start a new trial
  (2) View summary    — browse trial history
  (3) Settings        — view/edit configuration
  (4) Merge           — merge experiment code (not yet implemented)
  (5) Quit            — exit ResearchClaw\
"""

OPTION_MAP: dict[str, State | str] = {
    "1": State.EXPERIMENT_PLAN,
    "2": State.VIEW_SUMMARY,
    "3": State.SETTINGS,
    "4": State.MERGE_LOOP,
    "5": "quit",
}


def _build_trial_summary(trial_dir: Path, meta: TrialMeta) -> str:
    """Build a brief summary of the completed trial for display."""
    parts = [
        f"**Trial**: {trial_dir.name}",
        f"**Status**: {meta.status}",
    ]

    if meta.experiment_exit_code is not None:
        parts.append(f"**Experiment exit code**: {meta.experiment_exit_code}")
    if meta.eval_exit_code is not None:
        parts.append(f"**Eval exit code**: {meta.eval_exit_code}")
    if meta.decision:
        parts.append(f"**Previous decision**: {meta.decision}")

    # Try to show a snippet from REPORT.md if it exists
    report_path = trial_dir / "REPORT.md"
    if report_path.exists():
        report_text = report_path.read_text().strip()
        if report_text:
            # Show first 500 chars of report
            snippet = report_text[:500]
            if len(report_text) > 500:
                snippet += "\n[... truncated]"
            parts.append(f"\n**Report preview**:\n{snippet}")

    return "\n".join(parts)


def _parse_user_choice(text: str) -> str | None:
    """Parse user input to extract a valid option number.

    Returns the option key ('1'-'5') or None if invalid.
    """
    stripped = text.strip()
    # Direct number input
    if stripped in OPTION_MAP:
        return stripped

    # Try to extract first digit
    for ch in stripped:
        if ch in OPTION_MAP:
            return ch

    return None


def handle_decide(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the DECIDE state.

    Shows trial summary, presents decision options, and returns the chosen
    next state. In autopilot mode, automatically selects new experiment.

    Agent has read-only access in this state — no write or execute.

    Args:
        trial_dir: Current trial directory.
        meta: Current trial metadata.
        config: ResearchClaw configuration.
        chat_interface: Chat interface for user interaction.

    Returns:
        The next FSM state based on user choice.
    """
    # Show trial summary and autopilot status
    if chat_interface is not None:
        summary = _build_trial_summary(trial_dir, meta)
        autopilot_status = "ON" if config.autopilot else "OFF"
        chat_interface.send_status(
            f"[DECIDE] Trial complete. (Autopilot: {autopilot_status})"
        )
        chat_interface.send(summary)

    # Autopilot mode: auto-select new experiment
    if config.autopilot:
        meta.decision = "new_experiment"
        meta.decision_reasoning = "Autopilot mode: automatically starting new experiment."
        if chat_interface is not None:
            chat_interface.send_status(
                "[Autopilot] Automatically starting new experiment."
            )
        return State.EXPERIMENT_PLAN

    # Interactive mode: present options
    if chat_interface is not None:
        chat_interface.send(DECISION_OPTIONS)

    while True:
        if chat_interface is None:
            # No chat interface — default to new experiment
            meta.decision = "new_experiment"
            return State.EXPERIMENT_PLAN

        user_input = chat_interface.receive()

        # Handle slash commands
        from researchclaw.repl import SlashCommand, UserMessage

        if isinstance(user_input, SlashCommand):
            if user_input.name == "/quit":
                raise SystemExit("User quit via /quit")
            if user_input.name == "/abort":
                raise TrialAborted("User aborted trial at DECIDE")
            if user_input.name == "/status":
                if chat_interface is not None:
                    from researchclaw.status import render_status_string
                    project_dir = trial_dir.parent.parent.parent
                    status_output = render_status_string(project_dir)
                    chat_interface.send(status_output)
                continue
            if user_input.name == "/autopilot":
                # Confirmation prompt
                if chat_interface is not None:
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
                    if chat_interface is not None:
                        chat_interface.send("Autopilot not enabled.")
                    continue
                config.autopilot = True
                project_dir = trial_dir.parent.parent.parent
                _persist_autopilot(config, project_dir)
                meta.decision = "new_experiment"
                meta.decision_reasoning = "Autopilot enabled at DECIDE: starting new experiment."
                if chat_interface is not None:
                    chat_interface.send("Autopilot enabled. Starting new experiment.")
                return State.EXPERIMENT_PLAN
            if user_input.name == "/autopilot-stop":
                config.autopilot = False
                project_dir = trial_dir.parent.parent.parent
                _persist_autopilot(config, project_dir)
                if chat_interface is not None:
                    chat_interface.send("Autopilot disabled.")
                continue
            # Other slash commands — inform user
            if chat_interface is not None:
                chat_interface.send(f"Command {user_input.name} not available in DECIDE state. Please choose an option (1-5).")
            continue

        # Parse user message as option choice
        text = user_input.text if isinstance(user_input, UserMessage) else str(user_input)
        choice = _parse_user_choice(text)

        if choice is None:
            if chat_interface is not None:
                chat_interface.send("Please enter a number (1-5).")
            continue

        result = OPTION_MAP[choice]

        if result == "quit":
            meta.decision = "quit"
            raise SystemExit("User chose to quit at DECIDE")

        # Valid state transition
        assert isinstance(result, State)
        meta.decision = choice
        meta.decision_reasoning = f"User chose option {choice}"
        return result
