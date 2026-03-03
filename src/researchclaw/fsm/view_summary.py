from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.sandbox import SandboxManager


# Default number of recent trials to display
DEFAULT_DISPLAY_TRIALS = 10


def _parse_experiment_logs(project_dir: Path) -> list[dict[str, str]]:
    """Parse EXPERIMENT_LOGS.md into a list of trial entries.

    Each entry is a dict with keys: date, trial, summary, full_line.
    Format expected: '{YYYYMMDD} - trial_{N:03}: {summary}. Full Doc: [REPORT.md](...)'

    Returns entries in file order (oldest first).
    """
    logs_path = SandboxManager.sandbox_path(project_dir) / "EXPERIMENT_LOGS.md"
    if not logs_path.exists():
        return []

    text = logs_path.read_text().strip()
    if not text:
        return []

    entries: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match pattern: {YYYYMMDD} - trial_{NNN}: {summary}...
        match = re.match(
            r"(\d{8})\s*-\s*(trial_\d{3}):\s*(.+?)(?:\.\s*Full Doc:|\s*$)",
            line,
        )
        if match:
            entries.append({
                "date": match.group(1),
                "trial": match.group(2),
                "summary": match.group(3).strip().rstrip("."),
                "full_line": line,
            })

    return entries


def _get_unique_dates(entries: list[dict[str, str]]) -> list[str]:
    """Get unique dates from entries, preserving order (most recent last)."""
    seen: set[str] = set()
    dates: list[str] = []
    for entry in entries:
        d = entry["date"]
        if d not in seen:
            seen.add(d)
            dates.append(d)
    return dates


def _format_trial_table(entries: list[dict[str, str]]) -> str:
    """Format trial entries as a rich-style table string.

    Format: [{YYYYMMDD}] [trial_{N:03}] {summary}
    """
    if not entries:
        return "No trials found."

    lines: list[str] = []
    for entry in entries:
        lines.append(f"[{entry['date']}] [{entry['trial']}] {entry['summary']}")
    return "\n".join(lines)


def handle_view_summary(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the VIEW_SUMMARY state.

    Lists recent trials from EXPERIMENT_LOGS.md. User can browse older trials
    by date or exit back to DECIDE.

    Args:
        trial_dir: Current trial directory.
        meta: Current trial metadata.
        config: ResearchClaw configuration.
        chat_interface: Chat interface for user interaction.

    Returns:
        State.DECIDE to return to the decision menu.
    """
    project_dir = trial_dir.parent.parent.parent

    all_entries = _parse_experiment_logs(project_dir)

    if not all_entries:
        if chat_interface is not None:
            chat_interface.send("[VIEW_SUMMARY] No experiment logs found.")
        return State.DECIDE

    # Determine how many trials to display
    display_count = config.display_trials
    if display_count < 1:
        display_count = DEFAULT_DISPLAY_TRIALS

    # Show most recent N trials (entries are oldest-first, so take from end)
    recent = all_entries[-display_count:]
    recent_reversed = list(reversed(recent))  # most recent first

    if chat_interface is not None:
        header = f"[VIEW_SUMMARY] Showing {len(recent_reversed)} most recent trial(s):\n"
        table = _format_trial_table(recent_reversed)
        chat_interface.send(header + table)

        # Offer navigation options
        if len(all_entries) > display_count:
            chat_interface.send(
                f"\n{len(all_entries)} total trials. "
                "Type 'older' to browse by date, or 'back' to return to DECIDE."
            )
        else:
            chat_interface.send("\nType 'back' to return to DECIDE, or 'older' to browse by date.")

    # Interactive loop
    while True:
        if chat_interface is None:
            return State.DECIDE

        user_input = chat_interface.receive()

        from researchclaw.repl import SlashCommand, UserMessage

        if isinstance(user_input, SlashCommand):
            if user_input.name == "/quit":
                raise SystemExit("User quit via /quit")
            chat_interface.send(
                f"Command {user_input.name} not available in VIEW_SUMMARY. "
                "Type 'back' to return to DECIDE."
            )
            continue

        text = user_input.text if isinstance(user_input, UserMessage) else str(user_input)
        text_lower = text.strip().lower()

        if text_lower in ("back", "exit", "quit", "done", "q"):
            return State.DECIDE

        if text_lower == "older":
            _browse_by_date(all_entries, chat_interface)
            # After browsing by date, show the prompt again
            chat_interface.send("\nType 'back' to return to DECIDE, or 'older' to browse by date.")
            continue

        # Try to interpret as a date
        date_match = re.match(r"(\d{8})", text.strip())
        if date_match:
            date_str = date_match.group(1)
            date_entries = [e for e in all_entries if e["date"] == date_str]
            if date_entries:
                chat_interface.send(
                    f"\nTrials for {date_str}:\n" + _format_trial_table(date_entries)
                )
            else:
                chat_interface.send(f"No trials found for date {date_str}.")
            continue

        chat_interface.send("Type 'back' to return, 'older' to browse by date, or enter a date (YYYYMMDD).")


def _browse_by_date(
    entries: list[dict[str, str]],
    chat_interface: Any,
) -> None:
    """Show list of experiment dates. User picks a date to see its trials."""
    dates = _get_unique_dates(entries)
    dates_reversed = list(reversed(dates))  # most recent first

    lines = ["Available experiment dates:"]
    for i, d in enumerate(dates_reversed, 1):
        count = sum(1 for e in entries if e["date"] == d)
        lines.append(f"  ({i}) {d}  [{count} trial(s)]")
    lines.append("\nEnter a date (YYYYMMDD) or number to view trials, or 'back' to return.")

    chat_interface.send("\n".join(lines))

    from researchclaw.repl import SlashCommand, UserMessage

    while True:
        user_input = chat_interface.receive()

        if isinstance(user_input, SlashCommand):
            if user_input.name == "/quit":
                raise SystemExit("User quit via /quit")
            chat_interface.send("Type a date or 'back' to return.")
            continue

        text = user_input.text if isinstance(user_input, UserMessage) else str(user_input)
        text_lower = text.strip().lower()

        if text_lower in ("back", "exit", "quit", "done", "q"):
            return

        # Try number selection
        try:
            num = int(text.strip())
            if 1 <= num <= len(dates_reversed):
                selected_date = dates_reversed[num - 1]
                date_entries = [e for e in entries if e["date"] == selected_date]
                chat_interface.send(
                    f"\nTrials for {selected_date}:\n" + _format_trial_table(date_entries)
                )
                return
        except ValueError:
            pass

        # Try date input
        date_match = re.match(r"(\d{8})", text.strip())
        if date_match:
            date_str = date_match.group(1)
            date_entries = [e for e in entries if e["date"] == date_str]
            if date_entries:
                chat_interface.send(
                    f"\nTrials for {date_str}:\n" + _format_trial_table(date_entries)
                )
                return
            else:
                chat_interface.send(f"No trials found for date {date_str}. Try another date or 'back'.")
                continue

        chat_interface.send("Enter a date (YYYYMMDD), a number, or 'back' to return.")
