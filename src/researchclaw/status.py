from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from researchclaw.models import TrialMeta
from researchclaw.sandbox import SandboxManager


DEFAULT_STATUS_DISPLAY = 10


def _parse_log_summaries(project_dir: Path) -> dict[str, str]:
    """Parse EXPERIMENT_LOGS.md into a mapping of trial_name -> summary.

    Returns dict like: {"20260301_trial_001": "First experiment on transformers"}
    """
    logs_path = SandboxManager.sandbox_path(project_dir) / "EXPERIMENT_LOGS.md"
    if not logs_path.exists():
        return {}

    text = logs_path.read_text().strip()
    if not text:
        return {}

    summaries: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"(\d{8})\s*-\s*(trial_\d{3}):\s*(.+?)(?:\.\s*Full Doc:|\s*$)",
            line,
        )
        if match:
            date = match.group(1)
            trial = match.group(2)
            summary = match.group(3).strip().rstrip(".")
            trial_name = f"{date}_{trial}"
            summaries[trial_name] = summary

    return summaries


def _get_trial_dirs(project_dir: Path) -> list[Path]:
    """Get all trial directories, sorted by name (oldest first)."""
    experiments_dir = SandboxManager.sandbox_path(project_dir) / "experiments"
    if not experiments_dir.is_dir():
        return []
    return sorted(
        d for d in experiments_dir.iterdir()
        if d.is_dir() and "_trial_" in d.name
    )


def build_status_table(project_dir: Path, max_trials: int = DEFAULT_STATUS_DISPLAY) -> Table:
    """Build a rich Table showing trial status info.

    Columns: Trial Name, State, Status, Summary (1-line from EXPERIMENT_LOGS.md).
    Shows most recent max_trials trials.

    Args:
        project_dir: Project root directory.
        max_trials: Maximum number of trials to display.

    Returns:
        A rich Table object.
    """
    table = Table(title="ResearchClaw Trials")
    table.add_column("Trial", style="cyan")
    table.add_column("State", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Summary", style="white")

    trial_dirs = _get_trial_dirs(project_dir)
    if not trial_dirs:
        return table

    summaries = _parse_log_summaries(project_dir)

    # Show most recent N trials
    recent = trial_dirs[-max_trials:]
    recent_reversed = list(reversed(recent))  # most recent first

    for trial_dir in recent_reversed:
        try:
            meta = TrialMeta.from_json(trial_dir / "meta.json")
        except (FileNotFoundError, Exception):
            meta = TrialMeta()

        summary = summaries.get(trial_dir.name, "")
        table.add_row(
            trial_dir.name,
            meta.state,
            meta.status,
            summary,
        )

    return table


def render_status_string(project_dir: Path, max_trials: int = DEFAULT_STATUS_DISPLAY) -> str:
    """Render the status table as a plain string (for chat_interface.send)."""
    table = build_status_table(project_dir, max_trials)
    console = Console(width=120, no_color=True, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get()
