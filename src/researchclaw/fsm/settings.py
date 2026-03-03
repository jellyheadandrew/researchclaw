from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.sandbox import SandboxManager


# --- Config field descriptions ---

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "model": "LLM model to use (e.g. 'claude-opus-4-6')",
    "max_trials": "Maximum number of trials allowed per project",
    "max_retries": "Maximum retry attempts for failed experiments/evals",
    "autopilot": "Whether autopilot mode is enabled (true/false)",
    "experiment_timeout_seconds": "Timeout in seconds for experiment execution",
    "python_command": "Python command to use for venvs (e.g. 'python3')",
    "provider": "LLM provider: claude_cli, claude_agent_sdk, anthropic, openai",
    "api_key": "API key for the LLM provider (Tier 2 only)",
    "display_trials": "Number of recent trials to display in VIEW_SUMMARY",
}

# Fields that are safe to modify via settings
EDITABLE_FIELDS: set[str] = {
    "model",
    "max_trials",
    "max_retries",
    "autopilot",
    "experiment_timeout_seconds",
    "python_command",
    "provider",
    "api_key",
    "display_trials",
}

# Type coercion map
FIELD_TYPES: dict[str, type] = {
    "model": str,
    "max_trials": int,
    "max_retries": int,
    "autopilot": bool,
    "experiment_timeout_seconds": int,
    "python_command": str,
    "provider": str,
    "api_key": str,
    "display_trials": int,
}


def _format_config_display(config: ResearchClawConfig) -> str:
    """Format all config values with descriptions for display."""
    lines = ["**Current Settings:**\n"]
    for f in fields(config):
        value = getattr(config, f.name)
        desc = CONFIG_DESCRIPTIONS.get(f.name, "")
        # Mask api_key for display
        display_value = "***" if f.name == "api_key" and value else repr(value)
        lines.append(f"  **{f.name}** = {display_value}")
        if desc:
            lines.append(f"    _{desc}_")
    lines.append(
        "\nTo change a setting, type e.g. 'set max_retries 10' or "
        "'max_retries = 10'.\nType 'back' to return to DECIDE."
    )
    return "\n".join(lines)


def _parse_setting_change(text: str) -> tuple[str, str] | None:
    """Parse user input to extract a setting name and new value.

    Supports formats:
    - 'set <field> <value>'
    - 'set <field> to <value>'
    - '<field> = <value>'
    - '<field> <value>'

    Returns (field_name, raw_value) or None if not parseable.
    """
    stripped = text.strip()
    if not stripped:
        return None

    # Pattern: 'set <field> to <value>' or 'set <field> <value>'
    m = re.match(
        r"set\s+(\w+)\s+(?:to\s+)?(.+)",
        stripped,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).lower(), m.group(2).strip()

    # Pattern: '<field> = <value>'
    m = re.match(r"(\w+)\s*=\s*(.+)", stripped)
    if m:
        return m.group(1).lower(), m.group(2).strip()

    # Pattern: '<field> <value>' (only if field is a known config field)
    parts = stripped.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in EDITABLE_FIELDS:
        return parts[0].lower(), parts[1].strip()

    return None


def _coerce_value(field_name: str, raw_value: str) -> object | None:
    """Coerce a raw string value to the correct type for the field.

    Returns the coerced value, or None if coercion fails.
    """
    target_type = FIELD_TYPES.get(field_name)
    if target_type is None:
        return None

    if target_type is bool:
        lower = raw_value.lower().strip()
        if lower in ("true", "yes", "on", "1"):
            return True
        if lower in ("false", "no", "off", "0"):
            return False
        return None

    if target_type is int:
        try:
            return int(raw_value)
        except ValueError:
            return None

    # str
    return raw_value


def _apply_setting(
    config: ResearchClawConfig,
    field_name: str,
    value: object,
) -> None:
    """Apply a setting change to the config object."""
    setattr(config, field_name, value)


def _save_project_config(config: ResearchClawConfig, project_dir: Path) -> None:
    """Save config to the project settings file."""
    project_config_path = (
        SandboxManager.sandbox_path(project_dir)
        / "project_settings"
        / "researchclaw.yaml"
    )
    config.save_to_yaml(project_config_path)


def handle_settings(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the SETTINGS state.

    Lists all config values with explanations. User can change values
    via chat. Settings file is saved after each change.

    Args:
        trial_dir: Current trial directory.
        meta: Current trial metadata.
        config: ResearchClaw configuration.
        chat_interface: Chat interface for user interaction.

    Returns:
        State.DECIDE to return to the decision menu.
    """
    project_dir = trial_dir.parent.parent.parent

    if chat_interface is not None:
        chat_interface.send(
            "[SETTINGS] Configuration editor.\n\n"
            + _format_config_display(config)
        )

    while True:
        if chat_interface is None:
            return State.DECIDE

        user_input = chat_interface.receive()

        from researchclaw.repl import SlashCommand, UserMessage

        if isinstance(user_input, SlashCommand):
            if user_input.name == "/quit":
                raise SystemExit("User quit via /quit")
            chat_interface.send(
                f"Command {user_input.name} not available in SETTINGS. "
                "Type 'back' to return to DECIDE."
            )
            continue

        text = user_input.text if isinstance(user_input, UserMessage) else str(user_input)
        text_lower = text.strip().lower()

        if text_lower in ("back", "exit", "quit", "done", "q"):
            return State.DECIDE

        # Try to parse a setting change
        parsed = _parse_setting_change(text)
        if parsed is None:
            chat_interface.send(
                "Could not parse setting change. "
                "Try 'set <field> <value>' or '<field> = <value>'.\n"
                "Type 'back' to return to DECIDE."
            )
            continue

        field_name, raw_value = parsed

        if field_name not in EDITABLE_FIELDS:
            chat_interface.send(
                f"Unknown setting '{field_name}'. "
                f"Available settings: {', '.join(sorted(EDITABLE_FIELDS))}"
            )
            continue

        coerced = _coerce_value(field_name, raw_value)
        if coerced is None:
            expected = FIELD_TYPES.get(field_name, str).__name__
            chat_interface.send(
                f"Invalid value '{raw_value}' for '{field_name}'. "
                f"Expected type: {expected}."
            )
            continue

        old_value = getattr(config, field_name)
        _apply_setting(config, field_name, coerced)
        _save_project_config(config, project_dir)

        chat_interface.send(
            f"Updated **{field_name}**: {repr(old_value)} -> {repr(coerced)}"
        )
