"""Shared FSM utilities and constants.

This module centralises functions and string constants that were previously
duplicated across multiple FSM handler modules (experiment.py, evaluate.py,
plan.py, report.py, decide.py).

Created as part of US-R01.  All functions are exact copies of the originals
with the leading underscore removed from their names.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm import TrialAborted
from researchclaw.sandbox import SandboxManager


# ---------------------------------------------------------------------------
# System-prompt boilerplate constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_NO_TERMINAL = (
    "IMPORTANT: You do NOT have access to a terminal, file system, or any tools. You can only "
    "communicate via text. If the user needs to run a command, provide the exact command for them "
    "to copy and run themselves. Never say 'I ran this command' or 'I posted this to the terminal'."
)

SYSTEM_PROMPT_PROACTIVE = (
    "Be proactive: analyze the project structure, identify relevant files, and reference specific "
    "code when proposing experiments. Do not wait for the user to point you to relevant files.\n"
    "\n"
    "When you need user input or encounter ambiguity, present 2-3 concrete options with clear "
    "consequences. Never ask open-ended questions without proposing specific action paths."
)


# ---------------------------------------------------------------------------
# gather_project_context  (was plan.py  _gather_project_context + helpers)
# ---------------------------------------------------------------------------

EXCLUDED_DIRS = {"sandbox_researchclaw", "__pycache__", ".git", "node_modules", ".venv"}
KEY_FILES = ("README.md", "pyproject.toml", "setup.py")
MAX_CONTEXT_CHARS = 8000
MAX_KEY_FILE_CHARS = 2000
MAX_TREE_DEPTH = 3


def gather_project_context(project_dir: Path) -> str:
    """Walk project_dir and build a project context string for LLM prompts.

    Generates a file tree (max depth 3), reads key files (README.md,
    pyproject.toml, setup.py), and truncates to ~8000 chars.

    Args:
        project_dir: Root directory of the user's project.

    Returns:
        Project context string, or empty string if project_dir doesn't exist.
    """
    if not project_dir.is_dir():
        return ""

    parts: list[str] = ["## Project Structure\n"]

    # Build file tree (max depth 3)
    tree_lines: list[str] = []
    _walk_tree(current=project_dir, lines=tree_lines, depth=0, max_depth=MAX_TREE_DEPTH)
    parts.append("```\n" + "\n".join(tree_lines) + "\n```\n")

    # Read key files
    for fname in KEY_FILES:
        fpath = project_dir / fname
        if fpath.is_file():
            try:
                content = fpath.read_text(errors="replace")
                if len(content) > MAX_KEY_FILE_CHARS:
                    content = content[:MAX_KEY_FILE_CHARS] + "\n[... truncated]"
                parts.append(f"### {fname}\n```\n{content}\n```\n")
            except Exception:
                pass

    result = "\n".join(parts)

    # Truncate to ~8000 chars
    if len(result) > MAX_CONTEXT_CHARS:
        result = result[:MAX_CONTEXT_CHARS] + "\n[... truncated]"

    return result


def _walk_tree(
    current: Path,
    lines: list[str],
    depth: int,
    max_depth: int,
) -> None:
    """Recursively build a file tree list, excluding certain directories."""
    if depth > max_depth:
        return

    try:
        entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except PermissionError:
        return

    for entry in entries:
        if entry.is_dir() and entry.name in EXCLUDED_DIRS:
            continue

        indent = "  " * depth
        if entry.is_dir():
            lines.append(f"{indent}{entry.name}/")
            _walk_tree(entry, lines, depth + 1, max_depth)
        else:
            lines.append(f"{indent}{entry.name}")


# ---------------------------------------------------------------------------
# noop_context  (was experiment.py / evaluate.py / report.py  _noop_context)
# ---------------------------------------------------------------------------

@contextmanager
def noop_context() -> Iterator[None]:
    """No-op context manager for when chat_interface is None."""
    yield


# ---------------------------------------------------------------------------
# get_provider_safe  (was experiment.py / evaluate.py / plan.py / report.py
#                     _get_provider_safe)
# ---------------------------------------------------------------------------

def get_provider_safe(config: ResearchClawConfig) -> Any | None:
    """Try to get an LLM provider, return None if unavailable."""
    try:
        from researchclaw.llm.provider import get_provider
        return get_provider(config)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# prompt_llm_unavailable  (was experiment.py / evaluate.py
#                          _prompt_llm_unavailable)
# ---------------------------------------------------------------------------

def prompt_llm_unavailable(chat_interface: Any, error_msg: str) -> str:
    """Prompt user with structured options when LLM is unavailable.

    Shows a Rich Panel with numbered options, recommendation, and consequences.
    Returns 'retry', 'skip', or 'quit'. If no chat_interface, returns 'skip'.
    Raises TrialAborted if user sends /abort.
    """
    if chat_interface is None:
        return "skip"

    chat_interface.send(
        f"**{error_msg}**\n\n"
        "Choose an action:\n\n"
        "  **(1)** Retry          — attempt to connect to the LLM again (Recommended)\n"
        "  **(2)** Skip           — use placeholder code and continue the pipeline\n"
        "  **(3)** Quit           — exit ResearchClaw\n\n"
        "*Retry is recommended if this is a transient network issue. "
        "Skip will generate a placeholder that you can replace manually.*"
    )
    try:
        user_input = chat_interface.receive()
    except (SystemExit, KeyboardInterrupt):
        return "quit"

    from researchclaw.repl import SlashCommand, UserMessage
    if isinstance(user_input, SlashCommand):
        if user_input.name == "/quit":
            return "quit"
        if user_input.name == "/abort":
            raise TrialAborted("User aborted trial during LLM unavailability")
        return "skip"

    text = user_input.text.strip().lower() if isinstance(user_input, UserMessage) else ""
    if text in ("1", "r", "retry"):
        return "retry"
    if text in ("3", "q", "quit"):
        return "quit"
    return "skip"


# ---------------------------------------------------------------------------
# try_ensure_venv  (was experiment.py / evaluate.py  _try_ensure_venv)
# ---------------------------------------------------------------------------

def try_ensure_venv(
    trial_dir: Path,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> bool:
    """Try to create the trial venv, prompting user on failure.

    Returns True if venv is ready, False if user chose to skip.
    Raises SystemExit if user chose to quit.
    Raises TrialAborted if user sends /abort.
    """
    from researchclaw.sandbox.venv_manager import VenvManager

    while True:
        try:
            VenvManager.ensure_venv(trial_dir, config.python_command)
            return True
        except Exception as e:
            if chat_interface is None:
                return False
            chat_interface.send(
                f"**Venv creation failed:** {e}\n\n"
                "Choose an action:\n\n"
                f"  **(1)** Retry          — try creating the venv again (Recommended)\n"
                f"  **(2)** Skip           — run the experiment without a dedicated venv\n"
                f"  **(3)** Quit           — exit ResearchClaw\n\n"
                f"*Retry is recommended. Skipping means the experiment will run in the "
                f"system Python environment, which may cause dependency conflicts.*"
            )
            try:
                user_input = chat_interface.receive()
            except (SystemExit, KeyboardInterrupt):
                raise SystemExit("User quit during venv creation failure")

            from researchclaw.repl import SlashCommand, UserMessage
            if isinstance(user_input, SlashCommand):
                if user_input.name == "/quit":
                    raise SystemExit("User quit during venv creation failure")
                if user_input.name == "/abort":
                    raise TrialAborted("User aborted trial during venv creation failure")
                return False

            text = user_input.text.strip().lower() if isinstance(user_input, UserMessage) else ""
            if text in ("1", "r", "retry"):
                continue
            if text in ("3", "q", "quit"):
                raise SystemExit("User quit during venv creation failure")
            return False


# ---------------------------------------------------------------------------
# parse_llm_response  (was experiment.py / evaluate.py  _parse_llm_response)
# ---------------------------------------------------------------------------

def parse_llm_response(response: str) -> tuple[str, str]:
    """Parse the LLM response into (requirements, code) sections.

    Returns:
        Tuple of (requirements_text, code_text). Requirements may be empty
        or "NONE". Code is the main.py content.
    """
    requirements = ""
    code = ""

    # Try to find ### REQUIREMENTS and ### CODE sections
    response_upper = response.upper()

    # Find requirements section
    req_markers = ["### REQUIREMENTS", "## REQUIREMENTS", "REQUIREMENTS:"]
    code_markers = ["### CODE", "## CODE", "CODE:", "```python", "```"]

    req_start = -1
    for marker in req_markers:
        idx = response_upper.find(marker.upper())
        if idx != -1:
            req_start = idx + len(marker)
            break

    code_start = -1
    for marker in code_markers:
        idx = response_upper.find(marker.upper())
        if idx != -1:
            code_start = idx + len(marker)
            break

    if req_start != -1 and code_start != -1 and req_start < code_start:
        # Both sections found in order
        requirements = response[req_start:code_start].strip()
        # Clean up code section markers from requirements
        for marker in code_markers:
            upper_marker = marker.upper()
            if upper_marker in requirements.upper():
                end_idx = requirements.upper().find(upper_marker)
                requirements = requirements[:end_idx].strip()
                break
        code = response[code_start:].strip()
    elif code_start != -1:
        # Only code section found
        code = response[code_start:].strip()
    else:
        # No clear sections — treat entire response as code
        code = response.strip()

    # Clean up code: remove trailing ``` if present
    if code.endswith("```"):
        code = code[:-3].strip()
    # Remove leading ``` if present (from ```python marker)
    if code.startswith("```"):
        # Find end of first line
        newline = code.find("\n")
        if newline != -1:
            code = code[newline + 1:].strip()

    # Clean up requirements
    if requirements.upper().strip() == "NONE":
        requirements = ""
    # Strip lines and filter empty
    if requirements:
        lines = [line.strip() for line in requirements.splitlines() if line.strip()]
        # Remove lines that look like section headers
        lines = [l for l in lines if not l.startswith("#") and not l.startswith("```")]
        requirements = "\n".join(lines)

    return requirements, code


# ---------------------------------------------------------------------------
# persist_autopilot  (was plan.py / decide.py  _persist_autopilot)
# ---------------------------------------------------------------------------

def persist_autopilot(config: ResearchClawConfig, project_dir: Path) -> None:
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
