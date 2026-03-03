from __future__ import annotations

import stat
import subprocess
import threading
from pathlib import Path
from typing import Any

import jinja2

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm import TrialAborted
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta


# --- System prompt for coding agent ---

CODING_AGENT_SYSTEM = """\
You are the Coding Agent for ResearchClaw, a research experiment orchestrator.

Your task is to implement the experiment described in the plan below.

## Plan
{plan_content}

## Instructions
- Generate Python experiment code as a single file: main.py
- The code will be placed in the trial's experiment/codes_exp/ directory
- It will be run via: {venv_python} main.py (inside the codes_exp directory)
- Output results to stdout — they will be captured automatically
- If the experiment needs additional Python packages, list them (one per line, pip format)

## Output Format
Respond with TWO clearly separated sections:

### REQUIREMENTS
List any pip packages needed (one per line), or write NONE if no extra packages needed.

### CODE
The complete main.py content.
"""


def _get_provider_safe(config: ResearchClawConfig) -> Any | None:
    """Try to get an LLM provider, return None if unavailable."""
    try:
        from researchclaw.llm.provider import get_provider
        return get_provider(config)
    except Exception:
        return None


def _prompt_llm_unavailable(chat_interface: Any, error_msg: str) -> str:
    """Prompt user with retry/skip/quit options when LLM is unavailable.

    Returns 'retry', 'skip', or 'quit'. If no chat_interface, returns 'skip'.
    Raises TrialAborted if user sends /abort.
    """
    if chat_interface is None:
        return "skip"

    chat_interface.send(
        f"{error_msg}\n"
        "Options: (r)etry / (s)kip (use placeholder) / (q)uit"
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
    if text in ("r", "retry"):
        return "retry"
    if text in ("q", "quit"):
        return "quit"
    return "skip"


def _load_template() -> jinja2.Template:
    """Load the run_exp.sh.jinja2 template."""
    templates_dir = Path(__file__).parent.parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )
    return env.get_template("run_exp.sh.jinja2")


def _render_run_exp_sh(trial_dir: Path, config: ResearchClawConfig) -> str:
    """Render run_exp.sh from the Jinja2 template."""
    template = _load_template()
    return template.render(
        trial_dir=str(trial_dir),
        python_command=config.python_command,
    )


def _parse_llm_response(response: str) -> tuple[str, str]:
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


def _write_experiment_files(
    trial_dir: Path,
    code: str,
    requirements: str,
    config: ResearchClawConfig,
) -> None:
    """Write experiment code, requirements, and run_exp.sh to the trial directory.

    Only writes to allowed paths:
    - experiment/codes_exp/main.py
    - experiment/run_exp.sh
    - requirements.txt (trial root)
    """
    # Write main.py
    codes_dir = trial_dir / "experiment" / "codes_exp"
    codes_dir.mkdir(parents=True, exist_ok=True)
    (codes_dir / "main.py").write_text(code)

    # Write requirements.txt (overwrite)
    (trial_dir / "requirements.txt").write_text(requirements)

    # Render and write run_exp.sh
    script_content = _render_run_exp_sh(trial_dir, config)
    script_path = trial_dir / "experiment" / "run_exp.sh"
    script_path.write_text(script_content)
    # Make executable
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def handle_experiment_implement(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the EXPERIMENT_IMPLEMENT state.

    Uses the coding agent to read PLAN.md and generate experiment code
    under experiment/codes_exp/, run_exp.sh, and requirements.txt.

    Returns:
        State.EXPERIMENT_EXECUTE on success.
    """
    if chat_interface is not None:
        chat_interface.send(f"[EXPERIMENT_IMPLEMENT] Generating experiment code for {trial_dir.name}...")

    # Read PLAN.md
    plan_path = trial_dir / "PLAN.md"
    if plan_path.exists():
        plan_content = plan_path.read_text()
    else:
        plan_content = "No plan available."

    # Get LLM provider and generate code with retry/skip/quit on failure
    requirements = ""
    code = ""
    while True:
        provider = _get_provider_safe(config)
        if provider is None:
            action = _prompt_llm_unavailable(chat_interface, "No LLM provider available.")
            if action == "retry":
                continue
            elif action == "quit":
                raise SystemExit("User quit due to LLM unavailability")
            else:
                # skip — use placeholder
                code = _placeholder_code(plan_content)
                break

        system = CODING_AGENT_SYSTEM.format(
            plan_content=plan_content,
            venv_python="env/bin/python",
        )
        try:
            response = provider.chat(
                messages=[{
                    "role": "user",
                    "content": (
                        "Generate the experiment code based on the plan. "
                        "Follow the output format with REQUIREMENTS and CODE sections."
                    ),
                }],
                system=system,
            )
            requirements, code = _parse_llm_response(response)
            break
        except Exception as e:
            action = _prompt_llm_unavailable(chat_interface, f"LLM error: {e}")
            if action == "retry":
                continue
            elif action == "quit":
                raise SystemExit("User quit due to LLM error")
            else:
                code = _placeholder_code(plan_content)
                break

    # Write files
    _write_experiment_files(trial_dir, code, requirements, config)

    if chat_interface is not None:
        chat_interface.send(
            f"Experiment code written to {trial_dir.name}/experiment/codes_exp/main.py\n"
            f"Run script: {trial_dir.name}/experiment/run_exp.sh"
        )

    return State.EXPERIMENT_EXECUTE


def _placeholder_code(plan_content: str) -> str:
    """Generate placeholder experiment code when no LLM is available."""
    return (
        '"""Placeholder experiment — no LLM was available to generate code."""\n'
        "\n"
        "\n"
        "def main() -> None:\n"
        '    print("Experiment placeholder")\n'
        f'    print("Plan: {plan_content[:200]}")\n'
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def _run_experiment_subprocess(
    trial_dir: Path,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> tuple[int, str]:
    """Run experiment/run_exp.sh as a subprocess with live streaming.

    Uses a reader thread so that output streaming doesn't block timeout detection.

    Returns:
        Tuple of (exit_code, captured_output).
    """
    script_path = trial_dir / "experiment" / "run_exp.sh"
    if not script_path.exists():
        return 1, "run_exp.sh not found"

    output_lines: list[str] = []

    def _reader(proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            if chat_interface is not None:
                chat_interface.send(line.rstrip("\n"))

    try:
        proc = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(trial_dir),
        )
        reader_thread = threading.Thread(target=_reader, args=(proc,))
        reader_thread.start()
        proc.wait(timeout=config.experiment_timeout_seconds)
        reader_thread.join(timeout=5)
        return proc.returncode or 0, "".join(output_lines)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader_thread.join(timeout=5)
        output_lines.append("\n[TIMEOUT] Experiment exceeded timeout.\n")
        return 1, "".join(output_lines)
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        reader_thread.join(timeout=5)
        output_lines.append("\n[INTERRUPTED] Experiment interrupted by user.\n")
        raise
    except Exception as e:
        output_lines.append(f"\n[ERROR] {e}\n")
        return 1, "".join(output_lines)


def _save_output_log(trial_dir: Path, retry_count: int, output: str) -> Path:
    """Save captured output to experiment/outputs/log_iter{N:03}.

    Returns:
        Path to the saved log file.
    """
    outputs_dir = trial_dir / "experiment" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = outputs_dir / f"log_iter{retry_count:03d}"
    log_path.write_text(output)
    return log_path


def _try_ensure_venv(
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
                f"Venv creation failed: {e}\n"
                "Options: (r)etry / (s)kip (run anyway) / (q)uit"
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
            if text in ("r", "retry"):
                continue
            if text in ("q", "quit"):
                raise SystemExit("User quit during venv creation failure")
            return False


def handle_experiment_execute(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the EXPERIMENT_EXECUTE state.

    Runs experiment/run_exp.sh as a subprocess with live output streaming.
    Captures output to experiment/outputs/log_iter{N:03}.

    On success: saves exit code to meta, returns EVAL_IMPLEMENT.
    On failure: returns EXPERIMENT_IMPLEMENT for retry (up to max_retries).
    If max retries exceeded: returns EXPERIMENT_REPORT.
    """
    retry = meta.experiment_retry_count
    if chat_interface is not None:
        chat_interface.send(
            f"[EXPERIMENT_EXECUTE] Running experiment for {trial_dir.name} "
            f"(attempt {retry + 1}/{config.max_retries})..."
        )

    # Pre-flight venv check
    _try_ensure_venv(trial_dir, config, chat_interface)

    exit_code, output = _run_experiment_subprocess(
        trial_dir, config, chat_interface
    )

    _save_output_log(trial_dir, retry, output)

    meta.experiment_exit_code = exit_code

    if exit_code == 0:
        if chat_interface is not None:
            chat_interface.send(
                f"Experiment completed successfully (exit code 0)."
            )
        meta.status = "running"
        return State.EVAL_IMPLEMENT

    # Failure path
    meta.experiment_retry_count = retry + 1

    if meta.experiment_retry_count >= config.max_retries:
        if chat_interface is not None:
            chat_interface.send(
                f"Experiment failed (exit code {exit_code}). "
                f"Max retries ({config.max_retries}) exceeded. Skipping to report."
            )
        return State.EXPERIMENT_REPORT

    if chat_interface is not None:
        chat_interface.send(
            f"Experiment failed (exit code {exit_code}). "
            f"Retrying ({meta.experiment_retry_count}/{config.max_retries})..."
        )
    return State.EXPERIMENT_IMPLEMENT
