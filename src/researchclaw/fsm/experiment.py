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
from researchclaw.fsm._shared import (
    SYSTEM_PROMPT_NO_TERMINAL,
    SYSTEM_PROMPT_PROACTIVE,
    gather_project_context,
    get_provider_safe,
    noop_context,
    parse_llm_response,
    prompt_llm_unavailable,
    try_ensure_venv,
)
from researchclaw.models import TrialMeta
from researchclaw.permissions import build_read_paths_section

# Backward-compatibility aliases — tests import these private names via
# ``from researchclaw.fsm.experiment import _parse_llm_response`` or
# ``monkeypatch.setattr(experiment_mod, "_get_provider_safe", ...)``.
_noop_context = noop_context
_get_provider_safe = get_provider_safe
_prompt_llm_unavailable = prompt_llm_unavailable
_parse_llm_response = parse_llm_response
_try_ensure_venv = try_ensure_venv
_gather_project_context = gather_project_context


# --- System prompt for coding agent ---

CODING_AGENT_SYSTEM = (
    "You are the Coding Agent for ResearchClaw, a research experiment orchestrator.\n"
    "\n"
    "Your task is to implement the experiment described in the plan below.\n"
    "\n"
    + SYSTEM_PROMPT_NO_TERMINAL + "\n"
    "\n"
    + SYSTEM_PROMPT_PROACTIVE + "\n"
    "\n"
    "{project_context}\n"
    "\n"
    "## Plan\n"
    "{plan_content}\n"
    "\n"
    "## Instructions\n"
    "- Generate Python experiment code as a single file: main.py\n"
    "- The code will be placed in the trial's experiment/codes_exp/ directory\n"
    "- It will be run via: {venv_python} main.py (inside the codes_exp directory)\n"
    "- Output results to stdout — they will be captured automatically\n"
    "- If the experiment needs additional Python packages, list them (one per line, pip format)\n"
    "\n"
    "## Output Format\n"
    "Respond with TWO clearly separated sections:\n"
    "\n"
    "### REQUIREMENTS\n"
    "List any pip packages needed (one per line), or write NONE if no extra packages needed.\n"
    "\n"
    "### CODE\n"
    "The complete main.py content.\n"
)


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
        chat_interface.send_status(f"[EXPERIMENT_IMPLEMENT] Generating experiment code for {trial_dir.name}...")

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

        project_dir = trial_dir.parent.parent.parent
        project_context = _gather_project_context(project_dir)
        system = CODING_AGENT_SYSTEM.format(
            plan_content=plan_content,
            venv_python="env/bin/python",
            project_context=project_context,
        )
        system += build_read_paths_section(config, meta)
        try:
            thinking_ctx = chat_interface.show_thinking() if chat_interface is not None else _noop_context()
            with thinking_ctx:
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
        chat_interface.send_status(
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
