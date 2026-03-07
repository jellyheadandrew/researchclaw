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
# ``from researchclaw.fsm.evaluate import _parse_llm_response`` or
# ``monkeypatch.setattr(evaluate_mod, "_get_provider_safe", ...)``.
_noop_context = noop_context
_get_provider_safe = get_provider_safe
_prompt_llm_unavailable = prompt_llm_unavailable
_parse_llm_response = parse_llm_response
_try_ensure_venv = try_ensure_venv
_gather_project_context = gather_project_context


# --- System prompt for eval coding agent ---

EVAL_CODING_AGENT_SYSTEM = (
    "You are the Evaluation Coding Agent for ResearchClaw, a research experiment orchestrator.\n"
    "\n"
    "Your task is to write evaluation code that analyzes the experiment outputs.\n"
    "\n"
    + SYSTEM_PROMPT_NO_TERMINAL + "\n"
    "\n"
    + SYSTEM_PROMPT_PROACTIVE + "\n"
    "\n"
    "{project_context}\n"
    "\n"
    "## Experiment Plan\n"
    "{plan_content}\n"
    "\n"
    "## Experiment Outputs\n"
    "The experiment has been run and its outputs are available at: {outputs_dir}\n"
    "The trial root directory is: {trial_dir}\n"
    "\n"
    "## Instructions\n"
    "- Generate Python evaluation code as a single file: main.py\n"
    "- The code will be placed in the trial's experiment/codes_eval/ directory\n"
    "- It will be run via: {venv_python} main.py (inside the codes_eval directory)\n"
    "- Environment variables available: OUTPUTS_DIR (path to experiment outputs), TRIAL_DIR (path to trial root)\n"
    "- Read experiment outputs from OUTPUTS_DIR\n"
    "- Write any visualizations (.png, .mp4, etc.) to TRIAL_DIR (the trial root directory)\n"
    "- Print evaluation results to stdout — they will be captured automatically\n"
    "- If the evaluation needs additional Python packages, list them\n"
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


def _load_eval_template() -> jinja2.Template:
    """Load the eval.sh.jinja2 template."""
    templates_dir = Path(__file__).parent.parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )
    return env.get_template("eval.sh.jinja2")


def _render_run_eval_sh(trial_dir: Path, config: ResearchClawConfig) -> str:
    """Render run_eval.sh from the Jinja2 template."""
    template = _load_eval_template()
    return template.render(
        trial_dir=str(trial_dir),
        python_command=config.python_command,
    )


def _write_eval_files(
    trial_dir: Path,
    code: str,
    requirements: str,
    config: ResearchClawConfig,
) -> None:
    """Write evaluation code and run_eval.sh to the trial directory.

    Only writes to allowed paths:
    - experiment/codes_eval/main.py
    - experiment/run_eval.sh
    """
    # Write main.py
    codes_dir = trial_dir / "experiment" / "codes_eval"
    codes_dir.mkdir(parents=True, exist_ok=True)
    (codes_dir / "main.py").write_text(code)

    # Update requirements.txt if eval needs additional packages
    if requirements:
        existing = (trial_dir / "requirements.txt").read_text() if (trial_dir / "requirements.txt").exists() else ""
        existing_set = {line.strip() for line in existing.splitlines() if line.strip()}
        new_reqs = [line.strip() for line in requirements.splitlines() if line.strip()]
        for req in new_reqs:
            if req not in existing_set:
                existing_set.add(req)
        (trial_dir / "requirements.txt").write_text("\n".join(sorted(existing_set)) + "\n" if existing_set else "")

    # Render and write run_eval.sh
    script_content = _render_run_eval_sh(trial_dir, config)
    script_path = trial_dir / "experiment" / "run_eval.sh"
    script_path.write_text(script_content)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _placeholder_eval_code(plan_content: str) -> str:
    """Generate placeholder evaluation code when no LLM is available."""
    return (
        '"""Placeholder evaluation — no LLM was available to generate code."""\n'
        "import os\n"
        "\n"
        "\n"
        "def main() -> None:\n"
        '    outputs_dir = os.environ.get("OUTPUTS_DIR", ".")\n'
        '    print("Evaluation placeholder")\n'
        f'    print("Plan: {plan_content[:200]}")\n'
        '    print(f"Outputs dir: {{outputs_dir}}")\n'
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def handle_eval_implement(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the EVAL_IMPLEMENT state.

    Uses the coding agent to generate evaluation code under
    experiment/codes_eval/ and experiment/run_eval.sh.

    Returns:
        State.EVAL_EXECUTE on success.
    """
    if chat_interface is not None:
        chat_interface.send_status(f"[EVAL_IMPLEMENT] Generating evaluation code for {trial_dir.name}...")

    # Read PLAN.md
    plan_path = trial_dir / "PLAN.md"
    if plan_path.exists():
        plan_content = plan_path.read_text()
    else:
        plan_content = "No plan available."

    outputs_dir = trial_dir / "experiment" / "outputs"

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
                code = _placeholder_eval_code(plan_content)
                break

        project_dir = trial_dir.parent.parent.parent
        project_context = _gather_project_context(project_dir)
        system = EVAL_CODING_AGENT_SYSTEM.format(
            plan_content=plan_content,
            outputs_dir=str(outputs_dir),
            trial_dir=str(trial_dir),
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
                            "Generate the evaluation code based on the plan and experiment outputs. "
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
                code = _placeholder_eval_code(plan_content)
                break

    # Write files
    _write_eval_files(trial_dir, code, requirements, config)

    if chat_interface is not None:
        chat_interface.send(
            f"Evaluation code written to {trial_dir.name}/experiment/codes_eval/main.py\n"
            f"Run script: {trial_dir.name}/experiment/run_eval.sh"
        )

    return State.EVAL_EXECUTE


def _run_eval_subprocess(
    trial_dir: Path,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> tuple[int, str]:
    """Run experiment/run_eval.sh as a subprocess with live streaming.

    Uses a reader thread so that output streaming doesn't block timeout detection.

    Returns:
        Tuple of (exit_code, captured_output).
    """
    script_path = trial_dir / "experiment" / "run_eval.sh"
    if not script_path.exists():
        return 1, "run_eval.sh not found"

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
        output_lines.append("\n[TIMEOUT] Evaluation exceeded timeout.\n")
        return 1, "".join(output_lines)
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        reader_thread.join(timeout=5)
        output_lines.append("\n[INTERRUPTED] Evaluation interrupted by user.\n")
        raise
    except Exception as e:
        output_lines.append(f"\n[ERROR] {e}\n")
        return 1, "".join(output_lines)


def _save_eval_output_log(trial_dir: Path, retry_count: int, output: str) -> Path:
    """Save captured eval output to experiment/outputs/eval_log_iter{N:03}.

    Returns:
        Path to the saved log file.
    """
    outputs_dir = trial_dir / "experiment" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = outputs_dir / f"eval_log_iter{retry_count:03d}"
    log_path.write_text(output)
    return log_path


def handle_eval_execute(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the EVAL_EXECUTE state.

    Runs experiment/run_eval.sh as a subprocess with live output streaming.
    Captures output to experiment/outputs/eval_log_iter{N:03}.

    On success: saves exit code to meta, returns EXPERIMENT_REPORT.
    On failure: returns EVAL_IMPLEMENT for retry (up to max_retries).
    If max retries exceeded: returns EXPERIMENT_REPORT.
    """
    retry = meta.eval_retry_count
    if chat_interface is not None:
        chat_interface.send_status(
            f"[EVAL_EXECUTE] Running evaluation for {trial_dir.name} "
            f"(attempt {retry + 1}/{config.max_retries})..."
        )

    # Pre-flight venv check
    _try_ensure_venv(trial_dir, config, chat_interface)

    exit_code, output = _run_eval_subprocess(
        trial_dir, config, chat_interface
    )

    _save_eval_output_log(trial_dir, retry, output)

    meta.eval_exit_code = exit_code

    if exit_code == 0:
        if chat_interface is not None:
            chat_interface.send(
                "Evaluation completed successfully (exit code 0)."
            )
        return State.EXPERIMENT_REPORT

    # Failure path
    meta.eval_retry_count = retry + 1

    if meta.eval_retry_count >= config.max_retries:
        if chat_interface is not None:
            chat_interface.send(
                f"Evaluation failed (exit code {exit_code}). "
                f"Max retries ({config.max_retries}) exceeded. Skipping to report."
            )
        return State.EXPERIMENT_REPORT

    if chat_interface is not None:
        chat_interface.send(
            f"Evaluation failed (exit code {exit_code}). "
            f"Retrying ({meta.eval_retry_count}/{config.max_retries})..."
        )
    return State.EVAL_IMPLEMENT
