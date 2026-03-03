from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.sandbox import SandboxManager


# --- System prompts for summary agent ---

SUMMARY_AGENT_SYSTEM = """\
You are the Summary Agent for ResearchClaw, a research experiment orchestrator.

Your task is to analyze a completed experiment trial and generate a comprehensive REPORT.md.

## Trial Information
- Trial directory: {trial_name}
- Trial number: {trial_number}

## Experiment Plan
{plan_content}

## Experiment Code
{experiment_code}

## Experiment Outputs
{experiment_outputs}

## Evaluation Code
{eval_code}

## Evaluation Outputs
{eval_outputs}

## Visualizations
{visualizations}

## Prior Trial Reports
{prior_reports}

## Instructions
Generate a comprehensive REPORT.md that includes:
1. **What was done**: Describe the experiment methodology
2. **Results summary**: Key findings and metrics
3. **Comparison with prior trials**: How this trial compares (if prior trials exist)
4. **Future directions**: Suggestions for next experiments

Use markdown formatting. Be concise but thorough.
"""

SUMMARY_LOG_SYSTEM = """\
You are the Summary Agent for ResearchClaw. Summarize the following REPORT.md into \
exactly 2-3 lines for the experiment log. Be concise and capture the key findings.
"""


def _get_provider_safe(config: ResearchClawConfig) -> Any | None:
    """Try to get an LLM provider, return None if unavailable."""
    try:
        from researchclaw.llm.provider import get_provider
        return get_provider(config)
    except Exception:
        return None


def _gather_trial_content(trial_dir: Path) -> dict[str, str]:
    """Gather all relevant content from a trial directory for the summary agent.

    Returns a dict with keys: plan, experiment_code, experiment_outputs,
    eval_code, eval_outputs, visualizations.
    """
    content: dict[str, str] = {}

    # PLAN.md
    plan_path = trial_dir / "PLAN.md"
    content["plan"] = plan_path.read_text() if plan_path.exists() else "No plan available."

    # Experiment code
    codes_exp = trial_dir / "experiment" / "codes_exp"
    if codes_exp.is_dir():
        parts: list[str] = []
        for f in sorted(codes_exp.iterdir()):
            if f.is_file():
                parts.append(f"### {f.name}\n```\n{f.read_text()}\n```")
        content["experiment_code"] = "\n\n".join(parts) if parts else "No experiment code."
    else:
        content["experiment_code"] = "No experiment code."

    # Experiment outputs (logs)
    outputs_dir = trial_dir / "experiment" / "outputs"
    if outputs_dir.is_dir():
        parts = []
        for f in sorted(outputs_dir.iterdir()):
            if f.is_file() and f.name.startswith("log_iter"):
                text = f.read_text()
                # Truncate very long output logs
                if len(text) > 5000:
                    text = text[:5000] + "\n[... truncated]"
                parts.append(f"### {f.name}\n```\n{text}\n```")
        content["experiment_outputs"] = "\n\n".join(parts) if parts else "No experiment outputs."
    else:
        content["experiment_outputs"] = "No experiment outputs."

    # Evaluation code
    codes_eval = trial_dir / "experiment" / "codes_eval"
    if codes_eval.is_dir():
        parts = []
        for f in sorted(codes_eval.iterdir()):
            if f.is_file():
                parts.append(f"### {f.name}\n```\n{f.read_text()}\n```")
        content["eval_code"] = "\n\n".join(parts) if parts else "No evaluation code."
    else:
        content["eval_code"] = "No evaluation code."

    # Evaluation outputs (logs)
    if outputs_dir.is_dir():
        parts = []
        for f in sorted(outputs_dir.iterdir()):
            if f.is_file() and f.name.startswith("eval_log_iter"):
                text = f.read_text()
                if len(text) > 5000:
                    text = text[:5000] + "\n[... truncated]"
                parts.append(f"### {f.name}\n```\n{text}\n```")
        content["eval_outputs"] = "\n\n".join(parts) if parts else "No evaluation outputs."
    else:
        content["eval_outputs"] = "No evaluation outputs."

    # Visualizations (list files at trial root that are images/videos)
    viz_extensions = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".avi", ".svg", ".pdf"}
    viz_files = [
        f.name for f in trial_dir.iterdir()
        if f.is_file() and f.suffix.lower() in viz_extensions
    ]
    content["visualizations"] = ", ".join(sorted(viz_files)) if viz_files else "No visualizations."

    return content


def _gather_prior_reports(project_dir: Path, current_trial_name: str) -> str:
    """Gather REPORT.md content from prior trials (not the current one)."""
    sandbox = SandboxManager.sandbox_path(project_dir)
    experiments_dir = sandbox / "experiments"

    if not experiments_dir.is_dir():
        return "No prior trials."

    trial_dirs = sorted(
        d for d in experiments_dir.iterdir()
        if d.is_dir() and "_trial_" in d.name and d.name != current_trial_name
    )

    if not trial_dirs:
        return "No prior trials."

    parts: list[str] = []
    for td in trial_dirs:
        report_path = td / "REPORT.md"
        if report_path.exists():
            content = report_path.read_text().strip()
            if content:
                # Truncate long reports
                if len(content) > 2000:
                    content = content[:2000] + "\n[... truncated]"
                parts.append(f"### {td.name}\n{content}")

    return "\n\n".join(parts) if parts else "No prior trial reports."


def _generate_report_llm(
    trial_dir: Path,
    meta: TrialMeta,
    provider: Any,
    trial_content: dict[str, str],
    prior_reports: str,
) -> str:
    """Generate REPORT.md content using the summary agent LLM."""
    system = SUMMARY_AGENT_SYSTEM.format(
        trial_name=trial_dir.name,
        trial_number=meta.trial_number,
        plan_content=trial_content["plan"],
        experiment_code=trial_content["experiment_code"],
        experiment_outputs=trial_content["experiment_outputs"],
        eval_code=trial_content["eval_code"],
        eval_outputs=trial_content["eval_outputs"],
        visualizations=trial_content["visualizations"],
        prior_reports=prior_reports,
    )

    report = provider.chat(
        messages=[{
            "role": "user",
            "content": "Generate the REPORT.md for this trial.",
        }],
        system=system,
    )
    return report


def _generate_report_fallback(
    trial_dir: Path,
    meta: TrialMeta,
    trial_content: dict[str, str],
) -> str:
    """Generate a fallback REPORT.md when no LLM is available."""
    parts = [
        f"# Trial Report: {trial_dir.name}",
        "",
        f"**Trial Number**: {meta.trial_number}",
        f"**Status**: {meta.status}",
        f"**Experiment Exit Code**: {meta.experiment_exit_code}",
        f"**Eval Exit Code**: {meta.eval_exit_code}",
        "",
        "## Plan",
        trial_content["plan"],
        "",
        "## Experiment Outputs",
        trial_content["experiment_outputs"],
        "",
        "## Evaluation Outputs",
        trial_content["eval_outputs"],
        "",
        "## Visualizations",
        trial_content["visualizations"],
        "",
        "## Future Directions",
        "*Report auto-generated without LLM. Review outputs manually.*",
    ]
    return "\n".join(parts)


def _generate_log_summary_llm(provider: Any, report_content: str) -> str:
    """Generate a 2-3 line summary from REPORT.md using LLM."""
    summary = provider.chat(
        messages=[{
            "role": "user",
            "content": report_content,
        }],
        system=SUMMARY_LOG_SYSTEM,
    )
    # Ensure it's concise — take first 3 non-empty lines
    lines = [line.strip() for line in summary.strip().splitlines() if line.strip()]
    return " ".join(lines[:3])


def _generate_log_summary_fallback(
    meta: TrialMeta,
    trial_content: dict[str, str],
) -> str:
    """Generate a fallback 2-3 line summary when no LLM is available."""
    terminated = "terminated" in (meta.status or "").lower()
    prefix = "[TERMINATED] " if terminated else ""
    plan_snippet = trial_content["plan"][:150].replace("\n", " ").strip()
    exit_info = f"Experiment exit={meta.experiment_exit_code}, eval exit={meta.eval_exit_code}."
    return f"{prefix}{plan_snippet}. {exit_info}"


def _write_report(trial_dir: Path, content: str, terminated: bool) -> None:
    """Write REPORT.md to the trial directory.

    If terminated is True, prepends [TERMINATED-DURING-EXPERIMENT] marker.
    """
    report_path = trial_dir / "REPORT.md"
    if terminated:
        content = f"[TERMINATED-DURING-EXPERIMENT]\n\n{content}"
    report_path.write_text(content)


def _append_experiment_log(
    project_dir: Path,
    trial_dir: Path,
    summary: str,
) -> None:
    """Append a trial summary entry to EXPERIMENT_LOGS.md.

    Format: {YYYYMMDD} - trial_{N:03}: {summary}. Full Doc: [REPORT.md](experiments/{trial_name}/REPORT.md)
    """
    sandbox = SandboxManager.sandbox_path(project_dir)
    logs_path = sandbox / "EXPERIMENT_LOGS.md"

    trial_name = trial_dir.name
    # Parse date and trial number from trial name: {YYYYMMDD}_trial_{N:03}
    parts = trial_name.split("_trial_")
    date_str = parts[0] if parts else trial_name[:8]
    trial_num_str = parts[1] if len(parts) > 1 else "001"

    entry = (
        f"{date_str} - trial_{trial_num_str}: {summary}. "
        f"Full Doc: [REPORT.md](experiments/{trial_name}/REPORT.md)\n"
    )

    with open(logs_path, "a") as f:
        f.write(entry)


def handle_experiment_report(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the EXPERIMENT_REPORT state.

    The summary agent analyzes trial codes, outputs, visualizations, and
    prior trial REPORT.md files to generate REPORT.md. Then summarizes it
    to 2-3 lines and appends to EXPERIMENT_LOGS.md.

    REPL is disabled during this state — user cannot interrupt. The handler
    communicates via send() only.

    If trial was terminated (status contains 'terminated'), REPORT.md starts
    with [TERMINATED-DURING-EXPERIMENT].

    Returns:
        State.DECIDE always.
    """
    if chat_interface is not None:
        chat_interface.send(
            f"[EXPERIMENT_REPORT] Generating report for {trial_dir.name}... "
            f"(do not interrupt)"
        )

    # Determine if trial was terminated
    terminated = "terminated" in (meta.status or "").lower()

    # Gather trial content
    trial_content = _gather_trial_content(trial_dir)

    # Gather prior reports
    project_dir = trial_dir.parent.parent.parent
    prior_reports = _gather_prior_reports(project_dir, trial_dir.name)

    # Get LLM provider
    provider = _get_provider_safe(config)

    # Generate REPORT.md
    if provider is not None:
        try:
            report_content = _generate_report_llm(
                trial_dir, meta, provider, trial_content, prior_reports
            )
        except Exception as e:
            if chat_interface is not None:
                chat_interface.send(f"LLM error: {e}. Generating fallback report.")
            report_content = _generate_report_fallback(
                trial_dir, meta, trial_content
            )
    else:
        report_content = _generate_report_fallback(
            trial_dir, meta, trial_content
        )

    # Write REPORT.md
    _write_report(trial_dir, report_content, terminated)

    if chat_interface is not None:
        chat_interface.send(f"REPORT.md written to {trial_dir.name}/REPORT.md")

    # Generate log summary and append to EXPERIMENT_LOGS.md
    if provider is not None:
        try:
            log_summary = _generate_log_summary_llm(provider, report_content)
        except Exception:
            log_summary = _generate_log_summary_fallback(meta, trial_content)
    else:
        log_summary = _generate_log_summary_fallback(meta, trial_content)

    _append_experiment_log(project_dir, trial_dir, log_summary)

    if chat_interface is not None:
        chat_interface.send("EXPERIMENT_LOGS.md updated.")

    # Mark trial as completed (preserve terminated status)
    if not terminated:
        meta.status = "completed"

    return State.DECIDE
