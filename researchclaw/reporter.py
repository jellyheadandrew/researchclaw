from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from .models import TrialRecord


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return "(missing)"
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:]) if data else "(empty)"


def _read_truncated(path: Path, max_chars: int = 3000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + "\n... (truncated)"
        return text
    except Exception:
        return ""


REPORT_ANALYSIS_PROMPT = """\
You are a research experiment analyst for ResearchClaw.
Analyze the following trial data and produce a comprehensive report section.

Your analysis MUST include these sections:
1. **What Worked** — concrete successes observed in outputs/results
2. **What Failed** — errors, failures, or unexpected behavior
3. **Comparison with Prior Trials** — how this trial relates to past attempts (if history provided)
4. **Implementation Summary** — brief description of what the code does
5. **Next-Trial Directions** — specific, actionable suggestions for the next experiment

Be specific and data-driven. Reference actual file names, error messages, and metrics.
Keep each section to 3-5 bullet points. No boilerplate.
"""

SUMMARY_PROMPT = """\
Based on the following trial report, write a 2-3 line summary suitable for an experiment log.
Focus on: what was tried, the key outcome (success/failure/partial), and the most important finding.
Do NOT use markdown headers or bullet points. Just write 2-3 plain sentences.
"""


class Reporter:
    def __init__(
        self,
        base_dir: str,
        use_claude: bool = True,
        cli_path: str = "claude",
        model: str = "claude-sonnet-4-6",
    ):
        self.base_dir = Path(base_dir).resolve()
        self.use_claude = use_claude
        self.cli_path = cli_path
        self.model = model

    def generate_report(
        self,
        trial: TrialRecord,
        reason: str,
    ) -> tuple[str, str]:
        sandbox_root = self.base_dir / trial.sandbox_path
        codes_root = sandbox_root / "codes"
        outputs_root = self.base_dir / trial.outputs_path
        results_root = self.base_dir / trial.results_path

        codes_files = [str(p.relative_to(codes_root)) for p in codes_root.rglob("*") if p.is_file()] if codes_root.exists() else []
        output_files = [str(p.relative_to(outputs_root)) for p in outputs_root.rglob("*") if p.is_file()] if outputs_root.exists() else []
        result_files = [
            str(p.relative_to(results_root))
            for p in results_root.rglob("*")
            if p.is_file() and p.name != "REPORT.md"
        ] if results_root.exists() else []

        run_stdout = outputs_root / "experiment_stdout.log"
        run_stderr = outputs_root / "experiment_stderr.log"
        eval_stdout = results_root / "eval_stdout.log"
        eval_stderr = results_root / "eval_stderr.log"

        raw_data = self._gather_raw_data(
            trial, reason, codes_root, codes_files, output_files, result_files,
            run_stdout, run_stderr, eval_stdout, eval_stderr,
        )

        ai_analysis = self._ai_analyze(raw_data)
        ai_summary = self._ai_summarize(ai_analysis, trial) if ai_analysis else ""

        terminated_header = ""
        if trial.terminated:
            terminated_header = "[TERMINATED-DURING-EXPERIMENT]\n\n"

        metadata_section = (
            f"- Date: {trial.date}\n"
            f"- Trial ID: {trial.trial_id}\n"
            f"- Status: {trial.status.value}\n"
            f"- Selected project: {trial.selected_project or '(scratch)'}\n"
            f"- Plan approved: {trial.plan_approved}\n"
            f"- Experiment iterations: {trial.experiment_iter}\n"
            f"- Eval iterations: {trial.eval_iter}\n"
            f"- Terminated: {trial.terminated}\n"
            f"- Termination reason: {trial.termination_reason or '(none)'}\n"
            f"- Final reason: {reason}\n"
            f"- Generated at: {datetime.now().isoformat()}\n"
        )

        artifacts_section = (
            "## Codebase Summary\n"
            f"- Code files in sandbox/codes: {len(codes_files)}\n"
            f"- Outputs files: {len(output_files)}\n"
            f"- Results files: {len(result_files)}\n\n"
            "## Outputs (raw)\n"
            + ("\n".join(f"- {f}" for f in output_files[:200]) or "- (none)")
            + "\n\n## Results (polished)\n"
            + ("\n".join(f"- {f}" for f in result_files[:200]) or "- (none)")
        )

        logs_section = (
            "## Experiment Log Tail (stdout)\n"
            "```\n" + _tail(run_stdout, 60) + "\n```\n\n"
            "## Experiment Log Tail (stderr)\n"
            "```\n" + _tail(run_stderr, 40) + "\n```\n\n"
            "## Eval Log Tail (stdout)\n"
            "```\n" + _tail(eval_stdout, 60) + "\n```\n\n"
            "## Eval Log Tail (stderr)\n"
            "```\n" + _tail(eval_stderr, 40) + "\n```"
        )

        if ai_analysis:
            analysis_section = f"## AI Analysis\n\n{ai_analysis}"
        else:
            analysis_section = (
                "## What Worked\n"
                "- Analyze output/result artifacts above to determine successful components.\n\n"
                "## Implementation Summary\n"
                "- Trial code is stored under sandbox/<date>/<trial>/codes.\n"
                "- Raw runtime artifacts are stored under sandbox/<date>/<trial>/outputs.\n"
                "- Polished artifacts and report are stored under results/<date>/<trial>.\n\n"
                "## Next-Trial Directions\n"
                "- Revisit failure traces in stderr sections.\n"
                "- Tighten run.sh/eval.sh contracts to reduce non-determinism.\n"
                "- Compare this REPORT with previous trials in EXPERIMENT_LOGS.md before next PLAN cycle."
            )

        report = (
            f"{terminated_header}# REPORT - {trial.trial_name}\n\n"
            f"{metadata_section}\n"
            f"{artifacts_section}\n\n"
            f"{logs_section}\n\n"
            f"{analysis_section}\n"
        )

        report_path = results_root / "REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

        if ai_summary:
            summary = ai_summary
        else:
            summary = (
                f"iters(exp={trial.experiment_iter}, eval={trial.eval_iter}); "
                f"outputs={len(output_files)}, results={len(result_files)}"
            )
        if trial.terminated:
            summary = f"[TERMINATED] {summary}"

        report_rel = str(report_path.relative_to(self.base_dir))
        return report_rel, summary

    def _gather_raw_data(
        self,
        trial: TrialRecord,
        reason: str,
        codes_root: Path,
        codes_files: list[str],
        output_files: list[str],
        result_files: list[str],
        run_stdout: Path,
        run_stderr: Path,
        eval_stdout: Path,
        eval_stderr: Path,
    ) -> str:
        parts: list[str] = []
        parts.append(
            f"Trial: {trial.trial_name}\n"
            f"Date: {trial.date}\n"
            f"Project: {trial.selected_project or '(scratch)'}\n"
            f"Experiment iterations: {trial.experiment_iter}\n"
            f"Eval iterations: {trial.eval_iter}\n"
            f"Terminated: {trial.terminated}\n"
            f"Reason: {reason}\n"
        )

        # Key code file contents (first 5 files, truncated)
        if codes_root.exists():
            parts.append("### Key code files:")
            for fname in codes_files[:5]:
                content = _read_truncated(codes_root / fname, 2000)
                if content:
                    parts.append(f"#### {fname}\n```\n{content}\n```")

        parts.append(f"### Experiment stdout (tail):\n```\n{_tail(run_stdout, 80)}\n```")
        parts.append(f"### Experiment stderr (tail):\n```\n{_tail(run_stderr, 60)}\n```")
        parts.append(f"### Eval stdout (tail):\n```\n{_tail(eval_stdout, 80)}\n```")
        parts.append(f"### Eval stderr (tail):\n```\n{_tail(eval_stderr, 60)}\n```")

        # Prior history
        experiment_logs = _read_truncated(self.base_dir / "EXPERIMENT_LOGS.md", 5000)
        if experiment_logs:
            parts.append(f"### EXPERIMENT_LOGS.md (recent):\n{experiment_logs}")

        prior_reports = self._read_prior_reports()
        for i, report_text in enumerate(prior_reports, 1):
            parts.append(f"### Prior Report #{i}:\n{report_text}")

        return "\n\n".join(parts)

    def _read_prior_reports(self, max_reports: int = 3) -> list[str]:
        results_dir = self.base_dir / "results"
        if not results_dir.exists():
            return []
        report_files = sorted(results_dir.rglob("REPORT.md"), reverse=True)
        reports: list[str] = []
        for rf in report_files[:max_reports]:
            text = _read_truncated(rf, 3000)
            if text:
                reports.append(text)
        return reports

    def _ai_analyze(self, raw_data: str) -> str:
        if not self.use_claude:
            return ""
        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--system-prompt",
                    REPORT_ANALYSIS_PROMPT,
                    "--model",
                    self.model,
                    raw_data,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _ai_summarize(self, analysis: str, trial: TrialRecord) -> str:
        if not self.use_claude:
            return ""
        context = (
            f"Trial: {trial.trial_name}, "
            f"Project: {trial.selected_project or '(scratch)'}, "
            f"Iters: exp={trial.experiment_iter} eval={trial.eval_iter}\n\n"
            f"{analysis}"
        )
        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--system-prompt",
                    SUMMARY_PROMPT,
                    "--model",
                    self.model,
                    context,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return ""
