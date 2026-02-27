from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .models import TrialRecord


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return "(missing)"
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:]) if data else "(empty)"


class Reporter:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()

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

        terminated_header = ""
        if trial.terminated:
            terminated_header = "[TERMINATED-DURING-EXPERIMENT]\n\n"

        report = (
            f"{terminated_header}# REPORT - {trial.trial_name}\n\n"
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
            f"- Generated at: {datetime.now().isoformat()}\n\n"
            "## Codebase Summary\n"
            f"- Code files in sandbox/codes: {len(codes_files)}\n"
            f"- Outputs files: {len(output_files)}\n"
            f"- Results files: {len(result_files)}\n\n"
            "## Outputs (raw)\n"
            + ("\n".join(f"- {f}" for f in output_files[:200]) or "- (none)")
            + "\n\n## Results (polished)\n"
            + ("\n".join(f"- {f}" for f in result_files[:200]) or "- (none)")
            + "\n\n## Experiment Log Tail (stdout)\n"
            + "```\n"
            + _tail(run_stdout, 60)
            + "\n```\n\n"
            + "## Experiment Log Tail (stderr)\n"
            + "```\n"
            + _tail(run_stderr, 40)
            + "\n```\n\n"
            + "## Eval Log Tail (stdout)\n"
            + "```\n"
            + _tail(eval_stdout, 60)
            + "\n```\n\n"
            + "## Eval Log Tail (stderr)\n"
            + "```\n"
            + _tail(eval_stderr, 40)
            + "\n```\n\n"
            + "## What Worked\n"
            + "- Analyze output/result artifacts above to determine successful components.\n\n"
            + "## Implementation Summary\n"
            + "- Trial code is stored under sandbox/<date>/<trial>/codes.\n"
            + "- Raw runtime artifacts are stored under sandbox/<date>/<trial>/outputs.\n"
            + "- Polished artifacts and report are stored under results/<date>/<trial>.\n\n"
            + "## Next-Trial Directions\n"
            + "- Revisit failure traces in stderr sections.\n"
            + "- Tighten run.sh/eval.sh contracts to reduce non-determinism.\n"
            + "- Compare this REPORT with previous trials in EXPERIMENT_LOGS.md before next PLAN cycle.\n"
        )

        report_path = results_root / "REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

        summary = (
            f"iters(exp={trial.experiment_iter}, eval={trial.eval_iter}); "
            f"outputs={len(output_files)}, results={len(result_files)}"
        )
        if trial.terminated:
            summary = f"[TERMINATED] {summary}"

        report_rel = str(report_path.relative_to(self.base_dir))
        return report_rel, summary
