"""AIWorker — Fresh AI context per invocation (Ralph loop philosophy).

Each call spawns a fresh Claude CLI subprocess with no prior conversation context.
State is communicated entirely through files and explicit context strings,
ensuring each retry iteration gets a clean perspective on the problem.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


EXPERIMENT_FIX_PROMPT = """\
You are a code debugger for ResearchClaw experiments.
An experiment script (run.sh) failed. Your job is to fix the code so it succeeds.

You have:
- The experiment plan (PLAN.md)
- The failure logs (stdout + stderr)
- The current source code files

Fix ONLY what is needed to make the experiment succeed.
Output the full corrected content for each file that needs changing, using this format:

### FILE: <relative_path>
```
<full file content>
```

If no changes are needed (e.g., the failure is environmental), output:
NO_CHANGES_NEEDED: <explanation>
"""

EVAL_FIX_PROMPT = """\
You are a code debugger for ResearchClaw evaluation scripts.
An evaluation script (eval.sh) failed. Your job is to fix the eval code so it succeeds.

You have:
- The experiment outputs
- The failure logs (eval stdout + stderr)
- The current eval source code files

Fix ONLY what is needed to make the evaluation succeed.
Output the full corrected content for each file that needs changing, using this format:

### FILE: <relative_path>
```
<full file content>
```

If no changes are needed (e.g., the failure is environmental), output:
NO_CHANGES_NEEDED: <explanation>
"""

EXPERIMENT_IMPLEMENT_PROMPT = """\
You are a code implementation assistant for ResearchClaw experiments.

Given the experiment plan (PLAN.md) and optional starting project code, implement the
complete experiment. You must produce:

1. All source code files under codes/ that the experiment needs.
2. A run.sh bash script that executes the experiment.

Constraints:
- run.sh must use #!/usr/bin/env bash and set -euo pipefail.
- run.sh can read any file but must write outputs ONLY to the directory specified
  by the $RC_OUTPUTS_DIR environment variable.
- Code files go under codes/ (use codes/<filename> paths).
- run.sh goes at the top level (just "run.sh", not "codes/run.sh").
- Keep the implementation focused and minimal. No unnecessary abstractions.
- If existing code files are provided, build upon them rather than starting from scratch.

Output the full content for each file using this format:

### FILE: <relative_path>
```
<full file content>
```
"""

EVAL_IMPLEMENT_PROMPT = """\
You are an evaluation implementation assistant for ResearchClaw.

Given the experiment plan, the experiment source code, and the experiment outputs,
implement the evaluation. You must produce:

1. Evaluation code files under eval_codes/.
2. An eval.sh bash script that runs the evaluation.

Constraints:
- eval.sh must use #!/usr/bin/env bash and set -euo pipefail.
- eval.sh can read files from $RC_OUTPUTS_DIR (experiment outputs) and eval_codes/.
- eval.sh must write results ONLY to the directory specified by $RC_RESULTS_DIR.
- Results should include quantitative metrics, visualizations, or summaries as appropriate.
- Evaluation code goes under eval_codes/ (use eval_codes/<filename> paths).
- eval.sh goes at the top level (just "eval.sh", not "eval_codes/eval.sh").

Output the full content for each file using this format:

### FILE: <relative_path>
```
<full file content>
```
"""


class AIWorker:
    """Spawns a fresh Claude CLI per invocation — no prior conversation context."""

    def __init__(
        self,
        cli_path: str = "claude",
        model: str = "claude-sonnet-4-6",
    ):
        self.cli_path = cli_path
        self.model = model

    def invoke(
        self,
        system_prompt: str,
        context: str,
        timeout: int = 180,
    ) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--system-prompt",
                    system_prompt,
                    "--model",
                    self.model,
                    context,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, result.stdout.strip()
            stderr = result.stderr.strip() if result.stderr else ""
            return False, f"CLI returned code {result.returncode}: {stderr}"
        except subprocess.TimeoutExpired:
            return False, "AI worker timed out"
        except Exception as e:
            return False, f"AI worker error: {e}"

    def invoke_with_file_context(
        self,
        system_prompt: str,
        context_files: dict[str, Path],
        user_prompt: str,
        timeout: int = 180,
    ) -> tuple[bool, str]:
        parts: list[str] = []
        for label, path in context_files.items():
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    if len(content) > 5000:
                        content = content[:5000] + "\n... (truncated)"
                    parts.append(f"### {label}\n```\n{content}\n```")
                except Exception:
                    parts.append(f"### {label}\n(unreadable)")
            else:
                parts.append(f"### {label}\n(missing)")

        parts.append(f"### User instruction\n{user_prompt}")
        context = "\n\n".join(parts)
        return self.invoke(system_prompt, context, timeout)

    def implement_experiment(
        self,
        plan_path: Path,
        codes_dir: Path,
    ) -> tuple[bool, str]:
        """Generate experiment code from PLAN.md (fresh AI context)."""
        context_files: dict[str, Path] = {"PLAN.md": plan_path}

        if codes_dir.exists():
            for f in sorted(codes_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(codes_dir))
                    context_files[f"codes/{rel}"] = f

        return self.invoke_with_file_context(
            EXPERIMENT_IMPLEMENT_PROMPT,
            context_files,
            "Implement the experiment according to the plan. "
            "Produce all necessary code files and run.sh.",
            timeout=300,
        )

    def implement_eval(
        self,
        plan_path: Path,
        codes_dir: Path,
        outputs_dir: Path,
        eval_codes_dir: Path,
    ) -> tuple[bool, str]:
        """Generate evaluation code from PLAN.md and experiment outputs (fresh AI context)."""
        context_files: dict[str, Path] = {"PLAN.md": plan_path}

        if codes_dir.exists():
            for f in sorted(codes_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(codes_dir))
                    context_files[f"codes/{rel}"] = f

        if outputs_dir.exists():
            for f in sorted(outputs_dir.rglob("*"))[:15]:
                if f.is_file():
                    rel = str(f.relative_to(outputs_dir))
                    context_files[f"outputs/{rel}"] = f

        if eval_codes_dir.exists():
            for f in sorted(eval_codes_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(eval_codes_dir))
                    context_files[f"eval_codes/{rel}"] = f

        return self.invoke_with_file_context(
            EVAL_IMPLEMENT_PROMPT,
            context_files,
            "Implement the evaluation according to the plan. "
            "Produce eval code files and eval.sh that analyze the experiment outputs.",
            timeout=300,
        )

    def fix_experiment(
        self,
        plan_path: Path,
        codes_dir: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> tuple[bool, str]:
        context_files: dict[str, Path] = {"PLAN.md": plan_path}

        if codes_dir.exists():
            for f in sorted(codes_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(codes_dir))
                    context_files[f"codes/{rel}"] = f

        context_files["experiment_stdout.log"] = stdout_log
        context_files["experiment_stderr.log"] = stderr_log

        return self.invoke_with_file_context(
            EXPERIMENT_FIX_PROMPT,
            context_files,
            "The experiment failed. Analyze the logs and fix the code.",
        )

    def fix_eval(
        self,
        eval_codes_dir: Path,
        outputs_dir: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> tuple[bool, str]:
        context_files: dict[str, Path] = {}

        if eval_codes_dir.exists():
            for f in sorted(eval_codes_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(eval_codes_dir))
                    context_files[f"eval_codes/{rel}"] = f

        if outputs_dir.exists():
            for f in sorted(outputs_dir.rglob("*"))[:10]:
                if f.is_file():
                    rel = str(f.relative_to(outputs_dir))
                    context_files[f"outputs/{rel}"] = f

        context_files["eval_stdout.log"] = stdout_log
        context_files["eval_stderr.log"] = stderr_log

        return self.invoke_with_file_context(
            EVAL_FIX_PROMPT,
            context_files,
            "The evaluation failed. Analyze the logs and fix the eval code.",
        )

    @staticmethod
    def parse_file_patches(ai_output: str) -> dict[str, str]:
        """Parse AI output for file patches in the ### FILE: <path> format."""
        if "NO_CHANGES_NEEDED" in ai_output:
            return {}

        patches: dict[str, str] = {}
        current_file: str | None = None
        in_code_block = False
        lines: list[str] = []

        for line in ai_output.splitlines():
            if line.startswith("### FILE:"):
                if current_file and lines:
                    patches[current_file] = "\n".join(lines)
                current_file = line[len("### FILE:"):].strip()
                lines = []
                in_code_block = False
            elif current_file is not None:
                if line.startswith("```") and not in_code_block:
                    in_code_block = True
                elif line.startswith("```") and in_code_block:
                    in_code_block = False
                elif in_code_block:
                    lines.append(line)

        if current_file and lines:
            patches[current_file] = "\n".join(lines)

        return patches
