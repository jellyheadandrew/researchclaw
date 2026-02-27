from __future__ import annotations

import subprocess


PLAN_SYSTEM_PROMPT = """You are a research planning assistant.
Given the previous plan draft, user direction, and current selected project,
produce an updated concise execution plan in markdown.
Keep it practical and testable.
"""


class PlanEngine:
    def __init__(self, use_claude: bool = True, cli_path: str = "claude", model: str = "claude-sonnet-4-6"):
        self.use_claude = use_claude
        self.cli_path = cli_path
        self.model = model

    def update_plan(self, prior_plan: str, user_message: str, selected_project: str | None) -> str:
        context = (
            f"Selected project: {selected_project or '(scratch)'}\n"
            f"Current plan draft:\n{prior_plan or '(empty)'}\n\n"
            f"User message:\n{user_message}\n"
        )

        if self.use_claude:
            try:
                result = subprocess.run(
                    [
                        self.cli_path,
                        "-p",
                        "--system-prompt",
                        PLAN_SYSTEM_PROMPT,
                        "--model",
                        self.model,
                        context,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass

        # Fallback plan drafting path (no external dependency)
        stripped = user_message.strip()
        if not prior_plan.strip():
            return (
                "## Plan Draft\n"
                f"- Goal update: {stripped}\n"
                "- Define run.sh execution contract\n"
                "- Define eval.sh evaluation contract\n"
                "- Identify success criteria and stopping conditions"
            )
        return prior_plan + f"\n- Update: {stripped}"
