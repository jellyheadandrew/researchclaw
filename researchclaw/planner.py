from __future__ import annotations

import subprocess
from pathlib import Path


PLAN_SYSTEM_PROMPT = """\
You are a research experiment planning assistant for ResearchClaw.

Your job is to help the researcher iterate on a concrete, testable experiment plan.
Given the prior plan draft, user direction, selected starting project, and historical
context (past experiment logs and reports), produce an updated execution plan in markdown.

Guidelines:
- Keep the plan practical and testable in a single trial.
- Reference prior trial outcomes when available to avoid repeating known failures.
- If web search results are provided, incorporate relevant findings.
- Structure the plan with: Hypothesis, Approach, run.sh contract, eval.sh contract,
  Success Criteria, and Stopping Conditions.
- Be concise. No boilerplate.
"""

AUTOPILOT_PLAN_PROMPT = """\
You are an autonomous research experiment planner for ResearchClaw (autopilot mode).

You must produce a COMPLETE, ready-to-implement experiment plan without human interaction.
Based on past experiment logs and reports, web search results, and the selected project,
decide the next most valuable experiment to run.

Requirements:
- Analyze past trial outcomes to identify what has been tried and what gaps remain.
- Propose a novel or incremental experiment that builds on prior results.
- The plan must be immediately actionable: specific enough for a developer to implement.
- Structure: Hypothesis, Approach, run.sh contract, eval.sh contract, Success Criteria,
  Stopping Conditions.
- If no prior history exists, propose a baseline/smoke-test experiment.
- Be concise and decisive. No hedging or multiple options.
"""

WEB_SEARCH_PROMPT = """\
Search the web for recent research trends, techniques, or tools relevant to:
{query}

Summarize the top 3-5 most relevant findings in bullet points.
Focus on practical, implementable ideas.
"""


class PlanEngine:
    def __init__(
        self,
        use_claude: bool = True,
        cli_path: str = "claude",
        model: str = "claude-sonnet-4-6",
        base_dir: str = "",
        web_search_enabled: bool = True,
    ):
        self.use_claude = use_claude
        self.cli_path = cli_path
        self.model = model
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.web_search_enabled = web_search_enabled

    def proactive_search(
        self,
        selected_project: str | None,
        experiment_logs: str = "",
        prior_reports: list[str] | None = None,
    ) -> str:
        """Auto-search for recent research trends based on project and trial context."""
        if not self.web_search_enabled or not self.use_claude:
            return ""

        query_parts: list[str] = []
        if selected_project and selected_project != "scratch":
            query_parts.append(selected_project)

        if prior_reports:
            latest = prior_reports[0][:500]
            query_parts.append(latest)
        elif experiment_logs.strip():
            lines = experiment_logs.strip().splitlines()[-3:]
            query_parts.append(" ".join(lines))

        if not query_parts:
            query_parts.append("recent research methodology and experiment design trends")

        return self._web_search(" ".join(query_parts)[:300])

    def update_plan(
        self,
        prior_plan: str,
        user_message: str,
        selected_project: str | None,
        experiment_logs: str = "",
        prior_reports: list[str] | None = None,
        autopilot: bool = False,
    ) -> str:
        history_block = self._build_history_block(experiment_logs, prior_reports)

        search_block = ""
        if self.web_search_enabled and self.use_claude:
            query = self._derive_search_query(user_message, selected_project, prior_plan)
            if query:
                search_block = self._web_search(query)

        context = (
            f"Selected project: {selected_project or '(scratch)'}\n\n"
        )
        if history_block:
            context += f"## Prior experiment history\n{history_block}\n\n"
        if search_block:
            context += f"## Web search findings\n{search_block}\n\n"
        context += (
            f"## Current plan draft\n{prior_plan or '(empty)'}\n\n"
            f"## User message\n{user_message}\n"
        )

        system_prompt = AUTOPILOT_PLAN_PROMPT if autopilot else PLAN_SYSTEM_PROMPT

        if self.use_claude:
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
                    timeout=180,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass

        return self._fallback_plan(prior_plan, user_message)

    def _build_history_block(
        self, experiment_logs: str, prior_reports: list[str] | None
    ) -> str:
        parts: list[str] = []
        if experiment_logs.strip():
            truncated = experiment_logs[:5000]
            if len(experiment_logs) > 5000:
                truncated += "\n... (truncated)"
            parts.append(f"### EXPERIMENT_LOGS.md (recent)\n{truncated}")

        if prior_reports:
            for i, report in enumerate(prior_reports[:5], 1):
                truncated = report[:3000]
                if len(report) > 3000:
                    truncated += "\n... (truncated)"
                parts.append(f"### Prior Report #{i}\n{truncated}")

        return "\n\n".join(parts)

    def _derive_search_query(
        self, user_message: str, selected_project: str | None, prior_plan: str
    ) -> str:
        msg = user_message.strip()
        if not msg or msg.startswith("/"):
            return ""
        keywords = msg[:200]
        if selected_project and selected_project != "scratch":
            keywords = f"{selected_project}: {keywords}"
        return keywords

    def _web_search(self, query: str) -> str:
        try:
            prompt = WEB_SEARCH_PROMPT.format(query=query)
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--allowedTools",
                    "WebSearch",
                    "--model",
                    self.model,
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _fallback_plan(prior_plan: str, user_message: str) -> str:
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
