"""ResearchEngine — conversational research with paper search, summarization, brainstorming."""

from __future__ import annotations

import subprocess
from pathlib import Path


SEARCH_PAPERS_PROMPT = """\
Search the web for recent academic papers, arxiv preprints, or research articles about:
{query}

Return a structured list of up to 5 results. For each result include:
- Title
- Authors (if available)
- URL
- 1-2 sentence summary of key contribution

Format as markdown bullet points.
"""

SUMMARIZE_PROMPT = """\
Summarize the following content concisely. Focus on:
- Key contribution or finding
- Methodology used
- Main results or conclusions
- Relevance to practical implementation

Keep to 1-2 paragraphs.
"""

BRAINSTORM_PROMPT = """\
You are a creative research brainstorming assistant.
Given the researcher's topic, prior experiment history, and any search findings,
generate 3-5 concrete experiment ideas.

For each idea:
- State a clear hypothesis
- Suggest a practical approach (what to implement)
- Note expected outcomes and how to measure success

Be specific and actionable. Avoid vague suggestions.
"""

CONVERSATIONAL_PROMPT = """\
You are a research assistant for ResearchClaw.
Help the researcher think through their ideas, answer questions about research
methodology, and suggest relevant directions based on their experiment history.

Be concise and practical. If the researcher asks something you can answer,
answer it directly. If they seem to be brainstorming, help develop their ideas.
"""


class ResearchEngine:
    def __init__(
        self,
        base_dir: str,
        cli_path: str = "claude",
        model: str = "claude-sonnet-4-6",
        use_claude: bool = True,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.cli_path = cli_path
        self.model = model
        self.use_claude = use_claude

    def search_papers(self, query: str) -> str:
        if not self.use_claude:
            return f"(Paper search unavailable without Claude CLI. Query: {query})"

        prompt = SEARCH_PAPERS_PROMPT.format(query=query)
        try:
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
        return f"(Search failed for: {query})"

    def summarize(self, text: str) -> str:
        if not self.use_claude:
            return text[:500] + ("..." if len(text) > 500 else "")

        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--system-prompt",
                    SUMMARIZE_PROMPT,
                    "--model",
                    self.model,
                    text[:10000],
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return text[:500] + ("..." if len(text) > 500 else "")

    def brainstorm(self, topic: str, history: str = "") -> str:
        context = f"Topic: {topic}\n"
        if history:
            truncated = history[:5000]
            if len(history) > 5000:
                truncated += "\n... (truncated)"
            context += f"\nExperiment history:\n{truncated}\n"

        if not self.use_claude:
            return (
                f"Brainstorm ideas for: {topic}\n"
                "- (AI brainstorming unavailable without Claude CLI)\n"
                "- Consider reviewing EXPERIMENT_LOGS.md for patterns\n"
                "- Try searching papers with /research search <query>"
            )

        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--system-prompt",
                    BRAINSTORM_PROMPT,
                    "--allowedTools",
                    "WebSearch",
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
        return f"(Brainstorm failed for: {topic})"

    def chat(self, message: str, history: str = "") -> str:
        context = message
        if history:
            truncated = history[:3000]
            context = f"Experiment history context:\n{truncated}\n\nUser: {message}"

        if not self.use_claude:
            return "(Conversational research requires Claude CLI.)"

        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--system-prompt",
                    CONVERSATIONAL_PROMPT,
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
        return "(Research chat unavailable.)"

    def generate_nudge(self, history: str = "") -> str:
        """Generate a periodic research idea nudge based on experiment history."""
        context = "Generate a single interesting research idea or direction worth exploring."
        if history:
            truncated = history[:3000]
            context += f"\n\nBased on prior experiments:\n{truncated}"

        if not self.use_claude:
            return ""

        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    "--system-prompt",
                    "You are a research idea generator. Given experiment history, suggest one novel direction to explore. Keep it to 2-3 sentences.",
                    "--allowedTools",
                    "WebSearch",
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
        return ""
