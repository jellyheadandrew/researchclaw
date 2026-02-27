"""
command_translator.py — Natural language → shell command translation.

The LLM generates shell commands from researcher instructions. This module:
  1. Uses the LLM to translate a natural language request into a shell command
  2. Validates all output paths in the generated command against PathValidator
  3. Shows the command to the researcher for approval before any execution

CRITICAL: A generated command that writes outside the allowed sandbox is BLOCKED
and the researcher is told why. This prevents the LLM from accidentally (or
maliciously) writing to github_codes/ or other protected directories.
"""

from __future__ import annotations

import logging

from .access_control import PathValidator
from .llm import LLMProvider
from .models import TrialInfo

logger = logging.getLogger("researchclaw.command_translator")

TRANSLATION_SYSTEM_PROMPT = """\
You are a shell command generator for a research automation system.
Your job is to convert a researcher's natural language instruction into
a single shell command (or short shell script).

Rules:
- Output ONLY the shell command, no explanation, no markdown.
- The working directory is the current trial's sandbox directory (provided in context).
- All file paths must be relative to the sandbox directory, OR use the absolute
  sandbox path when needed.
- NEVER write to paths outside the sandbox directory.
- NEVER use sudo, rm -rf, or other destructive operations.
- If the instruction is ambiguous, output: ASK: <your clarifying question>
"""


class CommandTranslator:
    """Translates natural language instructions into validated shell commands."""

    def __init__(self, llm: LLMProvider, validator: PathValidator):
        self.llm = llm
        self.validator = validator

    def translate(
        self,
        natural_language: str,
        trial: TrialInfo,
        context: dict | None = None,
    ) -> str:
        """
        Convert a natural language instruction to a shell command.

        Args:
            natural_language: The researcher's instruction (e.g. "run train.py with lr=0.001")
            trial: The active TrialInfo (provides sandbox path context)
            context: Optional additional context (recent log output, file list, etc.)

        Returns:
            A shell command string, or "ASK: <question>" if clarification is needed.

        Raises:
            PermissionError: If the generated command would write outside the sandbox.
        """
        ctx_parts = [
            f"Sandbox directory: {trial.sandbox_path}",
            f"Report directory: {trial.report_path}",
            f"Research goal: {trial.goal or '(not specified)'}",
        ]
        if context:
            if "file_list" in context:
                ctx_parts.append(f"Files in sandbox: {context['file_list']}")
            if "log_tail" in context:
                ctx_parts.append(f"Recent log output:\n{context['log_tail']}")

        user_message = "\n".join(ctx_parts) + f"\n\nInstruction: {natural_language}"

        cmd = self.llm.complete(
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=512,
        ).strip()

        logger.debug("LLM generated command: %s", cmd)

        # Don't validate clarification requests
        if cmd.startswith("ASK:"):
            return cmd

        # Validate all output paths before returning
        try:
            self.validator.validate_shell_command(cmd)
        except PermissionError as e:
            logger.error("Generated command failed path validation: %s", e)
            raise PermissionError(
                f"The generated command would write outside the sandbox.\n"
                f"Command: {cmd}\n"
                f"Violation: {e}\n"
                f"This command has been BLOCKED."
            )

        return cmd

    def parse_output_paths(self, cmd: str) -> list[str]:
        """Delegate to PathValidator's shell command parser."""
        return self.validator._parse_output_paths(cmd)
