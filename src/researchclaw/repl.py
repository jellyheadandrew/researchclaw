from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console


# --- Message types returned by ChatInterface.receive() ---

class SlashCommand(NamedTuple):
    """A parsed slash command from user input."""
    name: str
    args: str


class UserMessage(NamedTuple):
    """A regular (non-command) message from the user."""
    text: str


# Union type for receive() return
ChatInput = SlashCommand | UserMessage


# --- Slash command definitions ---

SLASH_COMMANDS: dict[str, str] = {
    "/approve": "Approve the current plan or action",
    "/abort": "Abort the current trial",
    "/autopilot": "Enable autopilot mode",
    "/autopilot-stop": "Disable autopilot mode",
    "/status": "Show trial status table",
    "/quit": "Quit ResearchClaw",
    "/help": "Show available commands",
}


def parse_input(raw: str) -> ChatInput:
    """Parse raw user input into a SlashCommand or UserMessage.

    Slash commands start with '/' and match a known command name.
    Unknown slash-prefixed input is treated as a regular UserMessage.
    """
    stripped = raw.strip()
    if stripped.startswith("/"):
        parts = stripped.split(None, 1)
        cmd = parts[0].lower()
        if cmd in SLASH_COMMANDS:
            args = parts[1] if len(parts) > 1 else ""
            return SlashCommand(name=cmd, args=args)
    return UserMessage(text=stripped)


# --- Abstract ChatInterface ---

class ChatInterface(ABC):
    """Abstract chat interface for user interaction."""

    @abstractmethod
    def send(self, message: str) -> None:
        """Send a message to the user."""
        ...

    @abstractmethod
    def send_image(self, path: str, caption: str | None = None) -> None:
        """Send an image to the user."""
        ...

    @abstractmethod
    def receive(self) -> ChatInput:
        """Receive input from the user.

        Returns a SlashCommand or UserMessage.
        """
        ...


# --- TerminalChat implementation ---

class TerminalChat(ChatInterface):
    """Terminal-based chat interface using prompt-toolkit and rich."""

    def __init__(self, history_path: str | Path | None = None) -> None:
        self._console = Console()
        if history_path is not None:
            self._session: PromptSession[str] = PromptSession(
                history=FileHistory(str(history_path))
            )
        else:
            self._session = PromptSession()

    def send(self, message: str) -> None:
        """Display a message via rich console."""
        self._console.print(message)

    def send_image(self, path: str, caption: str | None = None) -> None:
        """Display image path (terminal cannot render images inline)."""
        msg = f"[Image: {path}]"
        if caption:
            msg += f" {caption}"
        self._console.print(msg)

    def receive(self) -> ChatInput:
        """Prompt user for input, parse slash commands.

        /help prints available commands and re-prompts.
        /quit raises SystemExit.
        """
        while True:
            raw = self._session.prompt("you> ")
            parsed = parse_input(raw)

            if isinstance(parsed, SlashCommand):
                if parsed.name == "/help":
                    self._print_help()
                    continue
                if parsed.name == "/quit":
                    raise SystemExit("User quit via /quit")
                return parsed

            if parsed.text:
                return parsed
            # Empty input — re-prompt

    def _print_help(self) -> None:
        """Print available slash commands."""
        self._console.print("\n[bold]Available commands:[/bold]")
        for cmd, desc in SLASH_COMMANDS.items():
            self._console.print(f"  {cmd:20s} {desc}")
        self._console.print()
