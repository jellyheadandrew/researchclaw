from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.panel import Panel


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
    "/verbose": "Toggle verbose mode (show thinking)",
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

    def send_status(self, message: str) -> None:
        """Send a status/transition message (dim, no panel). Default: call send()."""
        self.send(message)

    def send_stream(self, chunks: Iterator[str]) -> str:
        """Stream LLM response chunks to the user and return accumulated text.

        Default: collect all chunks and send as a single message.
        """
        accumulated = "".join(chunks)
        self.send(accumulated)
        return accumulated

    @contextmanager  # type: ignore[arg-type]
    def show_thinking(self) -> Iterator[None]:
        """Context manager that shows a 'Thinking...' indicator.

        Default: no-op context manager.
        """
        yield

    def send_thinking(self, thinking: str) -> None:
        """Display thinking content (shown only in verbose mode).

        Default: no-op.
        """

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
        self._verbose: bool = False
        if history_path is not None:
            self._session: PromptSession[str] = PromptSession(
                history=FileHistory(str(history_path))
            )
        else:
            self._session = PromptSession()

    def send(self, message: str) -> None:
        """Display a message in a Rich Panel with cyan border."""
        panel = Panel(
            message,
            border_style="cyan",
            title="[bold cyan]ResearchClaw[/bold cyan]",
        )
        self._console.print(panel)

    def send_status(self, message: str) -> None:
        """Display a status/transition message in dim text (no panel)."""
        self._console.print(f"[dim]{message}[/dim]")

    def send_stream(self, chunks: Iterator[str]) -> str:
        """Stream LLM response in a Rich Panel using Live display, return accumulated text."""
        accumulated = ""
        panel = Panel(
            accumulated or " ",
            border_style="cyan",
            title="[bold cyan]ResearchClaw[/bold cyan]",
        )
        with Live(panel, console=self._console, refresh_per_second=8) as live:
            for chunk in chunks:
                accumulated += chunk
                live.update(Panel(
                    accumulated,
                    border_style="cyan",
                    title="[bold cyan]ResearchClaw[/bold cyan]",
                ))
        return accumulated

    @contextmanager  # type: ignore[arg-type]
    def show_thinking(self) -> Iterator[None]:
        """Return a Rich Status spinner context manager with 'Thinking...' and dots spinner."""
        with self._console.status("Thinking...", spinner="dots"):
            yield

    def send_thinking(self, thinking: str) -> None:
        """Display thinking content in dim italic text when _verbose is True."""
        if self._verbose:
            self._console.print(f"[dim italic]{thinking}[/dim italic]")

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
            raw = self._session.prompt(HTML("<ansigreen>you&gt; </ansigreen>"))
            parsed = parse_input(raw)

            if isinstance(parsed, SlashCommand):
                if parsed.name == "/help":
                    self._print_help()
                    continue
                if parsed.name == "/quit":
                    raise SystemExit("User quit via /quit")
                if parsed.name == "/verbose":
                    self._verbose = not self._verbose
                    state = "on" if self._verbose else "off"
                    self._console.print(f"[dim]Verbose mode {state}.[/dim]")
                    continue
                return parsed

            if parsed.text:
                return parsed
            # Empty input — re-prompt

    def _print_help(self) -> None:
        """Print available slash commands in a yellow-bordered Panel."""
        lines = [f"  {cmd:20s} {desc}" for cmd, desc in SLASH_COMMANDS.items()]
        content = "[bold]Available commands:[/bold]\n" + "\n".join(lines)
        panel = Panel(content, border_style="yellow")
        self._console.print(panel)
