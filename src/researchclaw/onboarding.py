from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import yaml

from researchclaw.config import GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_PATH, ResearchClawConfig
from researchclaw.llm.installer import ensure_package
from researchclaw.repl import ChatInterface, SlashCommand, UserMessage


def _is_claude_cli_available() -> bool:
    """Check if 'claude' CLI is available on PATH."""
    return shutil.which("claude") is not None


def _prompt_choice(chat: ChatInterface, prompt: str, valid: set[str]) -> str:
    """Prompt user until they give a valid choice."""
    while True:
        chat.send(prompt)
        inp = chat.receive()
        if isinstance(inp, SlashCommand):
            if inp.name == "/quit":
                raise SystemExit("User quit during onboarding.")
            chat.send(f"Please enter one of: {', '.join(sorted(valid))}")
            continue
        text = inp.text.strip().lower()
        if text in valid:
            return text
        chat.send(f"Invalid choice '{text}'. Please enter one of: {', '.join(sorted(valid))}")


def _prompt_text(chat: ChatInterface, prompt: str) -> str:
    """Prompt user for free-text input."""
    chat.send(prompt)
    while True:
        inp = chat.receive()
        if isinstance(inp, SlashCommand):
            if inp.name == "/quit":
                raise SystemExit("User quit during onboarding.")
            chat.send("Please enter a text response.")
            continue
        text = inp.text.strip()
        if text:
            return text
        chat.send("Input cannot be empty. Please try again.")


def needs_onboarding() -> bool:
    """Check if onboarding is needed (no global config exists)."""
    return not GLOBAL_CONFIG_PATH.exists()


def run_onboarding(chat: ChatInterface) -> ResearchClawConfig:
    """Run first-time onboarding to configure LLM provider.

    Guides user through provider selection:
    - Tier 0 (skip): Uses claude CLI already on system. Zero extra packages.
    - Tier 1 (sdk): Installs claude-agent-sdk for richer integration.
    - Tier 2 (api): Uses anthropic or openai SDK with user's API key.

    Returns the created ResearchClawConfig.
    """
    chat.send("\n[bold]Welcome to ResearchClaw![/bold]")
    chat.send("Let's set up your LLM provider.\n")
    chat.send("Choose a provider tier:")
    chat.send("  [bold]skip[/bold]  — Use 'claude' CLI already on your system (Tier 0, no setup)")
    chat.send("  [bold]sdk[/bold]   — Install claude-agent-sdk for richer features (Tier 1)")
    chat.send("  [bold]api[/bold]   — Use Anthropic or OpenAI API with your key (Tier 2)")

    tier = _prompt_choice(chat, "\nEnter your choice (skip/sdk/api):", {"skip", "sdk", "api"})

    config = ResearchClawConfig()
    extra: dict[str, str] = {}

    if tier == "skip":
        # Tier 0: verify claude CLI is available
        if not _is_claude_cli_available():
            chat.send(
                "\n[bold red]Error:[/bold red] 'claude' CLI not found on PATH.\n"
                "Please install Claude Code CLI first:\n"
                "  npm install -g @anthropic-ai/claude-code\n\n"
                "Or re-run onboarding and choose 'sdk' or 'api' instead."
            )
            raise SystemExit("Claude CLI not found. Cannot proceed with Tier 0.")
        chat.send("\n[green]Claude CLI found. Using Tier 0 (claude CLI).[/green]")
        extra["provider"] = "claude_cli"

    elif tier == "sdk":
        # Tier 1: install claude-agent-sdk
        chat.send("\nInstalling claude-agent-sdk...")
        success = ensure_package("claude_agent_sdk", "claude-agent-sdk")
        if success:
            chat.send("[green]claude-agent-sdk installed successfully.[/green]")
        else:
            chat.send(
                "[yellow]Warning: Failed to install claude-agent-sdk. "
                "Falling back to claude CLI (Tier 0).[/yellow]"
            )
        extra["provider"] = "claude_agent_sdk" if success else "claude_cli"

    elif tier == "api":
        # Tier 2: select provider and enter API key
        provider_name = _prompt_choice(
            chat,
            "Which API provider? (anthropic/openai):",
            {"anthropic", "openai"},
        )
        api_key = _prompt_text(chat, f"Enter your {provider_name} API key:")

        pip_name = provider_name  # anthropic or openai
        chat.send(f"\nInstalling {pip_name} SDK...")
        success = ensure_package(provider_name, pip_name)
        if success:
            chat.send(f"[green]{pip_name} SDK installed successfully.[/green]")
        else:
            chat.send(
                f"[yellow]Warning: Failed to install {pip_name}. "
                "Falling back to claude CLI (Tier 0).[/yellow]"
            )

        extra["provider"] = provider_name if success else "claude_cli"
        extra["api_key"] = api_key

    # Save global config
    _save_global_config(config, extra)
    chat.send(f"\n[green]Configuration saved to {GLOBAL_CONFIG_PATH}[/green]")
    chat.send("Onboarding complete!\n")

    return config


def _save_global_config(config: ResearchClawConfig, extra: dict[str, str]) -> None:
    """Save config + extra provider fields to global config.yaml."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    data.update(extra)
    with open(GLOBAL_CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
