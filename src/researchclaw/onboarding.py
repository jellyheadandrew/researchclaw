from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml

from researchclaw.config import GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_PATH, ResearchClawConfig
from researchclaw.llm.installer import ensure_package
from researchclaw.repl import ChatInterface, SlashCommand, UserMessage

# ---------------------------------------------------------------------------
# Named constants (previously hardcoded / magic values)
# ---------------------------------------------------------------------------

NVM_VERSION = "v0.40.1"
NVM_INSTALL_URL = f"https://raw.githubusercontent.com/nvm-sh/nvm/{NVM_VERSION}/install.sh"
CLAUDE_CODE_NPM_PACKAGE = "@anthropic-ai/claude-code"
NVM_INSTALL_TIMEOUT = 120
NODE_INSTALL_TIMEOUT = 300
NPM_INSTALL_TIMEOUT = 300


def _is_claude_cli_available() -> bool:
    """Check if 'claude' CLI is available on PATH."""
    return shutil.which("claude") is not None


def _ensure_npm_available(chat: ChatInterface) -> bool:
    """Ensure npm is available on PATH. Installs nvm + Node LTS if needed.

    Returns True if npm is available (already or newly installed), False otherwise.
    """
    if shutil.which("npm") is not None:
        return True

    chat.send("npm not found. Attempting to install Node.js via nvm...")

    # Check if nvm is already installed but not loaded
    nvm_dir = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
    nvm_script = os.path.join(nvm_dir, "nvm.sh")

    if not os.path.isfile(nvm_script):
        # Install nvm via curl, fall back to wget
        chat.send("Installing nvm...")
        if shutil.which("curl") is not None:
            install_cmd = f"curl -o- {NVM_INSTALL_URL} | bash"
        elif shutil.which("wget") is not None:
            install_cmd = f"wget -qO- {NVM_INSTALL_URL} | bash"
        else:
            chat.send(
                "[bold red]Error:[/bold red] Neither curl nor wget found. "
                "Cannot install nvm automatically."
            )
            return False

        try:
            subprocess.run(
                ["bash", "-c", install_cmd],
                check=True,
                capture_output=True,
                text=True,
                timeout=NVM_INSTALL_TIMEOUT,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            chat.send("[bold red]Error:[/bold red] Failed to install nvm.")
            return False

    # Source nvm and install Node LTS
    chat.send("Installing Node.js LTS via nvm...")
    shell_cmd = (
        f'export NVM_DIR="{nvm_dir}" && '
        f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && '
        f"nvm install --lts && nvm use --lts && "
        f"which npm"
    )
    try:
        result = subprocess.run(
            ["bash", "-c", shell_cmd],
            check=True,
            capture_output=True,
            text=True,
            timeout=NODE_INSTALL_TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        chat.send("[bold red]Error:[/bold red] Failed to install Node.js via nvm.")
        return False

    # Extract the npm path from the last line of output and add its dir to PATH
    npm_path = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if npm_path:
        npm_bin_dir = os.path.dirname(npm_path)
        os.environ["PATH"] = npm_bin_dir + os.pathsep + os.environ.get("PATH", "")

    if shutil.which("npm") is not None:
        chat.send("[green]npm installed successfully.[/green]")
        return True

    chat.send("[bold red]Error:[/bold red] npm still not found after installation.")
    return False


def _ensure_claude_cli_available(chat: ChatInterface) -> bool:
    """Ensure the claude CLI is available on PATH. Installs npm + Claude Code if needed.

    Returns True if claude CLI is available (already or newly installed), False otherwise.
    """
    if shutil.which("claude") is not None:
        return True

    chat.send("Claude CLI not found. Attempting auto-install...")

    if not _ensure_npm_available(chat):
        chat.send(
            "[bold red]Error:[/bold red] Cannot install Claude Code without npm."
        )
        return False

    chat.send("Installing Claude Code via npm...")
    # Use the npm that's on PATH (possibly newly installed)
    npm_path = shutil.which("npm")
    if npm_path is None:
        return False

    try:
        subprocess.run(
            [npm_path, "install", "-g", CLAUDE_CODE_NPM_PACKAGE],
            check=True,
            capture_output=True,
            text=True,
            timeout=NPM_INSTALL_TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        chat.send("[bold red]Error:[/bold red] Failed to install Claude Code via npm.")
        return False

    if shutil.which("claude") is not None:
        chat.send("[green]Claude Code installed successfully.[/green]")
        return True

    chat.send(
        "[bold red]Error:[/bold red] claude CLI still not found after installation."
    )
    return False


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
        # Tier 0: verify claude CLI is available, auto-install if not
        if not _is_claude_cli_available():
            choice = _prompt_choice(
                chat,
                "\n[bold yellow]Claude CLI not found.[/bold yellow]\n"
                "  [bold]yes[/bold]   — Auto-install npm + Claude Code\n"
                "  [bold]no[/bold]    — Choose a different provider tier\n"
                "\nAuto-install? (yes/no):",
                {"yes", "no"},
            )
            if choice == "yes":
                success = _ensure_claude_cli_available(chat)
                if not success:
                    # Auto-install failed — offer retry or fallback
                    retry_choice = _prompt_choice(
                        chat,
                        "\nAuto-install failed.\n"
                        "  [bold]retry[/bold]  — Try installing again\n"
                        "  [bold]other[/bold]  — Choose a different provider tier\n"
                        "  [bold]quit[/bold]   — Exit\n"
                        "\nWhat would you like to do? (retry/other/quit):",
                        {"retry", "other", "quit"},
                    )
                    if retry_choice == "retry":
                        success = _ensure_claude_cli_available(chat)
                        if not success:
                            chat.send(
                                "[bold red]Auto-install failed again.[/bold red] "
                                "Falling back to provider selection."
                            )
                            return run_onboarding(chat)
                    elif retry_choice == "other":
                        return run_onboarding(chat)
                    else:
                        raise SystemExit("User chose to quit after failed auto-install.")
            else:
                # User declined auto-install — restart onboarding for different tier
                return run_onboarding(chat)
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
