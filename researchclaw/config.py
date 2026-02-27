"""
config.py — Configuration loading from config.yaml + .env.

Secrets (API keys, bot tokens) live in a .env file next to config.yaml.
They are loaded into environment variables before the Config dataclass is
populated, so the rest of the codebase can read them via os.environ.

Non-secret configuration stays in config.yaml and is safe to commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv as _load_dotenv  # python-dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

DEFAULT_CONFIG_PATH = "config.yaml"


@dataclass
class Config:
    base_dir: str
    project_name: str = "researchclaw-project"
    github_remote: str = "origin"
    github_branch: str = "main"

    # LLM
    llm_provider: str = "claude_cli"       # claude_cli | anthropic | openai | ollama
    llm_model: str = "claude-opus-4-6"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"
    llm_claude_cli_path: str = "claude"    # path to claude binary (claude_cli provider only)
    llm_openai_base_url: str = ""          # override base URL for OpenAI-compatible providers
                                           # defaults: openai=https://api.openai.com/v1
                                           #           openrouter=https://openrouter.ai/api/v1
                                           #           deepseek=https://api.deepseek.com/v1
                                           #           ollama=http://localhost:11434/v1

    # Messenger — which transport to use for researcher communication
    messenger_type: str = "stdio"              # slack | telegram | stdio
    # Slack (native Socket Mode)
    slack_channel: str = "#research-agent"
    slack_bot_token_env: str = "SLACK_BOT_TOKEN"    # name of env var holding xoxb-... token
    slack_app_token_env: str = "SLACK_APP_TOKEN"    # name of env var holding xapp-... token
    # Telegram
    telegram_chat_id: str = ""                 # chat/group ID (not a secret; find via get_chat_id)
    telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN"  # name of the env var holding the token
    telegram_poll_timeout: int = 30            # long-polling timeout in seconds
    telegram_poll_interval: float = 1.0        # retry delay after an empty poll

    # Watcher
    watcher_poll_interval: int = 10
    watcher_status_update_interval: int = 7200
    watcher_heartbeat_timeout: int = 300
    watcher_gpu_idle_threshold: int = 60

    # Environment management
    env_backend: str = "venv"            # "venv" or "conda"

    # Sandbox
    sandbox_copy_ignore: list[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "*.pyc", "wandb", "outputs",
        "checkpoints", "*.pt", "*.ckpt", "*.safetensors", "*.bin", "node_modules",
    ])
    sandbox_auto_continue_sequential: bool = True

    # Runner
    runner_default_env: str = ""
    runner_venv_path: str = ""
    runner_always_confirm: bool = True

    # Report
    report_include_diff: bool = True
    report_log_tail: int = 50
    report_include_gpu_info: bool = True

    # Agent — agentic loop settings
    agent_max_iterations: int = 10  # max tool-use iterations per user message


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
    """Load config.yaml and an optional .env file next to it.

    Secrets (tokens, API keys) are read from the .env file into environment
    variables before the Config dataclass is constructed.  Non-secret settings
    come from config.yaml.

    .env is optional — if it doesn't exist the function continues without error.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Copy the template from the spec and fill in your values."
        )

    # Load secrets from .env before reading Config fields so callers can rely
    # on os.environ already being populated when they access env-var fields.
    env_path = path.parent / ".env"
    if _HAS_DOTENV:
        _load_dotenv(env_path, override=False)  # override=False: real env vars win

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    llm = raw.get("llm", {})
    messenger = raw.get("messenger", {})
    # Legacy top-level slack key still supported for backwards compatibility
    slack_legacy = raw.get("slack", {})
    watcher = raw.get("watcher", {})
    sandbox = raw.get("sandbox", {})
    runner = raw.get("runner", {})
    report = raw.get("report", {})
    env_section = raw.get("env", {})
    agent_section = raw.get("agent", {})

    base_dir = raw.get("base_dir", str(Path(config_path).parent))

    return Config(
        base_dir=base_dir,
        project_name=raw.get("project_name", "researchclaw-project"),
        github_remote=raw.get("github_remote", "origin"),
        github_branch=raw.get("github_branch", "main"),

        llm_provider=llm.get("provider", "claude_cli"),
        llm_model=llm.get("model", "claude-opus-4-6"),
        llm_api_key_env=llm.get("api_key_env", "ANTHROPIC_API_KEY"),
        llm_claude_cli_path=llm.get("cli_path", "claude"),
        llm_openai_base_url=llm.get("base_url", ""),

        messenger_type=messenger.get("type", "stdio"),
        slack_channel=messenger.get("slack_channel", slack_legacy.get("channel", "#research-agent")),
        slack_bot_token_env=messenger.get("slack_bot_token_env", "SLACK_BOT_TOKEN"),
        slack_app_token_env=messenger.get("slack_app_token_env", "SLACK_APP_TOKEN"),
        telegram_chat_id=str(messenger.get("telegram_chat_id", "")),
        telegram_bot_token_env=messenger.get("telegram_bot_token_env", "TELEGRAM_BOT_TOKEN"),
        telegram_poll_timeout=messenger.get("telegram_poll_timeout", 30),
        telegram_poll_interval=float(messenger.get("telegram_poll_interval", 1.0)),

        watcher_poll_interval=watcher.get("poll_interval", 10),
        watcher_status_update_interval=watcher.get("status_update_interval", 7200),
        watcher_heartbeat_timeout=watcher.get("heartbeat_timeout", 300),
        watcher_gpu_idle_threshold=watcher.get("gpu_idle_threshold", 60),

        env_backend=env_section.get("backend", "venv"),

        sandbox_copy_ignore=sandbox.get("copy_ignore_patterns", [
            ".git", "__pycache__", "*.pyc", "wandb", "outputs",
            "checkpoints", "*.pt", "*.ckpt", "*.safetensors", "*.bin", "node_modules",
        ]),
        sandbox_auto_continue_sequential=sandbox.get("auto_continue_sequential", True),

        runner_default_env=runner.get("default_env", ""),
        runner_venv_path=runner.get("venv_path", ""),
        runner_always_confirm=runner.get("always_confirm", True),

        report_include_diff=report.get("include_diff", True),
        report_log_tail=report.get("include_log_tail", 50),
        report_include_gpu_info=report.get("include_gpu_info", True),

        agent_max_iterations=agent_section.get("max_iterations", 10),
    )
