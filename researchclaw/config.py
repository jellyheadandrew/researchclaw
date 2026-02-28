from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    base_dir: str
    messenger_type: str = "stdio"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    planner_use_claude: bool = True
    planner_claude_cli_path: str = "claude"
    planner_model: str = "claude-sonnet-4-6"
    planner_web_search: bool = True


def load_config(config_path: str = "config.yaml") -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    planner = raw.get("planner", {})
    messenger = raw.get("messenger", {})

    base_dir = str(raw.get("base_dir", str(path.parent / "workspace")))
    return Config(
        base_dir=base_dir,
        messenger_type=str(messenger.get("type", "stdio")),
        telegram_bot_token=str(messenger.get("bot_token", "")),
        telegram_chat_id=str(messenger.get("chat_id", "")),
        planner_use_claude=bool(planner.get("use_claude", True)),
        planner_claude_cli_path=str(planner.get("claude_cli_path", "claude")),
        planner_model=str(planner.get("model", "claude-sonnet-4-6")),
        planner_web_search=bool(planner.get("web_search", True)),
    )
