from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


GLOBAL_CONFIG_DIR = Path(os.path.expanduser("~/.config/researchclaw"))
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "config.yaml"


@dataclass
class ResearchClawConfig:
    """Configuration for ResearchClaw, loaded from researchclaw.yaml."""

    model: str = "claude-opus-4-6"
    max_trials: int = 20
    max_retries: int = 5
    autopilot: bool = False
    experiment_timeout_seconds: int = 3600
    python_command: str = "python3"
    provider: str = "claude_cli"
    api_key: str = ""
    display_trials: int = 10

    def save_to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> ResearchClawConfig:
        """Load config from a YAML file. Returns defaults for missing fields."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return cls()
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def load_merged_config(cls, project_dir: str | Path) -> ResearchClawConfig:
        """Load global config, then overlay project config on top.

        Global: ~/.config/researchclaw/config.yaml
        Project: {project_dir}/sandbox_researchclaw/project_settings/researchclaw.yaml
        """
        global_cfg = cls.load_from_yaml(GLOBAL_CONFIG_PATH)
        project_path = (
            Path(project_dir)
            / "sandbox_researchclaw"
            / "project_settings"
            / "researchclaw.yaml"
        )
        if not project_path.exists():
            return global_cfg
        with open(project_path) as f:
            project_data = yaml.safe_load(f)
        if not isinstance(project_data, dict):
            return global_cfg
        merged = asdict(global_cfg)
        valid_fields = {f.name for f in fields(cls)}
        for k, v in project_data.items():
            if k in valid_fields:
                merged[k] = v
        return cls(**merged)
