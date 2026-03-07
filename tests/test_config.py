from __future__ import annotations

from pathlib import Path

import yaml

from researchclaw.config import ResearchClawConfig


class TestDefaultValues:
    def test_defaults_match_spec(self) -> None:
        config = ResearchClawConfig()
        assert config.model == "claude-opus-4-6"
        assert config.max_trials == 20
        assert config.max_retries == 5
        assert config.autopilot is False
        assert config.experiment_timeout_seconds == 3600
        assert config.python_command == "python3"
        assert config.llm_timeout_seconds == 0


class TestSaveLoadRoundTrip:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        original = ResearchClawConfig(
            model="gpt-4",
            max_trials=10,
            max_retries=3,
            autopilot=True,
            experiment_timeout_seconds=7200,
            python_command="python3.12",
        )
        yaml_path = tmp_path / "config.yaml"
        original.save_to_yaml(yaml_path)
        loaded = ResearchClawConfig.load_from_yaml(yaml_path)
        assert loaded.model == "gpt-4"
        assert loaded.max_trials == 10
        assert loaded.max_retries == 3
        assert loaded.autopilot is True
        assert loaded.experiment_timeout_seconds == 7200
        assert loaded.python_command == "python3.12"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        yaml_path = tmp_path / "nested" / "dir" / "config.yaml"
        config.save_to_yaml(yaml_path)
        assert yaml_path.exists()

    def test_saved_yaml_is_valid(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        yaml_path = tmp_path / "config.yaml"
        config.save_to_yaml(yaml_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert data["model"] == "claude-opus-4-6"
        assert data["max_trials"] == 20


class TestLoadFromYaml:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        config = ResearchClawConfig.load_from_yaml(tmp_path / "nonexistent.yaml")
        assert config.model == "claude-opus-4-6"
        assert config.max_trials == 20

    def test_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        config = ResearchClawConfig.load_from_yaml(yaml_path)
        assert config.model == "claude-opus-4-6"

    def test_partial_yaml_fills_defaults(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text("model: gpt-4\nmax_trials: 5\n")
        config = ResearchClawConfig.load_from_yaml(yaml_path)
        assert config.model == "gpt-4"
        assert config.max_trials == 5
        assert config.max_retries == 5  # default
        assert config.autopilot is False  # default

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "extra.yaml"
        yaml_path.write_text("model: gpt-4\nunknown_key: hello\n")
        config = ResearchClawConfig.load_from_yaml(yaml_path)
        assert config.model == "gpt-4"
        assert not hasattr(config, "unknown_key")


class TestProjectOverridesGlobal:
    def test_project_overrides_global(self, tmp_path: Path, monkeypatch: object) -> None:
        # Setup global config
        global_dir = tmp_path / "global_config" / "researchclaw"
        global_dir.mkdir(parents=True)
        global_yaml = global_dir / "config.yaml"
        global_yaml.write_text(
            "model: claude-opus-4-6\nmax_trials: 20\nmax_retries: 5\n"
            "autopilot: false\nexperiment_timeout_seconds: 3600\npython_command: python3\n"
        )

        # Setup project config
        project_dir = tmp_path / "my_project"
        project_settings = project_dir / "sandbox_researchclaw" / "project_settings"
        project_settings.mkdir(parents=True)
        project_yaml = project_settings / "researchclaw.yaml"
        project_yaml.write_text("max_trials: 50\nautopilot: true\n")

        # Monkeypatch global config path
        import researchclaw.config as config_mod
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", global_yaml)  # type: ignore[attr-defined]

        config = ResearchClawConfig.load_merged_config(project_dir)
        # Global values preserved
        assert config.model == "claude-opus-4-6"
        assert config.max_retries == 5
        # Project overrides applied
        assert config.max_trials == 50
        assert config.autopilot is True

    def test_no_project_config_uses_global(self, tmp_path: Path, monkeypatch: object) -> None:
        # Setup global config only
        global_dir = tmp_path / "global_config" / "researchclaw"
        global_dir.mkdir(parents=True)
        global_yaml = global_dir / "config.yaml"
        global_yaml.write_text("model: gpt-4\nmax_trials: 15\n")

        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        import researchclaw.config as config_mod
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", global_yaml)  # type: ignore[attr-defined]

        config = ResearchClawConfig.load_merged_config(project_dir)
        assert config.model == "gpt-4"
        assert config.max_trials == 15
        assert config.max_retries == 5  # default

    def test_no_global_no_project_returns_defaults(self, tmp_path: Path, monkeypatch: object) -> None:
        import researchclaw.config as config_mod
        monkeypatch.setattr(  # type: ignore[attr-defined]
            config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "nonexistent.yaml"
        )

        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = ResearchClawConfig.load_merged_config(project_dir)
        assert config.model == "claude-opus-4-6"
        assert config.max_trials == 20
        assert config.max_retries == 5
        assert config.autopilot is False
