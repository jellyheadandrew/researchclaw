"""Tests for researchclaw/config.py"""

from __future__ import annotations

import os

import pytest

from researchclaw.config import Config, load_config


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config(tmp_path):
    """Minimal valid config.yaml with only base_dir set."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"base_dir: {str(tmp_path)}\n")
    return str(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# TestLoadConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_load_minimal_config(self, minimal_config, tmp_path):
        config = load_config(minimal_config)
        assert isinstance(config, Config)
        assert config.base_dir == str(tmp_path)

    def test_defaults_are_applied(self, minimal_config):
        config = load_config(minimal_config)
        assert config.project_name == "researchclaw-project"
        assert config.github_remote == "origin"
        assert config.github_branch == "main"
        assert config.messenger_type == "stdio"
        assert config.llm_provider == "claude_cli"
        assert config.llm_model == "claude-opus-4-6"
        assert config.watcher_poll_interval == 10
        assert config.runner_always_confirm is True

    def test_load_llm_section(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "llm:\n"
            "  provider: anthropic\n"
            "  model: claude-opus-4-6\n"
            "  api_key_env: MY_API_KEY\n"
        )
        config = load_config(str(cfg))
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-opus-4-6"
        assert config.llm_api_key_env == "MY_API_KEY"

    def test_load_llm_openai_base_url(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "llm:\n"
            "  provider: openai\n"
            "  model: gpt-4o\n"
            "  base_url: https://custom.example.com/v1\n"
        )
        config = load_config(str(cfg))
        assert config.llm_openai_base_url == "https://custom.example.com/v1"

    def test_llm_base_url_empty_by_default(self, minimal_config):
        config = load_config(minimal_config)
        assert config.llm_openai_base_url == ""

    def test_load_messenger_telegram(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "messenger:\n"
            "  type: telegram\n"
            "  telegram_chat_id: '12345'\n"
            "  telegram_poll_timeout: 20\n"
        )
        config = load_config(str(cfg))
        assert config.messenger_type == "telegram"
        assert config.telegram_chat_id == "12345"
        assert config.telegram_poll_timeout == 20

    def test_load_runner_venv_path(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "runner:\n"
            "  venv_path: /opt/my_venv\n"
        )
        config = load_config(str(cfg))
        assert config.runner_venv_path == "/opt/my_venv"

    def test_load_runner_default_env(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "runner:\n"
            "  default_env: myenv\n"
        )
        config = load_config(str(cfg))
        assert config.runner_default_env == "myenv"

    def test_load_custom_sandbox_ignore(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "sandbox:\n"
            "  copy_ignore_patterns:\n"
            "    - '.git'\n"
            "    - 'data/'\n"
        )
        config = load_config(str(cfg))
        assert ".git" in config.sandbox_copy_ignore
        assert "data/" in config.sandbox_copy_ignore

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "does_not_exist.yaml"))

    def test_base_dir_defaults_to_config_parent(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("project_name: test\n")  # no base_dir
        config = load_config(str(cfg))
        assert config.base_dir == str(tmp_path)

    def test_watcher_settings(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "watcher:\n"
            "  poll_interval: 5\n"
            "  heartbeat_timeout: 120\n"
            "  status_update_interval: 3600\n"
        )
        config = load_config(str(cfg))
        assert config.watcher_poll_interval == 5
        assert config.watcher_heartbeat_timeout == 120
        assert config.watcher_status_update_interval == 3600

    def test_report_settings(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"base_dir: {str(tmp_path)}\n"
            "report:\n"
            "  include_diff: false\n"
            "  include_log_tail: 25\n"
        )
        config = load_config(str(cfg))
        assert config.report_include_diff is False
        assert config.report_log_tail == 25

    def test_dotenv_loaded(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"base_dir: {str(tmp_path)}\n")
        env_file = tmp_path / ".env"
        env_file.write_text("RESEARCHCLAW_TEST_DOTENV_KEY=loaded_value\n")

        # Remove any pre-existing value so we're testing dotenv loading
        os.environ.pop("RESEARCHCLAW_TEST_DOTENV_KEY", None)
        try:
            load_config(str(cfg))
            assert os.environ.get("RESEARCHCLAW_TEST_DOTENV_KEY") == "loaded_value"
        finally:
            os.environ.pop("RESEARCHCLAW_TEST_DOTENV_KEY", None)
