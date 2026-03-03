from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from researchclaw.onboarding import (
    _is_claude_cli_available,
    _prompt_choice,
    _prompt_text,
    _save_global_config,
    needs_onboarding,
    run_onboarding,
)
from researchclaw.config import ResearchClawConfig
from researchclaw.repl import ChatInterface, ChatInput, SlashCommand, UserMessage
import researchclaw.onboarding as onboarding_mod
import researchclaw.config as config_mod


# --- Fake ChatInterface for testing ---

class FakeChat(ChatInterface):
    """Fake chat interface that returns pre-programmed responses."""

    def __init__(self, responses: list[ChatInput]) -> None:
        self.responses = list(responses)
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def send_image(self, path: str, caption: str | None = None) -> None:
        pass

    def receive(self) -> ChatInput:
        if not self.responses:
            raise SystemExit("No more responses")
        return self.responses.pop(0)


# --- Tests for needs_onboarding ---

class TestNeedsOnboarding:
    def test_needs_onboarding_no_config(self, tmp_path: Path, monkeypatch: object) -> None:
        fake_path = tmp_path / "config" / "researchclaw" / "config.yaml"
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_PATH", fake_path)  # type: ignore[attr-defined]
        assert needs_onboarding() is True

    def test_no_onboarding_when_config_exists(self, tmp_path: Path, monkeypatch: object) -> None:
        fake_path = tmp_path / "config" / "researchclaw" / "config.yaml"
        fake_path.parent.mkdir(parents=True)
        fake_path.write_text("model: claude-opus-4-6\n")
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_PATH", fake_path)  # type: ignore[attr-defined]
        assert needs_onboarding() is False


# --- Tests for _prompt_choice ---

class TestPromptChoice:
    def test_valid_choice_first_try(self) -> None:
        chat = FakeChat([UserMessage("skip")])
        result = _prompt_choice(chat, "Choose:", {"skip", "sdk", "api"})
        assert result == "skip"

    def test_valid_choice_case_insensitive(self) -> None:
        chat = FakeChat([UserMessage("SDK")])
        result = _prompt_choice(chat, "Choose:", {"skip", "sdk", "api"})
        assert result == "sdk"

    def test_invalid_then_valid(self) -> None:
        chat = FakeChat([UserMessage("invalid"), UserMessage("api")])
        result = _prompt_choice(chat, "Choose:", {"skip", "sdk", "api"})
        assert result == "api"
        # Should have sent an error message for the invalid choice
        assert any("Invalid choice" in m for m in chat.sent)

    def test_slash_command_then_valid(self) -> None:
        chat = FakeChat([SlashCommand("/status", ""), UserMessage("skip")])
        result = _prompt_choice(chat, "Choose:", {"skip", "sdk", "api"})
        assert result == "skip"

    def test_quit_during_prompt(self) -> None:
        chat = FakeChat([SlashCommand("/quit", "")])
        try:
            _prompt_choice(chat, "Choose:", {"skip", "sdk", "api"})
            assert False, "Expected SystemExit"
        except SystemExit:
            pass


# --- Tests for _prompt_text ---

class TestPromptText:
    def test_returns_text(self) -> None:
        chat = FakeChat([UserMessage("my-api-key")])
        result = _prompt_text(chat, "Enter key:")
        assert result == "my-api-key"

    def test_empty_input_reprompts(self) -> None:
        chat = FakeChat([UserMessage(""), UserMessage("valid")])
        result = _prompt_text(chat, "Enter key:")
        assert result == "valid"

    def test_quit_during_text(self) -> None:
        chat = FakeChat([SlashCommand("/quit", "")])
        try:
            _prompt_text(chat, "Enter key:")
            assert False, "Expected SystemExit"
        except SystemExit:
            pass


# --- Tests for _is_claude_cli_available ---

class TestClaudeCLICheck:
    def test_claude_available(self, monkeypatch: object) -> None:
        monkeypatch.setattr(onboarding_mod.shutil, "which", lambda cmd: "/usr/bin/claude")  # type: ignore[attr-defined]
        assert _is_claude_cli_available() is True

    def test_claude_not_available(self, monkeypatch: object) -> None:
        monkeypatch.setattr(onboarding_mod.shutil, "which", lambda cmd: None)  # type: ignore[attr-defined]
        assert _is_claude_cli_available() is False


# --- Tests for _save_global_config ---

class TestSaveGlobalConfig:
    def test_saves_config_file(self, tmp_path: Path, monkeypatch: object) -> None:
        fake_dir = tmp_path / "config" / "researchclaw"
        fake_path = fake_dir / "config.yaml"
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_DIR", fake_dir)  # type: ignore[attr-defined]
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_PATH", fake_path)  # type: ignore[attr-defined]

        config = ResearchClawConfig()
        _save_global_config(config, {"provider": "claude_cli"})

        assert fake_path.exists()
        data = yaml.safe_load(fake_path.read_text())
        assert data["provider"] == "claude_cli"
        assert data["model"] == "claude-opus-4-6"

    def test_saves_api_key(self, tmp_path: Path, monkeypatch: object) -> None:
        fake_dir = tmp_path / "config" / "researchclaw"
        fake_path = fake_dir / "config.yaml"
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_DIR", fake_dir)  # type: ignore[attr-defined]
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_PATH", fake_path)  # type: ignore[attr-defined]

        config = ResearchClawConfig()
        _save_global_config(config, {"provider": "anthropic", "api_key": "sk-test-123"})

        data = yaml.safe_load(fake_path.read_text())
        assert data["api_key"] == "sk-test-123"
        assert data["provider"] == "anthropic"


# --- Tests for run_onboarding ---

class TestRunOnboarding:
    def _patch_globals(self, monkeypatch: object, tmp_path: Path) -> Path:
        """Redirect global config to tmp_path and return the config file path."""
        fake_dir = tmp_path / "config" / "researchclaw"
        fake_path = fake_dir / "config.yaml"
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_DIR", fake_dir)  # type: ignore[attr-defined]
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_PATH", fake_path)  # type: ignore[attr-defined]
        return fake_path

    def test_tier0_skip_with_claude_available(self, tmp_path: Path, monkeypatch: object) -> None:
        config_path = self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod.shutil, "which", lambda cmd: "/usr/bin/claude")  # type: ignore[attr-defined]

        chat = FakeChat([UserMessage("skip")])
        config = run_onboarding(chat)

        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "claude_cli"
        assert isinstance(config, ResearchClawConfig)
        assert any("Onboarding complete" in m for m in chat.sent)

    def test_tier0_skip_without_claude_exits(self, tmp_path: Path, monkeypatch: object) -> None:
        self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod.shutil, "which", lambda cmd: None)  # type: ignore[attr-defined]

        chat = FakeChat([UserMessage("skip")])
        try:
            run_onboarding(chat)
            assert False, "Expected SystemExit"
        except SystemExit:
            pass
        assert any("not found" in m for m in chat.sent)

    def test_tier1_sdk_install_success(self, tmp_path: Path, monkeypatch: object) -> None:
        config_path = self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod, "ensure_package", lambda pkg, pip: True)  # type: ignore[attr-defined]

        chat = FakeChat([UserMessage("sdk")])
        config = run_onboarding(chat)

        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "claude_agent_sdk"
        assert isinstance(config, ResearchClawConfig)

    def test_tier1_sdk_install_failure_falls_back(self, tmp_path: Path, monkeypatch: object) -> None:
        config_path = self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod, "ensure_package", lambda pkg, pip: False)  # type: ignore[attr-defined]

        chat = FakeChat([UserMessage("sdk")])
        config = run_onboarding(chat)

        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "claude_cli"
        assert any("Falling back" in m for m in chat.sent)

    def test_tier2_anthropic_success(self, tmp_path: Path, monkeypatch: object) -> None:
        config_path = self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod, "ensure_package", lambda pkg, pip: True)  # type: ignore[attr-defined]

        chat = FakeChat([
            UserMessage("api"),
            UserMessage("anthropic"),
            UserMessage("sk-ant-test-key"),
        ])
        config = run_onboarding(chat)

        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "anthropic"
        assert data["api_key"] == "sk-ant-test-key"
        assert isinstance(config, ResearchClawConfig)

    def test_tier2_openai_success(self, tmp_path: Path, monkeypatch: object) -> None:
        config_path = self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod, "ensure_package", lambda pkg, pip: True)  # type: ignore[attr-defined]

        chat = FakeChat([
            UserMessage("api"),
            UserMessage("openai"),
            UserMessage("sk-openai-test-key"),
        ])
        config = run_onboarding(chat)

        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "openai"
        assert data["api_key"] == "sk-openai-test-key"

    def test_tier2_install_failure_falls_back(self, tmp_path: Path, monkeypatch: object) -> None:
        config_path = self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod, "ensure_package", lambda pkg, pip: False)  # type: ignore[attr-defined]

        chat = FakeChat([
            UserMessage("api"),
            UserMessage("anthropic"),
            UserMessage("sk-ant-key"),
        ])
        config = run_onboarding(chat)

        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "claude_cli"
        assert data["api_key"] == "sk-ant-key"
        assert any("Falling back" in m for m in chat.sent)

    def test_welcome_message_displayed(self, tmp_path: Path, monkeypatch: object) -> None:
        self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod.shutil, "which", lambda cmd: "/usr/bin/claude")  # type: ignore[attr-defined]

        chat = FakeChat([UserMessage("skip")])
        run_onboarding(chat)

        assert any("Welcome to ResearchClaw" in m for m in chat.sent)
        assert any("Choose a provider tier" in m for m in chat.sent)

    def test_config_saved_path_displayed(self, tmp_path: Path, monkeypatch: object) -> None:
        config_path = self._patch_globals(monkeypatch, tmp_path)
        monkeypatch.setattr(onboarding_mod.shutil, "which", lambda cmd: "/usr/bin/claude")  # type: ignore[attr-defined]

        chat = FakeChat([UserMessage("skip")])
        run_onboarding(chat)

        assert any("Configuration saved" in m for m in chat.sent)


# --- Tests for CLI integration ---

class TestCLIOnboardingIntegration:
    @staticmethod
    def _patch_fsm(monkeypatch: object) -> None:
        """Patch FSM engine to avoid real terminal interaction."""
        import researchclaw.cli as cli_mod
        monkeypatch.setattr(cli_mod, "TerminalChat", lambda **kw: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(cli_mod, "_build_handlers", lambda: {})  # type: ignore[attr-defined]
        class FakeEngine:
            def __init__(self, *a, **kw):
                pass
            def run(self):
                raise SystemExit("done")
        monkeypatch.setattr(cli_mod, "FSMEngine", FakeEngine)  # type: ignore[attr-defined]

    def test_cli_triggers_onboarding_when_no_config(self, tmp_path: Path, monkeypatch: object) -> None:
        """Test that CLI triggers onboarding when no global config exists."""
        from click.testing import CliRunner
        from researchclaw.cli import main

        self._patch_fsm(monkeypatch)

        fake_config_path = tmp_path / "config" / "researchclaw" / "config.yaml"
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", fake_config_path)  # type: ignore[attr-defined]
        monkeypatch.setattr(onboarding_mod, "GLOBAL_CONFIG_PATH", fake_config_path)  # type: ignore[attr-defined]

        # needs_onboarding returns True → onboarding would be called
        # We mock run_onboarding since it needs interactive input
        called = []
        def mock_onboarding(chat: ChatInterface) -> ResearchClawConfig:
            called.append(True)
            return ResearchClawConfig()

        monkeypatch.setattr(onboarding_mod, "run_onboarding", mock_onboarding)  # type: ignore[attr-defined]
        # Also patch cli module's imported reference
        import researchclaw.cli as cli_mod
        monkeypatch.setattr(cli_mod, "run_onboarding", mock_onboarding)  # type: ignore[attr-defined]
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: True)  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert len(called) == 1

    def test_cli_skips_onboarding_when_config_exists(self, tmp_path: Path, monkeypatch: object) -> None:
        """Test that CLI skips onboarding when global config already exists."""
        from click.testing import CliRunner
        from researchclaw.cli import main

        self._patch_fsm(monkeypatch)

        called = []
        def mock_onboarding(chat: ChatInterface) -> ResearchClawConfig:
            called.append(True)
            return ResearchClawConfig()

        import researchclaw.cli as cli_mod
        monkeypatch.setattr(cli_mod, "run_onboarding", mock_onboarding)  # type: ignore[attr-defined]
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert len(called) == 0
