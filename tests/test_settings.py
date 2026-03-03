from __future__ import annotations

from pathlib import Path

import pytest

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.settings import (
    CONFIG_DESCRIPTIONS,
    EDITABLE_FIELDS,
    FIELD_TYPES,
    _coerce_value,
    _format_config_display,
    _parse_setting_change,
    _save_project_config,
    handle_settings,
)
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager


class FakeChatInterface:
    """Fake chat interface with pre-programmed responses."""

    def __init__(self, responses: list[ChatInput] | None = None) -> None:
        self.sent: list[str] = []
        self._responses = list(responses) if responses else []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def receive(self) -> ChatInput:
        if not self._responses:
            raise SystemExit("No more responses")
        return self._responses.pop(0)


def _setup_sandbox(project_dir: Path) -> Path:
    """Helper: initialize sandbox and create a trial."""
    SandboxManager.initialize(project_dir)
    return SandboxManager.create_trial(project_dir)


# --- Tests for _parse_setting_change ---


class TestParseSettingChange:
    """Tests for _parse_setting_change helper."""

    def test_set_field_value(self) -> None:
        result = _parse_setting_change("set max_retries 10")
        assert result == ("max_retries", "10")

    def test_set_field_to_value(self) -> None:
        result = _parse_setting_change("set max_retries to 10")
        assert result == ("max_retries", "10")

    def test_field_equals_value(self) -> None:
        result = _parse_setting_change("max_retries = 10")
        assert result == ("max_retries", "10")

    def test_field_equals_no_spaces(self) -> None:
        result = _parse_setting_change("max_retries=10")
        assert result == ("max_retries", "10")

    def test_field_space_value(self) -> None:
        result = _parse_setting_change("max_retries 10")
        assert result == ("max_retries", "10")

    def test_case_insensitive_set(self) -> None:
        result = _parse_setting_change("SET model claude-sonnet-4-5-20250929")
        assert result == ("model", "claude-sonnet-4-5-20250929")

    def test_string_value_with_spaces(self) -> None:
        result = _parse_setting_change("set model to claude-opus-4-6")
        assert result == ("model", "claude-opus-4-6")

    def test_empty_string(self) -> None:
        result = _parse_setting_change("")
        assert result is None

    def test_single_word_unknown(self) -> None:
        result = _parse_setting_change("hello")
        assert result is None

    def test_unknown_field_space_value(self) -> None:
        """Unknown field names in '<field> <value>' pattern should not parse."""
        result = _parse_setting_change("foobar 123")
        assert result is None

    def test_boolean_value(self) -> None:
        result = _parse_setting_change("set autopilot true")
        assert result == ("autopilot", "true")

    def test_set_field_to_with_extra_spaces(self) -> None:
        result = _parse_setting_change("  set  max_retries  to  10  ")
        assert result == ("max_retries", "10")


# --- Tests for _coerce_value ---


class TestCoerceValue:
    """Tests for _coerce_value helper."""

    def test_int_field(self) -> None:
        assert _coerce_value("max_retries", "10") == 10

    def test_int_field_invalid(self) -> None:
        assert _coerce_value("max_retries", "abc") is None

    def test_bool_true_variants(self) -> None:
        assert _coerce_value("autopilot", "true") is True
        assert _coerce_value("autopilot", "yes") is True
        assert _coerce_value("autopilot", "on") is True
        assert _coerce_value("autopilot", "1") is True

    def test_bool_false_variants(self) -> None:
        assert _coerce_value("autopilot", "false") is False
        assert _coerce_value("autopilot", "no") is False
        assert _coerce_value("autopilot", "off") is False
        assert _coerce_value("autopilot", "0") is False

    def test_bool_invalid(self) -> None:
        assert _coerce_value("autopilot", "maybe") is None

    def test_str_field(self) -> None:
        assert _coerce_value("model", "claude-opus-4-6") == "claude-opus-4-6"

    def test_unknown_field(self) -> None:
        assert _coerce_value("nonexistent_field", "val") is None

    def test_bool_case_insensitive(self) -> None:
        assert _coerce_value("autopilot", "TRUE") is True
        assert _coerce_value("autopilot", "False") is False


# --- Tests for _format_config_display ---


class TestFormatConfigDisplay:
    """Tests for _format_config_display helper."""

    def test_contains_all_fields(self) -> None:
        config = ResearchClawConfig()
        result = _format_config_display(config)
        for f_name in EDITABLE_FIELDS:
            assert f_name in result

    def test_shows_current_values(self) -> None:
        config = ResearchClawConfig(max_retries=7)
        result = _format_config_display(config)
        assert "7" in result

    def test_masks_api_key(self) -> None:
        config = ResearchClawConfig(api_key="sk-secret-key")
        result = _format_config_display(config)
        assert "sk-secret-key" not in result
        assert "***" in result

    def test_shows_empty_api_key_as_empty(self) -> None:
        config = ResearchClawConfig(api_key="")
        result = _format_config_display(config)
        # Empty api_key should show repr('')
        assert "''" in result

    def test_shows_descriptions(self) -> None:
        config = ResearchClawConfig()
        result = _format_config_display(config)
        for desc in CONFIG_DESCRIPTIONS.values():
            assert desc in result

    def test_includes_instructions(self) -> None:
        config = ResearchClawConfig()
        result = _format_config_display(config)
        assert "set max_retries" in result
        assert "back" in result


# --- Tests for _save_project_config ---


class TestSaveProjectConfig:
    """Tests for _save_project_config helper."""

    def test_saves_to_project_settings(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        config = ResearchClawConfig(max_retries=42)
        _save_project_config(config, tmp_path)

        config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        loaded = ResearchClawConfig.load_from_yaml(config_path)
        assert loaded.max_retries == 42

    def test_preserves_all_fields(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        config = ResearchClawConfig(
            model="test-model",
            max_trials=5,
            autopilot=True,
        )
        _save_project_config(config, tmp_path)

        config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        loaded = ResearchClawConfig.load_from_yaml(config_path)
        assert loaded.model == "test-model"
        assert loaded.max_trials == 5
        assert loaded.autopilot is True


# --- Tests for CONFIG_DESCRIPTIONS and FIELD_TYPES ---


class TestConfigMetadata:
    """Tests for config metadata dicts."""

    def test_all_fields_have_descriptions(self) -> None:
        for f in EDITABLE_FIELDS:
            assert f in CONFIG_DESCRIPTIONS, f"Missing description for {f}"

    def test_all_fields_have_types(self) -> None:
        for f in EDITABLE_FIELDS:
            assert f in FIELD_TYPES, f"Missing type for {f}"

    def test_editable_fields_match_dataclass(self) -> None:
        from dataclasses import fields as dc_fields

        dc_names = {f.name for f in dc_fields(ResearchClawConfig)}
        for f in EDITABLE_FIELDS:
            assert f in dc_names, f"Editable field {f} not in ResearchClawConfig"


# --- Tests for handle_settings ---


class TestHandleSettingsBasic:
    """Tests for handle_settings handler basics."""

    def test_returns_decide_on_back(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("back")])
        result = handle_settings(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_returns_decide_on_exit(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("exit")])
        result = handle_settings(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_returns_decide_on_q(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("q")])
        result = handle_settings(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_none_chat_returns_decide(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        result = handle_settings(trial_dir, meta, ResearchClawConfig(), None)
        assert result == State.DECIDE

    def test_shows_settings_on_entry(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[UserMessage("back")])
        handle_settings(trial_dir, meta, ResearchClawConfig(), chat)
        assert any("[SETTINGS]" in m for m in chat.sent)
        assert any("max_retries" in m for m in chat.sent)

    def test_quit_slash_raises_system_exit(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[SlashCommand("/quit", "")])
        with pytest.raises(SystemExit):
            handle_settings(trial_dir, meta, ResearchClawConfig(), chat)

    def test_unknown_slash_reprompts(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        chat = FakeChatInterface(responses=[
            SlashCommand("/approve", ""),
            UserMessage("back"),
        ])
        result = handle_settings(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE
        assert any("not available" in m for m in chat.sent)


class TestHandleSettingsChanges:
    """Tests for setting value changes via handle_settings."""

    def test_change_int_setting(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set max_retries 10"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.max_retries == 10
        assert any("max_retries" in m and "10" in m for m in chat.sent)

    def test_change_bool_setting(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(autopilot=False)
        chat = FakeChatInterface(responses=[
            UserMessage("set autopilot true"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.autopilot is True

    def test_change_string_setting(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set model claude-sonnet-4-5-20250929"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.model == "claude-sonnet-4-5-20250929"

    def test_equals_syntax(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("max_retries = 15"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.max_retries == 15

    def test_saves_to_project_config(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set max_retries 20"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)

        # Verify saved to file
        config_path = (
            SandboxManager.sandbox_path(tmp_path)
            / "project_settings"
            / "researchclaw.yaml"
        )
        loaded = ResearchClawConfig.load_from_yaml(config_path)
        assert loaded.max_retries == 20

    def test_invalid_value_shows_error(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set max_retries abc"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.max_retries == 5  # unchanged
        assert any("Invalid value" in m for m in chat.sent)

    def test_unknown_field_shows_error(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set nonexistent_field 10"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert any("Unknown setting" in m for m in chat.sent)

    def test_unparseable_input_shows_help(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("change everything"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert any("Could not parse" in m for m in chat.sent)

    def test_multiple_changes(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set max_retries 10"),
            UserMessage("set max_trials 50"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.max_retries == 10
        assert config.max_trials == 50

    def test_shows_old_and_new_value(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig(max_retries=5)
        chat = FakeChatInterface(responses=[
            UserMessage("set max_retries 10"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        # Should show both old and new values
        assert any("5" in m and "10" in m for m in chat.sent)

    def test_change_display_trials(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set display_trials 20"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.display_trials == 20

    def test_change_experiment_timeout(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set experiment_timeout_seconds 7200"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.experiment_timeout_seconds == 7200

    def test_set_to_syntax(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set max_retries to 8"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.max_retries == 8

    def test_change_provider(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set provider anthropic"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.provider == "anthropic"

    def test_change_api_key(self, tmp_path: Path) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        meta = TrialMeta()
        config = ResearchClawConfig()
        chat = FakeChatInterface(responses=[
            UserMessage("set api_key sk-test-key-123"),
            UserMessage("back"),
        ])
        handle_settings(trial_dir, meta, config, chat)
        assert config.api_key == "sk-test-key-123"
