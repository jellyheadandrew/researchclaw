from __future__ import annotations

import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import ResearchClawConfig
from researchclaw.llm.provider import (
    AnthropicProvider,
    ClaudeAgentSDKProvider,
    ClaudeCLIProvider,
    LLMProvider,
    OpenAIProvider,
    get_provider,
)
from researchclaw.llm.installer import ensure_package


# ---- LLMProvider abstraction ----


class TestLLMProviderABC:
    """Test that LLMProvider cannot be instantiated directly."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]


# ---- ClaudeCLIProvider instantiation ----


class TestClaudeCLIProvider:
    """Test ClaudeCLIProvider construction and prompt building."""

    def test_default_model(self) -> None:
        provider = ClaudeCLIProvider()
        assert provider.model == "claude-opus-4-6"

    def test_custom_model(self) -> None:
        provider = ClaudeCLIProvider(model="claude-sonnet-4-5-20250929")
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_build_prompt_simple(self) -> None:
        provider = ClaudeCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        prompt = provider._build_prompt(messages)
        assert "[User]\nHello" in prompt

    def test_build_prompt_with_system(self) -> None:
        provider = ClaudeCLIProvider()
        messages = [{"role": "user", "content": "Hi"}]
        prompt = provider._build_prompt(messages, system="You are helpful.")
        assert "[System]\nYou are helpful." in prompt
        assert "[User]\nHi" in prompt

    def test_build_prompt_multi_turn(self) -> None:
        provider = ClaudeCLIProvider()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = provider._build_prompt(messages)
        assert "[User]\nHello" in prompt
        assert "[Assistant]\nHi there!" in prompt
        assert "[User]\nHow are you?" in prompt

    def test_chat_success(self) -> None:
        provider = ClaudeCLIProvider()
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="I'm fine!\n", stderr=""
        )
        with patch("researchclaw.llm.provider.subprocess.run", return_value=mock_result) as mock_run:
            result = provider.chat([{"role": "user", "content": "How are you?"}])
            assert result == "I'm fine!"
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == ["claude", "-p", "--model", "claude-opus-4-6"]

    def test_chat_with_system(self) -> None:
        provider = ClaudeCLIProvider()
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Response", stderr=""
        )
        with patch("researchclaw.llm.provider.subprocess.run", return_value=mock_result) as mock_run:
            provider.chat(
                [{"role": "user", "content": "test"}],
                system="Be helpful",
            )
            call_args = mock_run.call_args
            assert "[System]\nBe helpful" in call_args[1]["input"]

    def test_chat_cli_not_found(self) -> None:
        provider = ClaudeCLIProvider()
        with patch(
            "researchclaw.llm.provider.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            with pytest.raises(RuntimeError, match="Claude CLI not found"):
                provider.chat([{"role": "user", "content": "test"}])

    def test_chat_cli_timeout(self) -> None:
        provider = ClaudeCLIProvider()
        with patch(
            "researchclaw.llm.provider.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=120),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                provider.chat([{"role": "user", "content": "test"}])

    def test_chat_nonzero_exit(self) -> None:
        provider = ClaudeCLIProvider()
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="API error"
        )
        with patch("researchclaw.llm.provider.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="exit 1.*API error"):
                provider.chat([{"role": "user", "content": "test"}])

    def test_chat_stream_success(self) -> None:
        provider = ClaudeCLIProvider()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = iter(["line1\n", "line2\n"])
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("researchclaw.llm.provider.subprocess.Popen", return_value=mock_proc):
            chunks = list(provider.chat_stream([{"role": "user", "content": "test"}]))
            assert chunks == ["line1\n", "line2\n"]
            mock_proc.stdin.write.assert_called_once()
            mock_proc.stdin.close.assert_called_once()

    def test_chat_stream_cli_not_found(self) -> None:
        provider = ClaudeCLIProvider()
        with patch(
            "researchclaw.llm.provider.subprocess.Popen",
            side_effect=FileNotFoundError,
        ):
            with pytest.raises(RuntimeError, match="Claude CLI not found"):
                list(provider.chat_stream([{"role": "user", "content": "test"}]))

    def test_chat_stream_nonzero_exit(self) -> None:
        provider = ClaudeCLIProvider()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = "stream error"
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1

        with patch("researchclaw.llm.provider.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="exit 1.*stream error"):
                list(provider.chat_stream([{"role": "user", "content": "test"}]))


# ---- ClaudeAgentSDKProvider (Tier 1) ----


class TestClaudeAgentSDKProvider:
    """Test Tier 1 provider with lazy claude_agent_sdk import."""

    def test_instantiation(self) -> None:
        provider = ClaudeAgentSDKProvider()
        assert provider.model == "claude-opus-4-6"
        assert provider._sdk is None

    def test_custom_model(self) -> None:
        provider = ClaudeAgentSDKProvider(model="claude-sonnet-4-5-20250929")
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_lazy_import_error(self) -> None:
        provider = ClaudeAgentSDKProvider()
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            with pytest.raises(RuntimeError, match="claude-agent-sdk is not installed"):
                provider.chat([{"role": "user", "content": "test"}])

    def test_chat_success(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.query.return_value = "SDK response"
        provider = ClaudeAgentSDKProvider()
        provider._sdk = mock_sdk
        result = provider.chat(
            [{"role": "user", "content": "Hello"}],
            system="Be helpful",
        )
        assert result == "SDK response"
        mock_sdk.query.assert_called_once_with(
            prompt="Hello",
            system="Be helpful",
            model="claude-opus-4-6",
        )

    def test_chat_empty_messages(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.query.return_value = ""
        provider = ClaudeAgentSDKProvider()
        provider._sdk = mock_sdk
        result = provider.chat([], system="sys")
        assert result == ""
        mock_sdk.query.assert_called_once_with(
            prompt="",
            system="sys",
            model="claude-opus-4-6",
        )

    def test_chat_stream_yields_single_result(self) -> None:
        mock_sdk = MagicMock()
        mock_sdk.query.return_value = "streamed"
        provider = ClaudeAgentSDKProvider()
        provider._sdk = mock_sdk
        chunks = list(provider.chat_stream([{"role": "user", "content": "test"}]))
        assert chunks == ["streamed"]

    def test_get_sdk_caches(self) -> None:
        fake_module = MagicMock()
        provider = ClaudeAgentSDKProvider()
        with patch.dict(sys.modules, {"claude_agent_sdk": fake_module}):
            sdk1 = provider._get_sdk()
            sdk2 = provider._get_sdk()
            assert sdk1 is sdk2


# ---- AnthropicProvider (Tier 2) ----


class TestAnthropicProvider:
    """Test Tier 2 Anthropic provider with lazy import."""

    def test_instantiation(self) -> None:
        provider = AnthropicProvider(api_key="sk-test")
        assert provider.api_key == "sk-test"
        assert provider.model == "claude-opus-4-6"
        assert provider._client is None

    def test_custom_model(self) -> None:
        provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-5-20250929")
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_lazy_import_error(self) -> None:
        provider = AnthropicProvider(api_key="sk-test")
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(RuntimeError, match="anthropic SDK is not installed"):
                provider.chat([{"role": "user", "content": "test"}])

    def test_chat_success(self) -> None:
        mock_block = MagicMock()
        mock_block.text = "Anthropic response"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="sk-test")
        provider._client = mock_client

        result = provider.chat(
            [{"role": "user", "content": "Hello"}],
            system="Be helpful",
        )
        assert result == "Anthropic response"
        mock_client.messages.create.assert_called_once_with(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": "Hello"}],
            system="Be helpful",
        )

    def test_chat_without_system(self) -> None:
        mock_block = MagicMock()
        mock_block.text = "Response"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="sk-test")
        provider._client = mock_client

        provider.chat([{"role": "user", "content": "Hello"}])
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_chat_multiple_content_blocks(self) -> None:
        block1 = MagicMock()
        block1.text = "Part 1"
        block2 = MagicMock()
        block2.text = "Part 2"
        mock_response = MagicMock()
        mock_response.content = [block1, block2]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="sk-test")
        provider._client = mock_client

        result = provider.chat([{"role": "user", "content": "test"}])
        assert result == "Part 1\nPart 2"

    def test_chat_stream_success(self) -> None:
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["chunk1", "chunk2"])
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        provider = AnthropicProvider(api_key="sk-test")
        provider._client = mock_client

        chunks = list(provider.chat_stream(
            [{"role": "user", "content": "test"}],
            system="sys",
        ))
        assert chunks == ["chunk1", "chunk2"]

    def test_get_client_caches(self) -> None:
        mock_anthropic_mod = MagicMock()
        provider = AnthropicProvider(api_key="sk-test")
        with patch.dict(sys.modules, {"anthropic": mock_anthropic_mod}):
            client1 = provider._get_client()
            client2 = provider._get_client()
            assert client1 is client2


# ---- OpenAIProvider (Tier 2) ----


class TestOpenAIProvider:
    """Test Tier 2 OpenAI provider with lazy import."""

    def test_instantiation(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        assert provider.api_key == "sk-test"
        assert provider.model == "gpt-4"
        assert provider._client is None

    def test_custom_model(self) -> None:
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        assert provider.model == "gpt-4o"

    def test_lazy_import_error(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(RuntimeError, match="openai SDK is not installed"):
                provider.chat([{"role": "user", "content": "test"}])

    def test_chat_success(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="sk-test")
        provider._client = mock_client

        result = provider.chat(
            [{"role": "user", "content": "Hello"}],
            system="Be helpful",
        )
        assert result == "OpenAI response"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        # System message should be prepended
        assert call_kwargs["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Hello"}

    def test_chat_without_system(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="sk-test")
        provider._client = mock_client

        provider.chat([{"role": "user", "content": "Hello"}])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # No system message prepended
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_chat_none_content(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="sk-test")
        provider._client = mock_client

        result = provider.chat([{"role": "user", "content": "test"}])
        assert result == ""

    def test_chat_stream_success(self) -> None:
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "chunk1"
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "chunk2"
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None  # Final chunk with no content

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        provider = OpenAIProvider(api_key="sk-test")
        provider._client = mock_client

        chunks = list(provider.chat_stream(
            [{"role": "user", "content": "test"}],
            system="sys",
        ))
        assert chunks == ["chunk1", "chunk2"]

    def test_get_client_caches(self) -> None:
        mock_openai_mod = MagicMock()
        provider = OpenAIProvider(api_key="sk-test")
        with patch.dict(sys.modules, {"openai": mock_openai_mod}):
            client1 = provider._get_client()
            client2 = provider._get_client()
            assert client1 is client2


# ---- get_provider factory ----


class TestGetProvider:
    """Test the provider factory function."""

    def test_default_returns_claude_cli(self) -> None:
        config = ResearchClawConfig()
        provider = get_provider(config)
        assert isinstance(provider, ClaudeCLIProvider)

    def test_uses_model_from_config(self) -> None:
        config = ResearchClawConfig(model="claude-sonnet-4-5-20250929")
        provider = get_provider(config)
        assert isinstance(provider, ClaudeCLIProvider)
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_explicit_claude_cli(self) -> None:
        config = ResearchClawConfig(provider="claude_cli")
        provider = get_provider(config)
        assert isinstance(provider, ClaudeCLIProvider)

    def test_claude_agent_sdk_provider(self) -> None:
        config = ResearchClawConfig(provider="claude_agent_sdk")
        provider = get_provider(config)
        assert isinstance(provider, ClaudeAgentSDKProvider)
        assert provider.model == "claude-opus-4-6"

    def test_anthropic_provider(self) -> None:
        config = ResearchClawConfig(provider="anthropic", api_key="sk-ant-test")
        provider = get_provider(config)
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "sk-ant-test"
        assert provider.model == "claude-opus-4-6"

    def test_openai_provider(self) -> None:
        config = ResearchClawConfig(provider="openai", api_key="sk-oai-test")
        provider = get_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "sk-oai-test"

    def test_anthropic_missing_api_key(self) -> None:
        config = ResearchClawConfig(provider="anthropic", api_key="")
        with pytest.raises(ValueError, match="API key"):
            get_provider(config)

    def test_openai_missing_api_key(self) -> None:
        config = ResearchClawConfig(provider="openai", api_key="")
        with pytest.raises(ValueError, match="API key"):
            get_provider(config)

    def test_unknown_provider_falls_back_to_cli(self) -> None:
        config = ResearchClawConfig(provider="unknown_provider")
        provider = get_provider(config)
        assert isinstance(provider, ClaudeCLIProvider)


# ---- ensure_package ----


class TestEnsurePackage:
    """Test runtime SDK installation logic."""

    def test_already_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import researchclaw.llm.installer as installer_mod

        mock_import = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(installer_mod.importlib, "import_module", mock_import)

        result = ensure_package("json")
        assert result is True
        mock_import.assert_called_once_with("json")

    def test_install_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import researchclaw.llm.installer as installer_mod

        call_count = 0

        def mock_import(name: str) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ImportError("not installed")
            return MagicMock()

        mock_run = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
        )
        monkeypatch.setattr(installer_mod.importlib, "import_module", mock_import)
        monkeypatch.setattr(installer_mod.subprocess, "run", mock_run)

        result = ensure_package("some_pkg", "some-pkg")
        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "pip" in call_args
        assert "some-pkg" in call_args

    def test_install_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import researchclaw.llm.installer as installer_mod

        monkeypatch.setattr(
            installer_mod.importlib,
            "import_module",
            MagicMock(side_effect=ImportError("not installed")),
        )
        monkeypatch.setattr(
            installer_mod.subprocess,
            "run",
            MagicMock(side_effect=subprocess.CalledProcessError(1, "pip")),
        )

        result = ensure_package("nonexistent_pkg")
        assert result is False

    def test_pip_name_defaults_to_package_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import researchclaw.llm.installer as installer_mod

        monkeypatch.setattr(
            installer_mod.importlib,
            "import_module",
            MagicMock(side_effect=ImportError("not installed")),
        )
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "pip"))
        monkeypatch.setattr(installer_mod.subprocess, "run", mock_run)

        ensure_package("my_package")
        call_args = mock_run.call_args[0][0]
        assert "my_package" in call_args

    def test_install_succeeds_but_import_still_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import researchclaw.llm.installer as installer_mod

        monkeypatch.setattr(
            installer_mod.importlib,
            "import_module",
            MagicMock(side_effect=ImportError("still broken")),
        )
        monkeypatch.setattr(
            installer_mod.subprocess,
            "run",
            MagicMock(
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                )
            ),
        )

        result = ensure_package("broken_pkg")
        assert result is False
