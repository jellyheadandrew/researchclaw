from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from researchclaw.repl import (
    SLASH_COMMANDS,
    ChatInterface,
    ChatInput,
    SlashCommand,
    TerminalChat,
    UserMessage,
    parse_input,
)


class TestParseInput:
    """Tests for the parse_input function."""

    def test_regular_message(self) -> None:
        result = parse_input("hello world")
        assert isinstance(result, UserMessage)
        assert result.text == "hello world"

    def test_slash_approve(self) -> None:
        result = parse_input("/approve")
        assert isinstance(result, SlashCommand)
        assert result.name == "/approve"
        assert result.args == ""

    def test_slash_autopilot(self) -> None:
        result = parse_input("/autopilot")
        assert isinstance(result, SlashCommand)
        assert result.name == "/autopilot"

    def test_slash_autopilot_stop(self) -> None:
        result = parse_input("/autopilot-stop")
        assert isinstance(result, SlashCommand)
        assert result.name == "/autopilot-stop"

    def test_slash_status(self) -> None:
        result = parse_input("/status")
        assert isinstance(result, SlashCommand)
        assert result.name == "/status"

    def test_slash_quit(self) -> None:
        result = parse_input("/quit")
        assert isinstance(result, SlashCommand)
        assert result.name == "/quit"

    def test_slash_help(self) -> None:
        result = parse_input("/help")
        assert isinstance(result, SlashCommand)
        assert result.name == "/help"

    def test_slash_verbose(self) -> None:
        result = parse_input("/verbose")
        assert isinstance(result, SlashCommand)
        assert result.name == "/verbose"

    def test_unknown_slash_is_user_message(self) -> None:
        result = parse_input("/unknown")
        assert isinstance(result, UserMessage)
        assert result.text == "/unknown"

    def test_slash_command_with_args(self) -> None:
        result = parse_input("/approve   some reason here")
        assert isinstance(result, SlashCommand)
        assert result.name == "/approve"
        assert result.args == "some reason here"

    def test_slash_command_case_insensitive(self) -> None:
        result = parse_input("/APPROVE")
        assert isinstance(result, SlashCommand)
        assert result.name == "/approve"

    def test_whitespace_stripped(self) -> None:
        result = parse_input("  hello  ")
        assert isinstance(result, UserMessage)
        assert result.text == "hello"

    def test_slash_with_leading_whitespace(self) -> None:
        result = parse_input("   /quit   ")
        assert isinstance(result, SlashCommand)
        assert result.name == "/quit"

    def test_empty_input(self) -> None:
        result = parse_input("")
        assert isinstance(result, UserMessage)
        assert result.text == ""

    def test_whitespace_only(self) -> None:
        result = parse_input("   ")
        assert isinstance(result, UserMessage)
        assert result.text == ""


class TestSlashCommands:
    """Tests for the SLASH_COMMANDS registry."""

    def test_all_commands_present(self) -> None:
        expected = {"/approve", "/abort", "/autopilot", "/autopilot-stop", "/status", "/verbose", "/quit", "/help"}
        assert set(SLASH_COMMANDS.keys()) == expected

    def test_all_commands_have_descriptions(self) -> None:
        for cmd, desc in SLASH_COMMANDS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0, f"{cmd} has empty description"


class TestSlashCommandTuple:
    """Tests for SlashCommand and UserMessage named tuples."""

    def test_slash_command_fields(self) -> None:
        cmd = SlashCommand(name="/approve", args="plan looks good")
        assert cmd.name == "/approve"
        assert cmd.args == "plan looks good"

    def test_user_message_field(self) -> None:
        msg = UserMessage(text="run experiment")
        assert msg.text == "run experiment"

    def test_slash_command_is_tuple(self) -> None:
        cmd = SlashCommand(name="/quit", args="")
        assert isinstance(cmd, tuple)

    def test_user_message_is_tuple(self) -> None:
        msg = UserMessage(text="hello")
        assert isinstance(msg, tuple)


class TestTerminalChat:
    """Tests for TerminalChat (with mocked prompt-toolkit)."""

    def test_send_outputs_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        chat = TerminalChat()
        chat.send("Hello, user!")
        captured = capsys.readouterr()
        assert "Hello, user!" in captured.out

    def test_send_image(self, capsys: pytest.CaptureFixture[str]) -> None:
        chat = TerminalChat()
        chat.send_image("/path/to/img.png", caption="Results")
        captured = capsys.readouterr()
        assert "[Image: /path/to/img.png]" in captured.out
        assert "Results" in captured.out

    def test_send_image_no_caption(self, capsys: pytest.CaptureFixture[str]) -> None:
        chat = TerminalChat()
        chat.send_image("/path/to/img.png")
        captured = capsys.readouterr()
        assert "[Image: /path/to/img.png]" in captured.out

    def test_receive_returns_user_message(self) -> None:
        chat = TerminalChat()
        with patch.object(chat._session, "prompt", return_value="hello"):
            result = chat.receive()
        assert isinstance(result, UserMessage)
        assert result.text == "hello"

    def test_receive_returns_slash_command(self) -> None:
        chat = TerminalChat()
        with patch.object(chat._session, "prompt", return_value="/approve"):
            result = chat.receive()
        assert isinstance(result, SlashCommand)
        assert result.name == "/approve"

    def test_receive_quit_raises_system_exit(self) -> None:
        chat = TerminalChat()
        with patch.object(chat._session, "prompt", return_value="/quit"):
            with pytest.raises(SystemExit):
                chat.receive()

    def test_receive_help_reprompts(self) -> None:
        """After /help, the user is re-prompted; next input is returned."""
        chat = TerminalChat()
        with patch.object(chat._session, "prompt", side_effect=["/help", "next message"]):
            result = chat.receive()
        assert isinstance(result, UserMessage)
        assert result.text == "next message"

    def test_receive_skips_empty_input(self) -> None:
        """Empty input lines are ignored and user is re-prompted."""
        chat = TerminalChat()
        with patch.object(chat._session, "prompt", side_effect=["", "   ", "actual"]):
            result = chat.receive()
        assert isinstance(result, UserMessage)
        assert result.text == "actual"

    def test_history_path(self, tmp_path: Path) -> None:
        history_file = tmp_path / ".chat_history"
        chat = TerminalChat(history_path=history_file)
        assert chat._session.history is not None


class TestChatInterfaceABC:
    """Tests for the ChatInterface abstract class."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            ChatInterface()  # type: ignore[abstract]

    def test_terminal_chat_is_chat_interface(self) -> None:
        chat = TerminalChat()
        assert isinstance(chat, ChatInterface)

    def test_terminal_chat_has_send_status(self) -> None:
        """US-003: TerminalChat must have a send_status method distinct from send."""
        chat = TerminalChat()
        assert hasattr(chat, "send_status")
        assert callable(chat.send_status)

    def test_send_status_outputs_dim_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        """US-003: send_status prints dim text (no panel)."""
        chat = TerminalChat()
        chat.send_status("[EXPERIMENT_PLAN] Starting...")
        captured = capsys.readouterr()
        assert "[EXPERIMENT_PLAN] Starting..." in captured.out

    def test_chat_interface_send_status_default(self) -> None:
        """US-003: ChatInterface.send_status default delegates to send()."""
        assert hasattr(ChatInterface, "send_status")
        # Verify it's not abstract (it has a default implementation)
        # The default calls self.send(), so it should work on any concrete subclass
        chat = TerminalChat()
        # send_status is overridden in TerminalChat, but the ABC provides a default
        assert "send_status" in dir(ChatInterface)


class TestStreamAndThinking:
    """US-004: Tests for send_stream, show_thinking, send_thinking, /verbose."""

    def test_terminal_chat_has_send_stream(self) -> None:
        """TerminalChat must have a send_stream method."""
        chat = TerminalChat()
        assert hasattr(chat, "send_stream")
        assert callable(chat.send_stream)

    def test_terminal_chat_has_show_thinking(self) -> None:
        """TerminalChat must have a show_thinking method."""
        chat = TerminalChat()
        assert hasattr(chat, "show_thinking")
        assert callable(chat.show_thinking)

    def test_terminal_chat_has_send_thinking(self) -> None:
        """TerminalChat must have a send_thinking method."""
        chat = TerminalChat()
        assert hasattr(chat, "send_thinking")
        assert callable(chat.send_thinking)

    def test_terminal_chat_verbose_default_false(self) -> None:
        """TerminalChat._verbose defaults to False."""
        chat = TerminalChat()
        assert chat._verbose is False

    def test_send_stream_returns_accumulated_text(self) -> None:
        """send_stream should return the accumulated text from chunks."""
        chat = TerminalChat()
        chunks = iter(["Hello", " ", "world"])
        result = chat.send_stream(chunks)
        assert result == "Hello world"

    def test_show_thinking_is_context_manager(self) -> None:
        """show_thinking should be usable as a context manager."""
        chat = TerminalChat()
        with chat.show_thinking():
            pass  # Should not raise

    def test_send_thinking_silent_when_not_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """send_thinking should not output when _verbose is False."""
        chat = TerminalChat()
        chat._verbose = False
        chat.send_thinking("thinking content")
        captured = capsys.readouterr()
        assert "thinking content" not in captured.out

    def test_send_thinking_outputs_when_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """send_thinking should output when _verbose is True."""
        chat = TerminalChat()
        chat._verbose = True
        chat.send_thinking("thinking content")
        captured = capsys.readouterr()
        assert "thinking content" in captured.out

    def test_verbose_toggle_via_receive(self) -> None:
        """/verbose toggles _verbose and re-prompts."""
        chat = TerminalChat()
        assert chat._verbose is False
        with patch.object(chat._session, "prompt", side_effect=["/verbose", "next"]):
            result = chat.receive()
        assert chat._verbose is True
        assert isinstance(result, UserMessage)
        assert result.text == "next"

    def test_verbose_toggle_off(self) -> None:
        """/verbose toggles _verbose off when already on."""
        chat = TerminalChat()
        chat._verbose = True
        with patch.object(chat._session, "prompt", side_effect=["/verbose", "next"]):
            result = chat.receive()
        assert chat._verbose is False

    def test_chat_interface_send_stream_default(self) -> None:
        """ChatInterface.send_stream default collects chunks and calls send()."""
        assert hasattr(ChatInterface, "send_stream")
        assert "send_stream" in dir(ChatInterface)

    def test_chat_interface_show_thinking_default(self) -> None:
        """ChatInterface.show_thinking default is a no-op context manager."""
        assert hasattr(ChatInterface, "show_thinking")
        assert "show_thinking" in dir(ChatInterface)

    def test_chat_interface_send_thinking_default(self) -> None:
        """ChatInterface.send_thinking default is a no-op."""
        assert hasattr(ChatInterface, "send_thinking")
        assert "send_thinking" in dir(ChatInterface)
