"""Tests for researchclaw/messenger.py"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from researchclaw.messenger import MessengerStdio, get_messenger


# ──────────────────────────────────────────────────────────────────────────────
# TestMessengerStdio
# ──────────────────────────────────────────────────────────────────────────────

class TestMessengerStdio:
    @pytest.fixture
    def m(self):
        return MessengerStdio()

    def test_send_prints_researchclaw_prefix(self, m, capsys):
        m.send("hello world")
        out = capsys.readouterr().out
        assert "[ResearchClaw]" in out

    def test_send_includes_message_text(self, m, capsys):
        m.send("hello world")
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_receive_returns_stripped_line(self, m, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("hello\n"))
        result = m.receive()
        assert result == "hello"

    def test_receive_returns_none_on_eof(self, m, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        result = m.receive()
        assert result is None

    def test_receive_returns_none_on_eoferror(self, m, monkeypatch):
        class _StdinThatRaises:
            def readline(self):
                raise EOFError

        monkeypatch.setattr("sys.stdin", _StdinThatRaises())
        assert m.receive() is None

    def test_confirm_returns_true_for_y(self, m, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
        assert m.confirm("Merge?") is True

    def test_confirm_returns_true_for_yes(self, m, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
        assert m.confirm("Merge?") is True

    def test_confirm_returns_true_for_uppercase_Y(self, m, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("Y\n"))
        assert m.confirm("Merge?") is True

    def test_confirm_returns_false_for_n(self, m, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
        assert m.confirm("Merge?") is False

    def test_confirm_returns_false_on_timeout(self, m, monkeypatch, capsys):
        # EOF simulates timeout — receive() returns None → confirm() returns False
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert m.confirm("Merge?") is False

    def test_confirm_sends_yn_prompt(self, m, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
        m.confirm("Approve trial?")
        out = capsys.readouterr().out
        assert "Approve trial?" in out
        assert "[Y/N]" in out

    def test_send_code_block_includes_triple_backtick(self, m, capsys):
        m.send_code_block("x = 1", language="python")
        out = capsys.readouterr().out
        assert "```" in out

    def test_send_code_block_includes_language_tag(self, m, capsys):
        m.send_code_block("x = 1", language="python")
        out = capsys.readouterr().out
        assert "python" in out

    def test_send_code_block_includes_code_content(self, m, capsys):
        m.send_code_block("x = 42", language="python")
        out = capsys.readouterr().out
        assert "x = 42" in out

    def test_send_diff_includes_diff_language_tag(self, m, capsys):
        m.send_diff("+ added line\n- removed line")
        out = capsys.readouterr().out
        assert "diff" in out
        assert "```" in out

    def test_send_diff_includes_diff_content(self, m, capsys):
        m.send_diff("+ added line\n- removed line")
        out = capsys.readouterr().out
        assert "added line" in out


# ──────────────────────────────────────────────────────────────────────────────
# TestGetMessengerFactory
# ──────────────────────────────────────────────────────────────────────────────

class TestGetMessengerFactory:
    def _make_config(self, messenger_type: str, **kwargs):
        cfg = MagicMock()
        cfg.messenger_type = messenger_type
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    def test_stdio_returns_messenger_stdio(self):
        config = self._make_config("stdio")
        m = get_messenger(config)
        assert isinstance(m, MessengerStdio)

    def test_stdio_case_insensitive(self):
        config = self._make_config("STDIO")
        m = get_messenger(config)
        assert isinstance(m, MessengerStdio)

    def test_unknown_type_raises_value_error(self):
        config = self._make_config("carrier_pigeon")
        with pytest.raises(ValueError, match="Unknown messenger type"):
            get_messenger(config)

    def test_telegram_without_token_raises_environment_error(self, monkeypatch):
        monkeypatch.delenv("RESEARCHCLAW_TEST_BOT_TOKEN", raising=False)
        config = self._make_config(
            "telegram",
            telegram_chat_id="12345",
            telegram_bot_token_env="RESEARCHCLAW_TEST_BOT_TOKEN",
            telegram_poll_timeout=30,
            telegram_poll_interval=1.0,
        )
        with pytest.raises(EnvironmentError):
            get_messenger(config)

    def test_telegram_with_empty_chat_id_raises_value_error(self, monkeypatch):
        # Token is present but chat_id is empty → ValueError before telebot is imported
        monkeypatch.setenv("RESEARCHCLAW_TEST_BOT_TOKEN", "fake-token-abc123")
        config = self._make_config(
            "telegram",
            telegram_chat_id="",
            telegram_bot_token_env="RESEARCHCLAW_TEST_BOT_TOKEN",
            telegram_poll_timeout=30,
            telegram_poll_interval=1.0,
        )
        with pytest.raises(ValueError):
            get_messenger(config)
