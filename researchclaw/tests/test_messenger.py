"""Tests for messenger module — QueueMessenger, get_messenger factory, TelegramMessenger."""
from __future__ import annotations

import pytest

from researchclaw.messenger import QueueMessenger, StdIOMessenger, get_messenger


def test_queue_messenger_send_receive() -> None:
    """QueueMessenger.send stores messages; push/receive round-trips."""
    m = QueueMessenger()
    m.send("hello")
    assert m.sent == ["hello"]

    m.push("reply")
    assert m.receive() == "reply"


def test_queue_messenger_receive_empty() -> None:
    """QueueMessenger.receive returns None when inbox is empty."""
    m = QueueMessenger()
    assert m.receive(timeout=0.01) is None


def test_queue_messenger_confirm_yes() -> None:
    """QueueMessenger.confirm returns True for 'yes'."""
    m = QueueMessenger()
    m.push("yes")
    assert m.confirm("Continue?") is True
    assert any("[Y/N]" in s for s in m.sent)


def test_queue_messenger_confirm_no() -> None:
    """QueueMessenger.confirm returns False for 'no'."""
    m = QueueMessenger()
    m.push("no")
    assert m.confirm("Continue?") is False


def test_queue_messenger_confirm_empty() -> None:
    """QueueMessenger.confirm returns False when no reply."""
    m = QueueMessenger()
    assert m.confirm("Continue?") is False


def test_get_messenger_stdio() -> None:
    """get_messenger('stdio') returns StdIOMessenger."""
    m = get_messenger("stdio")
    assert isinstance(m, StdIOMessenger)


def test_get_messenger_invalid() -> None:
    """get_messenger with unknown type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported messenger type"):
        get_messenger("unknown")


def test_get_messenger_telegram_missing_config() -> None:
    """get_messenger('telegram') without bot_token/chat_id raises ValueError."""
    with pytest.raises(ValueError, match="requires bot_token"):
        get_messenger("telegram")

    with pytest.raises(ValueError, match="requires bot_token"):
        get_messenger("telegram", bot_token="tok", chat_id="")


def test_telegram_messenger_import_error() -> None:
    """TelegramMessenger raises ImportError if python-telegram-bot is not installed."""
    # Only test if telegram is NOT installed; skip otherwise
    try:
        import telegram  # noqa: F401
        pytest.skip("python-telegram-bot is installed; cannot test ImportError")
    except ImportError:
        pass

    from researchclaw.messenger import TelegramMessenger

    with pytest.raises(ImportError, match="python-telegram-bot"):
        TelegramMessenger(bot_token="fake", chat_id="123")


def test_queue_messenger_multiple_messages() -> None:
    """QueueMessenger handles multiple messages in order."""
    m = QueueMessenger()
    m.push("first")
    m.push("second")
    m.push("third")

    assert m.receive() == "first"
    assert m.receive() == "second"
    assert m.receive() == "third"
    assert m.receive(timeout=0.01) is None


def test_queue_messenger_sent_accumulates() -> None:
    """QueueMessenger.sent accumulates all sent messages."""
    m = QueueMessenger()
    m.send("a")
    m.send("b")
    m.send("c")
    assert m.sent == ["a", "b", "c"]
