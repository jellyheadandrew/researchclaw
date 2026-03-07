"""Shared test fixtures and utilities for ResearchClaw tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from researchclaw.repl import ChatInput


class FakeChat:
    """Fake chat interface with pre-programmed responses."""

    def __init__(self, responses: list[ChatInput] | None = None) -> None:
        self.sent: list[str] = []
        self._responses = list(responses) if responses else []

    # Backward-compat alias: some test files reference .messages instead of .sent
    @property
    def messages(self) -> list[str]:
        return self.sent

    def send(self, message: str) -> None:
        self.sent.append(message)

    def send_status(self, message: str) -> None:
        self.sent.append(message)

    def send_stream(self, chunks: Any) -> str:
        accumulated = "".join(chunks)
        self.sent.append(accumulated)
        return accumulated

    @contextmanager
    def show_thinking(self) -> Iterator[None]:
        yield

    def send_thinking(self, thinking: str) -> None:
        pass

    def send_image(self, path: str, caption: str | None = None) -> None:
        pass

    def receive(self) -> ChatInput:
        if not self._responses:
            raise SystemExit("No more responses")
        return self._responses.pop(0)


# Backward-compat alias for test files that use FakeChatInterface as the name
FakeChatInterface = FakeChat
