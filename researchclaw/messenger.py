from __future__ import annotations

import queue
import sys
from abc import ABC, abstractmethod


class Messenger(ABC):
    @abstractmethod
    def send(self, text: str) -> None:
        ...

    @abstractmethod
    def receive(self, timeout: float | None = None) -> str | None:
        ...

    def confirm(self, prompt: str) -> bool:
        self.send(f"{prompt} [Y/N]")
        reply = self.receive(timeout=None)
        if reply is None:
            return False
        return reply.strip().lower() in {"y", "yes"}


class StdIOMessenger(Messenger):
    def send(self, text: str) -> None:
        print(text)
        sys.stdout.flush()

    def receive(self, timeout: float | None = None) -> str | None:
        try:
            return input().strip()
        except EOFError:
            return None


class QueueMessenger(Messenger):
    """In-memory messenger for tests."""

    def __init__(self) -> None:
        self.inbox: queue.Queue[str] = queue.Queue()
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        self.sent.append(text)

    def receive(self, timeout: float | None = None) -> str | None:
        try:
            if timeout is None:
                return self.inbox.get_nowait()
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def push(self, text: str) -> None:
        self.inbox.put(text)


def get_messenger(kind: str) -> Messenger:
    if kind == "stdio":
        return StdIOMessenger()
    raise ValueError(f"Unsupported messenger type for v2: {kind!r}")
