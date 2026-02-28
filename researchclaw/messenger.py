from __future__ import annotations

import queue
import sys
import threading
import time
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


class TelegramMessenger(Messenger):
    """Telegram Bot messenger using python-telegram-bot."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        try:
            import telegram
            import telegram.ext
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required for Telegram messenger. "
                "Install with: pip install 'researchclaw[telegram]'"
            )

        self.bot_token = bot_token
        self.chat_id = int(chat_id)
        self._bot = telegram.Bot(token=bot_token)
        self._inbox: queue.Queue[str] = queue.Queue()
        self._last_update_id = 0
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_polling()

    def _start_polling(self) -> None:
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        import telegram

        while not self._stop_event.is_set():
            try:
                import asyncio

                loop = asyncio.new_event_loop()
                updates = loop.run_until_complete(
                    self._bot.get_updates(
                        offset=self._last_update_id + 1,
                        timeout=10,
                    )
                )
                loop.close()

                for update in updates:
                    self._last_update_id = update.update_id
                    if (
                        update.message
                        and update.message.text
                        and update.message.chat_id == self.chat_id
                    ):
                        self._inbox.put(update.message.text.strip())
            except Exception:
                time.sleep(2)

    def send(self, text: str) -> None:
        import asyncio

        # Telegram has a 4096 character limit per message
        chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
        loop = asyncio.new_event_loop()
        try:
            for chunk in chunks:
                loop.run_until_complete(
                    self._bot.send_message(chat_id=self.chat_id, text=chunk)
                )
        except Exception:
            pass
        finally:
            loop.close()

    def receive(self, timeout: float | None = None) -> str | None:
        try:
            if timeout is None:
                return self._inbox.get(block=True)
            return self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)


def get_messenger(kind: str, **kwargs: str) -> Messenger:
    if kind == "stdio":
        return StdIOMessenger()
    if kind == "telegram":
        bot_token = kwargs.get("bot_token", "")
        chat_id = kwargs.get("chat_id", "")
        if not bot_token or not chat_id:
            raise ValueError(
                "Telegram messenger requires bot_token and chat_id in config."
            )
        return TelegramMessenger(bot_token=bot_token, chat_id=chat_id)
    raise ValueError(f"Unsupported messenger type: {kind!r}")
