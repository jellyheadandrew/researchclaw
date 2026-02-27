"""
messenger.py — Communication layer between ResearchClaw and the researcher.

Supported transports:
  slack    — Native Slack Socket Mode (production, no public IP needed)
  telegram — Telegram Bot API via long-polling (production, VM-friendly)
  stdio    — stdin/stdout (local development and testing)

The agent uses the Messenger ABC as its sole communication interface, so
swapping transports requires only a config.yaml change (messenger.type).

Web UI / backend developers: the messenger type and its non-secret settings
live in config.yaml under the `messenger:` key.  Secrets (bot tokens) go in
.env.  See .env.example for the full list of expected env-var names.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
from abc import ABC, abstractmethod

logger = logging.getLogger("researchclaw.messenger")


# ──────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ──────────────────────────────────────────────────────────────────────────────

class Messenger(ABC):
    """Abstract messenger interface. Swap implementations to change transport."""

    @abstractmethod
    def send(self, text: str) -> None:
        """Send a plain-text message to the researcher."""
        ...

    @abstractmethod
    def receive(self, timeout: float | None = None) -> str | None:
        """
        Block until a message arrives (or timeout expires).
        Returns the message text, or None if timed out.
        """
        ...

    def send_code_block(self, code: str, language: str = "") -> None:
        """Send a formatted code block."""
        self.send(f"```{language}\n{code}\n```")

    def send_diff(self, diff: str) -> None:
        """Send a diff with syntax highlighting."""
        self.send_code_block(diff, language="diff")

    def confirm(self, prompt: str) -> bool:
        """
        Ask the researcher a yes/no question.
        Returns True if they respond with y/yes, False otherwise.
        """
        self.send(f"{prompt} [Y/N]")
        reply = self.receive(timeout=300)  # 5 minute timeout
        if reply is None:
            logger.warning("No response to confirmation prompt — defaulting to N")
            return False
        return reply.strip().lower() in ("y", "yes")


# ──────────────────────────────────────────────────────────────────────────────
# Slack via Socket Mode (native — no external dependencies like OpenClaw)
# ──────────────────────────────────────────────────────────────────────────────

class SlackMessenger(Messenger):
    """
    Sends and receives messages via Slack using Socket Mode.

    Socket Mode uses an outbound WebSocket connection — no public IP or
    port forwarding needed (same idea as Telegram long-polling).

    Setup (one-time):
      1. Create a Slack App at https://api.slack.com/apps
      2. Under "OAuth & Permissions", add Bot Token Scopes:
           chat:write, channels:history, channels:read
      3. Install the app to your workspace → copy Bot Token (xoxb-...)
      4. Under "Socket Mode", enable it → generate App-Level Token (xapp-...)
         with the connections:write scope
      5. Under "Event Subscriptions", subscribe to bot events:
           message.channels (and message.groups if using private channels)
      6. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in .env
      7. Set messenger.slack_channel in config.yaml
      8. Invite the bot to the target channel in Slack (/invite @botname)

    Security:
      - Only processes messages from the configured channel (others are dropped)
      - Bot's own messages are filtered out (echo prevention via bot user ID)
      - Message subtypes (edits, deletes, joins, etc.) are ignored
      - Tokens are never logged; only channel IDs appear in debug output

    Thread safety:
      The SocketModeClient runs its listener in a background thread.
      Incoming messages are queued. receive() blocks on the queue.
      send() is safe to call from any thread (WebClient is thread-safe).
    """

    _MAX_MSG_LEN = 3900  # Slack allows ~40k, but keep chunks readable

    def __init__(
        self,
        channel: str,
        bot_token_env: str = "SLACK_BOT_TOKEN",
        app_token_env: str = "SLACK_APP_TOKEN",
    ):
        bot_token = os.environ.get(bot_token_env)
        if not bot_token:
            raise EnvironmentError(
                f"Environment variable {bot_token_env!r} is not set.\n"
                "Create a Slack App, install it, then add the Bot Token to .env:\n"
                f"  {bot_token_env}=xoxb-..."
            )
        app_token = os.environ.get(app_token_env)
        if not app_token:
            raise EnvironmentError(
                f"Environment variable {app_token_env!r} is not set.\n"
                "Enable Socket Mode in your Slack App, generate an App-Level Token,\n"
                "then add it to .env:\n"
                f"  {app_token_env}=xapp-..."
            )

        try:
            from slack_sdk import WebClient
            from slack_sdk.socket_mode import SocketModeClient
        except ImportError:
            raise ImportError(
                "slack_sdk is not installed.  Run:\n"
                "  pip install 'slack_sdk[socket_mode]'"
            )

        self._web = WebClient(token=bot_token)
        self._queue: queue.Queue[str] = queue.Queue()

        # Resolve channel name → ID and discover bot's own user ID
        self._channel_id = self._resolve_channel(channel)
        self._bot_user_id = self._get_bot_user_id()

        # Start Socket Mode listener (runs in a background thread)
        self._socket = SocketModeClient(
            app_token=app_token,
            web_client=self._web,
        )
        self._socket.socket_mode_request_listeners.append(self._on_event)
        self._socket.connect()

        logger.info(
            "SlackMessenger ready (channel=%s, channel_id=%s)",
            channel, self._channel_id,
        )

    def _resolve_channel(self, channel: str) -> str:
        """Convert #channel-name to Slack channel ID."""
        name = channel.lstrip("#")
        try:
            cursor = None
            while True:
                kwargs: dict = {"types": "public_channel,private_channel", "limit": 200}
                if cursor:
                    kwargs["cursor"] = cursor
                resp = self._web.conversations_list(**kwargs)
                for ch in resp["channels"]:
                    if ch["name"] == name:
                        return ch["id"]
                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
        except Exception as exc:
            raise EnvironmentError(f"Failed to list Slack channels: {exc}")
        raise EnvironmentError(
            f"Slack channel '#{name}' not found. "
            "Make sure the channel exists and the bot has been invited to it."
        )

    def _get_bot_user_id(self) -> str:
        """Get the bot's own user ID for echo prevention."""
        try:
            resp = self._web.auth_test()
            return resp["user_id"]
        except Exception as exc:
            logger.warning("Could not determine bot user ID: %s", exc)
            return ""

    def _on_event(self, client, req) -> None:
        """Socket Mode event handler — filters and queues relevant messages."""
        from slack_sdk.socket_mode.response import SocketModeResponse

        # Acknowledge immediately to prevent Slack retries
        client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        if req.type != "events_api":
            return

        event = req.payload.get("event", {})

        # Only handle plain messages in our target channel
        if event.get("type") != "message":
            return
        if event.get("channel") != self._channel_id:
            return

        # Echo prevention: drop bot's own messages
        if self._bot_user_id and event.get("user") == self._bot_user_id:
            return

        # Drop message subtypes (edits, deletes, bot_message, joins, etc.)
        if event.get("subtype"):
            return

        text = event.get("text", "").strip()
        if text:
            logger.debug("← Slack [%s]: %s", self._channel_id, text[:80])
            self._queue.put(text)

    # ── Sending ──────────────────────────────────────────────────────────────

    def send(self, text: str) -> None:
        logger.debug("→ Slack [%s]: %s", self._channel_id, text[:80])
        for i in range(0, max(len(text), 1), self._MAX_MSG_LEN):
            chunk = text[i : i + self._MAX_MSG_LEN]
            try:
                self._web.chat_postMessage(channel=self._channel_id, text=chunk)
            except Exception as exc:
                logger.error("Slack send error: %s", exc)
                raise

    def send_code_block(self, code: str, language: str = "") -> None:
        """Send a code block. Truncates to stay under Slack's display limit."""
        truncated = code[:3800] + ("\n... (truncated)" if len(code) > 3800 else "")
        self.send(f"```\n{truncated}\n```")

    # ── Receiving ────────────────────────────────────────────────────────────

    def receive(self, timeout: float | None = None) -> str | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Telegram via Bot API (long-polling)
# ──────────────────────────────────────────────────────────────────────────────

class TelegramMessenger(Messenger):
    """
    Sends and receives messages via Telegram Bot API using long-polling.

    Preferred over webhooks because the VM typically has no public IP.

    Setup (one-time):
      1. Message @BotFather on Telegram: /newbot  → copy the bot token.
      2. Set TELEGRAM_BOT_TOKEN in your .env file (or environment).
      3. Send any message to the bot (or add it to a group).
      4. Find your chat_id:  python -m researchclaw.get_chat_id
      5. Set telegram.chat_id in config.yaml.

    Thread safety:
      receive() uses long-polling.  Only one thread should call receive() at
      a time (the main agent loop).  send() is safe to call from any thread.
    """

    # Telegram's hard limit per message is 4096 UTF-8 characters.
    _MAX_MSG_LEN = 4000  # leave a small buffer

    def __init__(
        self,
        chat_id: str,
        bot_token_env: str = "TELEGRAM_BOT_TOKEN",
        poll_timeout: int = 30,
        poll_interval: float = 1.0,
    ):
        token = os.environ.get(bot_token_env)
        if not token:
            raise EnvironmentError(
                f"Environment variable {bot_token_env!r} is not set.\n"
                "Create a Telegram bot via @BotFather, then add the token to .env:\n"
                f"  {bot_token_env}=<your-token>"
            )
        if not chat_id:
            raise ValueError(
                "telegram.chat_id is not set in config.yaml.\n"
                "Run `python -m researchclaw.get_chat_id` to find it."
            )

        try:
            import telebot  # pyTelegramBotAPI
        except ImportError:
            raise ImportError(
                "pyTelegramBotAPI is not installed.  Run:\n"
                "  pip install pyTelegramBotAPI"
            )

        self.chat_id = int(chat_id)
        self._poll_timeout = poll_timeout
        self._poll_interval = poll_interval
        self._bot = telebot.TeleBot(token, parse_mode=None)
        self._next_offset: int = 0  # tracks getUpdates offset; avoids re-delivery
        logger.info("TelegramMessenger ready (chat_id=%s)", self.chat_id)

    # ── Sending ──────────────────────────────────────────────────────────────

    def send(self, text: str) -> None:
        """Send a plain-text message.  Auto-splits if longer than Telegram's limit."""
        logger.debug("→ Telegram [%s]: %s", self.chat_id, text[:80])
        for i in range(0, max(len(text), 1), self._MAX_MSG_LEN):
            chunk = text[i:i + self._MAX_MSG_LEN]
            try:
                self._bot.send_message(self.chat_id, chunk)
            except Exception as exc:
                logger.error("Telegram send error: %s", exc)
                raise

    def send_code_block(self, code: str, language: str = "") -> None:
        """
        Send a code block.  Sent as plain text with triple-backtick fences
        to avoid MarkdownV2 escaping issues with arbitrary code content.
        Truncates at ~3800 chars to stay safely under Telegram's limit.
        """
        truncated = code[:3800] + ("\n... (truncated)" if len(code) > 3800 else "")
        self.send(f"```\n{truncated}\n```")

    # ── Receiving ────────────────────────────────────────────────────────────

    def receive(self, timeout: float | None = None) -> str | None:
        """
        Poll Telegram for the next message addressed to this chat_id.

        Uses long-polling (getUpdates with a server-side timeout) to avoid
        hammering the API.  The wall-clock `timeout` is honoured by looping
        with short API calls until the deadline is reached.

        Returns the message text, or None if the wall-clock timeout expired.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return None

            # Clamp API timeout to remaining wall-clock time so we don't overshoot.
            if deadline is not None:
                remaining = max(0, deadline - time.monotonic())
                api_timeout = min(self._poll_timeout, int(remaining))
            else:
                api_timeout = self._poll_timeout

            try:
                updates = self._bot.get_updates(
                    offset=self._next_offset,
                    timeout=api_timeout,
                    allowed_updates=["message"],
                )
            except Exception as exc:
                logger.warning("Telegram getUpdates error: %s — retrying in 2 s", exc)
                time.sleep(2)
                continue

            for update in updates:
                self._next_offset = update.update_id + 1
                if (
                    update.message
                    and update.message.chat
                    and update.message.chat.id == self.chat_id
                    and update.message.text
                ):
                    text = update.message.text
                    logger.debug("← Telegram [%s]: %s", self.chat_id, text[:80])
                    return text

            # No relevant message in this batch — honour poll_interval before retrying.
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(self._poll_interval)


# ──────────────────────────────────────────────────────────────────────────────
# Stdio fallback (local development)
# ──────────────────────────────────────────────────────────────────────────────

class MessengerStdio(Messenger):
    """
    Fallback messenger for local development and testing.
    Reads from stdin, writes to stdout.
    No external service required.
    """

    def send(self, text: str) -> None:
        print(f"\n[ResearchClaw] {text}", flush=True)

    def receive(self, timeout: float | None = None) -> str | None:
        print("→ ", end="", flush=True)
        try:
            line = sys.stdin.readline()
            if line:
                return line.rstrip("\n")
            return None
        except (EOFError, KeyboardInterrupt):
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_messenger(config: "Config") -> Messenger:  # type: ignore[name-defined]
    """
    Factory: instantiate the right Messenger based on config.messenger_type.

    Dispatch:
        "telegram" → TelegramMessenger  (bot token from env, chat_id from config)
        "slack"    → OpenClawMessenger  (falls back to stdio if OpenClaw missing)
        "stdio"    → MessengerStdio

    Args:
        config: fully loaded Config object (from researchclaw.config.load_config)

    Web backend note:
        Set messenger.type in config.yaml and put secrets in .env.
        Run `python -m researchclaw.init <config_path>` after writing both files
        to validate the setup and test the connection.
    """
    messenger_type = config.messenger_type.lower()

    if messenger_type == "telegram":
        logger.info("Using Telegram messenger (chat_id=%s)", config.telegram_chat_id)
        return TelegramMessenger(
            chat_id=config.telegram_chat_id,
            bot_token_env=config.telegram_bot_token_env,
            poll_timeout=config.telegram_poll_timeout,
            poll_interval=config.telegram_poll_interval,
        )

    elif messenger_type == "slack":
        logger.info("Using Slack messenger (channel=%s)", config.slack_channel)
        return SlackMessenger(
            channel=config.slack_channel,
            bot_token_env=config.slack_bot_token_env,
            app_token_env=config.slack_app_token_env,
        )

    elif messenger_type == "stdio":
        logger.info("Using stdio messenger (development mode)")
        return MessengerStdio()

    else:
        raise ValueError(
            f"Unknown messenger type: {messenger_type!r}. "
            "Supported values: slack, telegram, stdio"
        )
