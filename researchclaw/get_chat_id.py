"""
get_chat_id.py — Helper to discover your Telegram chat ID.

Usage:
    python -m researchclaw.get_chat_id [--token <BOT_TOKEN>]

If --token is not provided, the token is read from the TELEGRAM_BOT_TOKEN
environment variable (or .env in the current directory).

How to use:
  1. Create a bot via @BotFather on Telegram (/newbot).
  2. Send *any* message to your new bot (or add it to a group and send a message).
  3. Run this script.  It prints the chat_id for each sender.
  4. Copy the chat_id into config.yaml under messenger.telegram_chat_id.
"""

from __future__ import annotations

import os
import sys


def _load_env() -> None:
    """Try to load .env from the current directory."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


def main() -> None:
    _load_env()

    # ── Parse arguments ──────────────────────────────────────────────────────
    args = sys.argv[1:]
    token: str | None = None
    raw_mode = "--raw" in args  # machine-readable: one chat_id per line, no decoration

    if "--token" in args:
        idx = args.index("--token")
        if idx + 1 < len(args):
            token = args[idx + 1]
        else:
            print("Error: --token requires a value", file=sys.stderr)
            sys.exit(1)

    if token is None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")

    if not token:
        print(
            "No bot token found.\n"
            "Provide it via --token <TOKEN> or set TELEGRAM_BOT_TOKEN in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Fetch updates ────────────────────────────────────────────────────────
    try:
        import telebot
    except ImportError:
        print(
            "pyTelegramBotAPI is not installed.  Run:\n"
            "  pip install pyTelegramBotAPI",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = telebot.TeleBot(token, parse_mode=None)

    if not raw_mode:
        print("Fetching recent messages from the bot API...")
    try:
        updates = bot.get_updates(timeout=10)
    except Exception as exc:
        print(f"Error fetching updates: {exc}", file=sys.stderr)
        sys.exit(1)

    if not updates:
        if not raw_mode:
            print(
                "\nNo messages found.\n"
                "Send any message to your bot first, then re-run this script.\n"
                "(If the bot is in a group, make sure it can read messages.)"
            )
        sys.exit(0)

    seen: set[int] = set()
    if not raw_mode:
        print()
    for update in updates:
        if not update.message or not update.message.chat:
            continue
        chat = update.message.chat
        if chat.id in seen:
            continue
        seen.add(chat.id)

        if raw_mode:
            print(chat.id)
        else:
            chat_type = chat.type  # private | group | supergroup | channel
            if chat_type == "private":
                label = f"@{chat.username}" if chat.username else chat.first_name or "?"
            else:
                label = f"group: {chat.title or '?'}"
            print(f"  chat_id: {chat.id}   ({chat_type}, {label})")

    if not raw_mode:
        print(
            "\nCopy the chat_id for your account into config.yaml:\n"
            "  messenger:\n"
            "    telegram_chat_id: \"<paste here>\""
        )


if __name__ == "__main__":
    main()
