"""Interactive git identity setup for onboarding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load_config
from .models import Settings
from .storage import StorageManager


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)
    storage = StorageManager(cfg.base_dir)
    storage.ensure_layout()
    settings = storage.load_settings()

    print("\n=== Git Identity Setup ===")
    print("These values are used when committing and pushing to GitHub.")
    print("Leave blank to skip (you can set them later via /settings).\n")

    name = input("Git user.name: ").strip()
    email = input("Git user.email: ").strip()

    print("\nAuth method:")
    print("  system - use your local git/ssh credentials (default)")
    print("  token  - store a GitHub PAT for HTTPS push")
    auth = input("Auth method [system]: ").strip().lower() or "system"
    if auth not in {"system", "token"}:
        auth = "system"

    if name:
        settings.git_user_name = name
    if email:
        settings.git_user_email = email
    settings.git_auth_method = auth

    storage.save_settings(settings)
    print("\nGit identity saved to settings.")


if __name__ == "__main__":
    main()
