from __future__ import annotations

import importlib
import subprocess
import sys


def ensure_package(package_name: str, pip_name: str | None = None) -> bool:
    """Ensure a Python package is installed in the current environment.

    Tries to import the package first. If not available, installs it using
    sys.executable -m pip install. This works correctly with pipx, uv tool,
    and plain pip installs.

    Args:
        package_name: The importable Python package name (e.g., 'anthropic').
        pip_name: The pip install name if different from package_name
                  (e.g., 'claude-agent-sdk' for package 'claude_agent_sdk').
                  Defaults to package_name if not provided.

    Returns:
        True if the package is available (already installed or newly installed).
        False if installation failed.
    """
    if pip_name is None:
        pip_name = package_name

    # Check if already importable
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        pass

    # Attempt installation
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False

    # Verify installation succeeded
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False
