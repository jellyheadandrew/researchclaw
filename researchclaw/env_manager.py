"""
env_manager.py — Copy-on-write environment manager for experiment trials.

Manages a chain of Python environments (venv or conda) under envs/:
  envs/env_000/  — bootstrap (bare)
  envs/env_001/  — after first pip install
  envs/env_002/  — after second pip install
  ...

All trials share the current environment until a pip/conda install is
detected.  When that happens, the current env is frozen (pip freeze),
a new env is created from the freeze, and the mutation command is
applied in the new env.  Subsequent trials use the new env.

State is persisted to .envs.jsonl (append-only, one JSON object per line).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("researchclaw.env_manager")

# ---------------------------------------------------------------------------
# Regex to detect environment-mutating commands.
# Matches pip install/uninstall, conda install/remove, mamba, python -m pip, etc.
# ---------------------------------------------------------------------------
_ENV_MUTATION_RE = re.compile(
    r"(?:^|[|;&]\s*)"                           # start or pipeline/chain
    r"(?:"
    r"pip3?\s+"                                  # pip / pip3
    r"|python[\w.]*\s+(?:-\S+\s+)*-m\s+pip\s+"  # python -m pip
    r"|conda\s+"                                 # conda
    r"|mamba\s+"                                 # mamba
    r")"
    r"(?:install|uninstall|remove|update|upgrade)\b"
)


class EnvManager:
    """Copy-on-write environment manager.

    Environments live at ``{project_root}/envs/env_000/``, ``env_001/``, etc.
    A ledger at ``{project_root}/.envs.jsonl`` tracks lineage.
    """

    def __init__(self, project_root: str, backend: str = "venv"):
        """
        Args:
            project_root: Absolute path to the ResearchClaw project directory
                          (e.g. ``/path/to/ResearchClaw``).
            backend: ``"venv"`` or ``"conda"``.
        """
        self.project_root = Path(project_root).resolve()
        self.envs_dir = self.project_root / "envs"
        self.ledger_file = self.project_root / ".envs.jsonl"
        self.backend = backend
        self._current_env_id: int = -1  # set by _load_state() or bootstrap()
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bootstrap(self) -> Path:
        """Create ``env_000`` if it doesn't already exist.

        Called once at agent startup.  Idempotent — if ``env_000`` exists and
        the ledger records it, this is a no-op.

        Returns:
            Path to the bootstrapped environment.
        """
        env_path = self._env_path(0)

        if env_path.exists() and self._current_env_id >= 0:
            logger.info("env_000 already exists, skipping bootstrap")
            return self.get_active_env_path()

        self.envs_dir.mkdir(parents=True, exist_ok=True)

        if not env_path.exists():
            self._create_bare_env(env_path)

        if self._current_env_id < 0:
            self._current_env_id = 0
            self._save_entry({
                "env_id": 0,
                "created_at": datetime.now().isoformat(),
                "parent_id": None,
                "backend": self.backend,
                "trigger": "bootstrap",
                "trial": None,
            })

        logger.info("Bootstrapped env_000 at %s", env_path)
        return env_path

    def get_active_env_path(self) -> Path:
        """Return the path to the current active environment."""
        if self._current_env_id < 0:
            raise RuntimeError(
                "No active environment. Call bootstrap() first."
            )
        return self._env_path(self._current_env_id)

    @property
    def current_env_id(self) -> int:
        """The numeric id of the active environment (e.g. 0, 1, 2)."""
        return self._current_env_id

    def is_env_mutation(self, cmd: str) -> bool:
        """Return True if *cmd* would mutate the Python environment."""
        return bool(_ENV_MUTATION_RE.search(cmd))

    def fork_env(self, reason: str, trial_name: str = "") -> Path:
        """Clone the current env into a new one via pip freeze.

        Steps:
          1. ``pip freeze`` from current env → requirements text
          2. Create a fresh bare env at ``envs/env_{M+1}``
          3. ``pip install -r`` the frozen requirements into the new env
          4. Persist to ledger, bump ``_current_env_id``

        Args:
            reason: Human-readable reason (e.g. the pip install command).
            trial_name: Optional trial name for the ledger.

        Returns:
            Path to the newly created environment.
        """
        old_id = self._current_env_id
        new_id = old_id + 1
        old_path = self._env_path(old_id)
        new_path = self._env_path(new_id)

        if new_path.exists():
            raise FileExistsError(f"Environment already exists: {new_path}")

        logger.info("Forking env_%03d -> env_%03d (reason: %s)", old_id, new_id, reason)

        # 1. Freeze the old environment
        frozen = self._pip_freeze(old_path)

        # 2. Create new bare env
        self._create_bare_env(new_path)

        # 3. Install frozen requirements (if any)
        if frozen.strip():
            self._pip_install_requirements(new_path, frozen)

        # 4. Update state
        self._current_env_id = new_id
        self._save_entry({
            "env_id": new_id,
            "created_at": datetime.now().isoformat(),
            "parent_id": old_id,
            "backend": self.backend,
            "trigger": reason,
            "trial": trial_name or None,
        })

        logger.info("Created env_%03d at %s", new_id, new_path)
        return new_path

    def apply_mutation(self, cmd: str, trial_name: str = "") -> Path:
        """Fork the environment, then run the mutation command in the new env.

        Args:
            cmd: The full shell command (e.g. ``pip install torch``).
            trial_name: Optional trial name for the ledger.

        Returns:
            Path to the new environment (with the mutation applied).
        """
        new_path = self.fork_env(reason=cmd, trial_name=trial_name)

        # Run the mutation command inside the new environment
        self._run_in_env(new_path, cmd)

        return new_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _env_path(self, env_id: int) -> Path:
        """Return ``envs/env_{id:03d}``."""
        return self.envs_dir / f"env_{env_id:03d}"

    def _create_bare_env(self, env_path: Path) -> None:
        """Create a bare environment (venv or conda) at *env_path*."""
        if self.backend == "conda":
            # Determine Python version to match the current interpreter
            pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
            subprocess.run(
                ["conda", "create", "-p", str(env_path), f"python={pyver}", "-y", "--quiet"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "venv", str(env_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        logger.info("Created bare %s env at %s", self.backend, env_path)

    def _pip_freeze(self, env_path: Path) -> str:
        """Run ``pip freeze`` inside *env_path* and return the output."""
        pip = self._pip_executable(env_path)
        result = subprocess.run(
            [str(pip), "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def _pip_install_requirements(self, env_path: Path, requirements: str) -> None:
        """Write *requirements* to a temp file and ``pip install -r`` into *env_path*."""
        pip = self._pip_executable(env_path)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="envmgr_freeze_", delete=False
        ) as f:
            f.write(requirements)
            f.flush()
            tmp_path = f.name

        try:
            subprocess.run(
                [str(pip), "install", "--quiet", "-r", tmp_path],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _run_in_env(self, env_path: Path, cmd: str) -> None:
        """Run a shell command with *env_path*'s bin/ prepended to PATH."""
        import os

        env = os.environ.copy()
        bin_dir = str(env_path / "bin")
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(env_path)
        env.pop("CONDA_DEFAULT_ENV", None)
        env.pop("CONDA_PREFIX", None)

        result = subprocess.run(
            cmd,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                "Env mutation command failed (rc=%d): %s\nstderr: %s",
                result.returncode, cmd, result.stderr,
            )
            raise RuntimeError(
                f"Environment command failed (exit {result.returncode}):\n"
                f"  cmd: {cmd}\n"
                f"  stderr: {result.stderr.strip()}"
            )

    def _pip_executable(self, env_path: Path) -> Path:
        """Return the pip executable inside *env_path*."""
        pip = env_path / "bin" / "pip"
        if not pip.exists():
            pip = env_path / "bin" / "pip3"
        if not pip.exists():
            raise FileNotFoundError(f"pip not found in {env_path / 'bin'}")
        return pip

    def _save_entry(self, entry: dict) -> None:
        """Append an entry to ``.envs.jsonl``."""
        with open(self.ledger_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_state(self) -> None:
        """Read ``.envs.jsonl`` and set ``_current_env_id`` to the latest."""
        if not self.ledger_file.exists():
            self._current_env_id = -1
            return

        max_id = -1
        with open(self.ledger_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    eid = entry.get("env_id", -1)
                    if eid > max_id:
                        max_id = eid
                except (json.JSONDecodeError, KeyError):
                    continue

        self._current_env_id = max_id
