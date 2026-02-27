"""
access_control.py — PathValidator: enforces read/write/execute permission rules.

CRITICAL: All file I/O in the agent MUST go through this class.
The agent runs as the same OS user as the researcher, so we cannot rely on
OS-level file permissions. Every file operation is validated here.

Permission model:
  READ-ONLY     : github_codes/**, sandbox/**, experiment_reports/**,
                  reference/**  (user-managed context: papers, docs, external codebases)
                  researchclaw/** (own code, readable/writable during prototype)
                  config.yaml, .trials.jsonl, MEMORY.md, RESEARCH_TRIAL_SUMMARY.md
  ALWAYS-WRITE  : MEMORY.md — agent-managed persistent memory; writable at any time,
                  no active trial required. Agent overwrites the whole file each update.
                  RESEARCH_TRIAL_SUMMARY.md — auto-updated after each trial finalization.
  READ-WRITE    : sandbox/{today}/trial_{N}/** and experiment_reports/{today}/trial_{N}/**
                  — only when current trial status == "active"
  NO ACCESS     : anything outside base_dir
                  — write to github_codes/ is always denied via PathValidator
                  — write to reference/ is always denied (user manages it manually)
                  — trial write access permanently revoked when trial is approved/rejected
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path

from .models import TrialInfo

logger = logging.getLogger(__name__)

# Directories the agent is allowed to read (relative to base_dir).
# reference/ is user-managed (papers, docs, external codebases) — always read-only.
_READABLE_ROOTS = ("github_codes", "sandbox", "experiment_reports", "reference", "researchclaw")

# Top-level files at base_dir that are readable
_READABLE_BASE_FILES = ("config.yaml", ".trials.jsonl", "MEMORY.md", "RESEARCH_TRIAL_SUMMARY.md")

# Top-level files the agent may overwrite at any time — no active trial required.
# MEMORY.md is the agent's persistent cross-session memory (analogous to Claude Code's MEMORY.md).
_ALWAYS_WRITABLE_BASE_FILES: frozenset[str] = frozenset({"MEMORY.md", "RESEARCH_TRIAL_SUMMARY.md"})

# ── Shell command security rules ──────────────────────────────────────────────

# Regex: detect `python -c` (or python3 -c, python3.12 -c) anywhere in a pipeline.
# Matches after start-of-string, pipe (|), semicolon (;), or && / ||.
_PYTHON_DASH_C_RE = re.compile(
    r'(?:^|[|;&])\s*python\w*(?:\s+\S+)*\s+-c\b'
)

# git subcommands that modify repository state.
# These MUST go through GitManager's explicit authorization flow — never raw shell.
_GIT_MUTATIVE_SUBCOMMANDS: frozenset[str] = frozenset({
    "add", "commit", "push", "merge", "rebase", "reset", "checkout",
    "restore", "switch", "fetch", "pull", "cherry-pick", "stash",
    "tag", "rm", "mv", "clean", "bisect", "reflog",
})


class PathValidator:
    """
    Enforces ResearchClaw's three-tier access control model.

    Usage:
        validator = PathValidator("/path/to/workspace")
        validator.set_trial(trial)           # grant write access to active trial
        path = validator.validate_write(p)  # raises PermissionError if denied
        validator.set_trial(None)            # revoke write access (trial finalized)
    """

    def __init__(self, base_dir: str, current_trial: TrialInfo | None = None):
        self.base_dir = Path(base_dir).resolve()
        self.current_trial = current_trial

    def set_trial(self, trial: TrialInfo | None) -> None:
        """
        Update the current trial reference.
        Call with a TrialInfo when a trial becomes active.
        Call with None to revoke all write access (trial finalized).
        """
        self.current_trial = trial

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    def can_read(self, path: str) -> bool:
        """Returns True if the agent is allowed to read this path."""
        try:
            p = Path(path).resolve()
        except Exception:
            return False

        # Must be inside base_dir (catches /etc/passwd, ../../ traversals, etc.)
        try:
            p.relative_to(self.base_dir)
        except ValueError:
            return False

        # Allow whitelisted subdirectories
        for root in _READABLE_ROOTS:
            try:
                p.relative_to(self.base_dir / root)
                return True
            except ValueError:
                pass

        # Allow specific top-level files
        if p.parent == self.base_dir and p.name in _READABLE_BASE_FILES:
            return True

        return False

    def can_write(self, path: str) -> bool:
        """
        Returns True if the agent is allowed to write/create/delete this path.

        Two write tiers:
          1. Always-writable base files (e.g. MEMORY.md) — no active trial required.
          2. Trial-scoped writes — requires an active trial, limited to:
               sandbox/{trial.date}/{trial.trial_name}/**
               experiment_reports/{trial.date}/{trial.trial_name}/**
        """
        try:
            p = Path(path).resolve()
        except Exception:
            return False

        # Tier 1: agent-managed base files are always writable (no trial needed).
        if p.parent == self.base_dir and p.name in _ALWAYS_WRITABLE_BASE_FILES:
            return True

        # Tier 2: trial-scoped writes.
        if self.current_trial is None or not self.current_trial.is_writable:
            return False

        trial = self.current_trial
        allowed_dirs = [
            self.base_dir / trial.sandbox_path,
            self.base_dir / trial.report_path,
        ]
        for allowed in allowed_dirs:
            try:
                p.relative_to(allowed)
                return True
            except ValueError:
                pass

        return False

    def can_execute(self, path: str) -> bool:
        """
        Returns True if the agent can execute a file at this path.
        Execution is allowed only inside the current trial's sandbox (same as write).
        """
        return self.can_write(path)

    # ------------------------------------------------------------------
    # Validated accessors (raise PermissionError on denial)
    # ------------------------------------------------------------------

    def validate_read(self, path: str) -> Path:
        """Validate read access and return resolved Path, or raise PermissionError."""
        if not self.can_read(path):
            msg = f"READ denied: {path}"
            logger.error(msg)
            raise PermissionError(msg)
        return Path(path).resolve()

    def validate_write(self, path: str) -> Path:
        """Validate write access and return resolved Path, or raise PermissionError."""
        if not self.can_write(path):
            trial_info = (
                f"trial={self.current_trial.trial_name}, status={self.current_trial.status}"
                if self.current_trial
                else "no active trial"
            )
            msg = f"WRITE denied ({trial_info}): {path}"
            logger.error(msg)
            raise PermissionError(msg)
        return Path(path).resolve()

    def validate_execute(self, path: str) -> Path:
        """Validate execute access and return resolved Path, or raise PermissionError."""
        if not self.can_execute(path):
            msg = f"EXECUTE denied: {path}"
            logger.error(msg)
            raise PermissionError(msg)
        return Path(path).resolve()

    # ------------------------------------------------------------------
    # Shell command validation
    # ------------------------------------------------------------------

    def validate_shell_command(self, cmd: str) -> None:
        """
        Parse a shell command and apply all security rules before execution.
        Raises PermissionError on any violation.

        Rules enforced (in order):
          1. python -c is warned — allowed only because runner cwd is sandbox/.
             Shell-level output paths are still validated by rule 3.
          2. Mutative git subcommands are blocked — must use GitManager flow.
          3. git -C <path> must point inside github_codes/.
          4. Output path destinations (>, tee, cp, mv, -o) must pass can_write().
        """
        self._warn_python_dash_c(cmd)
        self._check_git_scope(cmd)

        output_paths = self._parse_output_paths(cmd)
        for p in output_paths:
            if not self.can_write(p):
                msg = f"WRITE denied in shell command output path '{p}' in: {cmd}"
                logger.error(msg)
                raise PermissionError(msg)

    def _warn_python_dash_c(self, cmd: str) -> None:
        """
        Log a warning when `python -c` is detected.

        python -c is permitted because:
          - runner.py always sets cwd to the trial's sandbox directory
          - Shell-level output paths (redirects, tee, cp) are caught by the
            output-path validator below
          - The OS user (appuser) cannot write outside their home directory

        We still log a warning so it's visible in the audit trail.
        """
        if _PYTHON_DASH_C_RE.search(cmd):
            logger.warning(
                "python -c detected in command — allowed (cwd=sandbox); "
                "shell-level output paths will still be validated: %s", cmd
            )

    def _check_git_scope(self, cmd: str) -> None:
        """
        Enforce git command restrictions:
          - Mutative git subcommands (commit, push, add, merge, …) are always
            blocked via shell.  They must go through GitManager's explicit
            authorization flow.
          - If a git command uses '-C <path>', that path must resolve to within
            github_codes/.  This prevents operating on repos outside the project.

        Read-only git subcommands (status, log, diff, branch, show) without
        a -C flag are allowed — they will naturally fail if the cwd isn't a
        git repository, which is the expected sandbox environment.
        """
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()

        # Find 'git' token anywhere in the pipeline (handles: cmd && git ..., etc.)
        git_idx = next((i for i, t in enumerate(tokens) if t == "git"), None)
        if git_idx is None:
            return  # no git command present

        git_args = tokens[git_idx + 1:]

        # Walk git's global flags to find the subcommand and any -C path.
        explicit_dir: str | None = None
        subcommand: str | None = None
        i = 0
        while i < len(git_args):
            tok = git_args[i]
            if tok == "-C" and i + 1 < len(git_args):
                explicit_dir = git_args[i + 1]
                i += 2
            elif tok.startswith("-C") and len(tok) > 2:  # -C/path (no space)
                explicit_dir = tok[2:]
                i += 1
            elif tok.startswith("--git-dir="):
                explicit_dir = tok[len("--git-dir="):]
                i += 1
            elif tok.startswith("--work-tree="):
                explicit_dir = tok[len("--work-tree="):]
                i += 1
            elif tok.startswith("-"):
                i += 1  # skip other global flags
            else:
                subcommand = tok  # first non-flag token is the subcommand
                break

        # 1. Block mutative git subcommands regardless of path.
        if subcommand and subcommand in _GIT_MUTATIVE_SUBCOMMANDS:
            msg = (
                f"Blocked: 'git {subcommand}' is not allowed via shell commands.\n"
                "Git write operations (add, commit, push, merge, pull, …) are "
                "handled exclusively by ResearchClaw's built-in GitManager flow "
                "and require explicit researcher approval.\n"
                f"Command: {cmd}"
            )
            logger.error(msg)
            raise PermissionError(msg)

        # 2. If -C path is given, it must be within github_codes/.
        if explicit_dir is not None:
            try:
                resolved = Path(explicit_dir).resolve()
                resolved.relative_to(self.base_dir / "github_codes")
            except ValueError:
                msg = (
                    f"Blocked: git -C path must be within github_codes/.\n"
                    f"Given path: {explicit_dir}\n"
                    f"Command: {cmd}"
                )
                logger.error(msg)
                raise PermissionError(msg)

    def _parse_output_paths(self, cmd: str) -> list[str]:
        """Extract output file paths from a shell command string."""
        paths: list[str] = []

        # Tokenize safely (best-effort; handles quoted strings)
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()

        # 1. Redirect operators: >, >>, 2>, &>, 2>>, &>>
        redirect_pattern = re.compile(
            r'(?:^|(?<=\s))(?:2>|&>|>>|2>>|&>>|>)\s*(\S+)'
        )
        for match in redirect_pattern.finditer(cmd):
            paths.append(match.group(1))

        # 2. tee [options] file... — tee can appear after a pipe, find it anywhere
        for i, tok in enumerate(tokens):
            if tok == "tee":
                for sub in tokens[i + 1:]:
                    if sub == "|":
                        break  # stop at next pipe
                    if not sub.startswith("-"):
                        paths.append(sub)

        # 3. cp src... dest  (last argument is destination)
        for i, tok in enumerate(tokens):
            if tok == "cp":
                args = [t for t in tokens[i + 1:] if not t.startswith("-") and t != "|"]
                if len(args) >= 2:
                    paths.append(args[-1])
                break

        # 4. mv src... dest  (last argument is destination)
        for i, tok in enumerate(tokens):
            if tok == "mv":
                args = [t for t in tokens[i + 1:] if not t.startswith("-") and t != "|"]
                if len(args) >= 2:
                    paths.append(args[-1])
                break

        # 5. -o / --output <file>
        for i, tok in enumerate(tokens):
            if tok in ("-o", "--output") and i + 1 < len(tokens):
                paths.append(tokens[i + 1])

        return paths
