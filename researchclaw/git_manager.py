"""
git_manager.py — GitHub operations: pull, diff, merge approved trials, push.

The researcher's canonical codebase lives in github_codes/.
When a trial is approved, changes from sandbox/trial_N/ are merged back
into github_codes/ and committed. The researcher is then prompted to push.
push() is only called after explicit researcher confirmation — never silently.
"""

from __future__ import annotations

import filecmp
import logging
import shutil
import subprocess
from pathlib import Path

from .models import TrialInfo

logger = logging.getLogger("researchclaw.git_manager")


class GitManager:
    """Manage the github_codes/ repository and trial merge workflow."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.repo_path = self.base_dir / "github_codes"
        # Must call authorize_merge() before merge_trial() to prevent accidental merges.
        self._merge_authorized: bool = False

    def authorize_merge(self) -> None:
        """
        Grant one-time authorization to call merge_trial().
        Must be set explicitly by the approval handler after researcher confirms.
        Authorization is consumed (reset to False) after each merge_trial() call.
        """
        self._merge_authorized = True
        logger.info("Merge into github_codes/ authorized for next merge_trial() call.")

    # ------------------------------------------------------------------
    # Repository operations
    # ------------------------------------------------------------------

    def pull(self) -> str:
        """
        Pull latest changes from remote.
        Uses --rebase to keep a clean history.
        Returns git output.
        """
        rc, stdout, stderr = self._git("pull", "--rebase")
        if rc != 0:
            raise RuntimeError(f"git pull failed:\n{stderr}")
        logger.info("git pull: %s", stdout.strip())
        return stdout

    def status(self) -> str:
        """Return current branch, last commit, and working tree status."""
        _, branch, _ = self._git("branch", "--show-current")
        _, log, _ = self._git("log", "--oneline", "-1")
        _, status, _ = self._git("status", "--short")
        return (
            f"Branch: {branch.strip()}\n"
            f"Last commit: {log.strip()}\n"
            f"Status: {status.strip() or 'clean'}"
        )

    def push(self) -> str:
        """
        Push to remote.
        Only called when researcher explicitly requests it.
        Returns git output.
        """
        rc, stdout, stderr = self._git("push")
        if rc != 0:
            raise RuntimeError(f"git push failed:\n{stderr}")
        logger.info("git push: %s", stdout.strip())
        return stdout

    # ------------------------------------------------------------------
    # Trial diff and merge
    # ------------------------------------------------------------------

    def get_diff(self, trial: TrialInfo) -> str:
        """
        Generate a human-readable diff between github_codes/ and sandbox/trial_N/.
        Returns a summary of what the agent changed during the trial.
        """
        sandbox = self.base_dir / trial.sandbox_path
        if not sandbox.exists():
            return f"(sandbox directory not found: {sandbox})"

        # Find files that differ
        changed, added, removed = self._compare_trees(self.repo_path, sandbox)

        lines = []
        if changed:
            lines.append(f"**Modified files ({len(changed)}):**")
            for f in sorted(changed):
                lines.append(f"  ~ {f}")
        if added:
            lines.append(f"\n**New files ({len(added)}):**")
            for f in sorted(added):
                lines.append(f"  + {f}")
        if removed:
            lines.append(f"\n**Removed files ({len(removed)}):**")
            for f in sorted(removed):
                lines.append(f"  - {f}")

        if not lines:
            return "(no differences found)"

        return "\n".join(lines)

    def get_full_diff(self, trial: TrialInfo) -> str:
        """
        Generate a unified diff (git-style) between github_codes/ and sandbox/trial_N/.
        Returns the raw diff text for inclusion in REPORT.md.
        """
        sandbox = self.base_dir / trial.sandbox_path
        changed, _, _ = self._compare_trees(self.repo_path, sandbox)

        diff_parts = []
        for rel_path in sorted(changed):
            src = self.repo_path / rel_path
            dst = sandbox / rel_path
            if src.exists() and dst.exists():
                result = subprocess.run(
                    ["diff", "-u", str(src), str(dst)],
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    diff_parts.append(result.stdout)

        return "\n".join(diff_parts) if diff_parts else "(no changes)"

    def merge_trial(self, trial: TrialInfo, commit_message: str) -> str:
        """
        Merge approved trial code back into github_codes/.

        REQUIRES: authorize_merge() must be called first. Authorization is consumed
        after this call to prevent accidental re-merges.

        Steps:
        1. Copy only changed/new files from sandbox/trial_N/ → github_codes/
           (unchanged files are NOT overwritten)
        2. git add -A
        3. git commit -m "{commit_message}"
        4. Does NOT push. Researcher must explicitly say "push".

        Returns the commit hash.
        """
        if not self._merge_authorized:
            raise PermissionError(
                "merge_trial() called without authorization. "
                "Call authorize_merge() after explicit researcher approval first."
            )
        self._merge_authorized = False  # consume authorization — one-time use

        sandbox = self.base_dir / trial.sandbox_path
        if not sandbox.exists():
            raise FileNotFoundError(f"Sandbox not found: {sandbox}")

        changed, added, _ = self._compare_trees(self.repo_path, sandbox)

        # Copy changed and new files
        files_copied = 0
        for rel_path in changed | added:
            src = sandbox / rel_path
            dst = self.repo_path / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logger.debug("Copied: %s", rel_path)
            files_copied += 1

        if files_copied == 0:
            logger.info("No files to merge — sandbox is identical to github_codes/")
            return "(no changes to commit)"

        # Stage and commit
        rc, _, stderr = self._git("add", "-A")
        if rc != 0:
            raise RuntimeError(f"git add failed:\n{stderr}")

        rc, stdout, stderr = self._git("commit", "-m", commit_message)
        if rc != 0:
            raise RuntimeError(f"git commit failed:\n{stderr}")

        # Extract commit hash
        _, hash_out, _ = self._git("rev-parse", "--short", "HEAD")
        commit_hash = hash_out.strip()

        logger.info("Merged trial %s → commit %s (%d files)", trial.trial_name, commit_hash, files_copied)
        return commit_hash

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compare_trees(
        self,
        src: Path,
        dst: Path,
    ) -> tuple[set[str], set[str], set[str]]:
        """
        Compare two directory trees.
        Returns (changed_files, added_files, removed_files) as sets of relative paths.

        Excludes: .git/, __pycache__/, *.pyc, *.pt, *.ckpt (large binary files)
        """
        ignore_patterns = {".git", "__pycache__", "wandb", "checkpoints"}
        ignore_extensions = {".pyc", ".pt", ".ckpt", ".safetensors", ".bin"}

        def collect_files(root: Path) -> set[str]:
            result = set()
            for p in root.rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(root))
                    # Skip ignored directories
                    parts = Path(rel).parts
                    if any(part in ignore_patterns for part in parts):
                        continue
                    if p.suffix in ignore_extensions:
                        continue
                    result.add(rel)
            return result

        src_files = collect_files(src)
        dst_files = collect_files(dst)

        added = dst_files - src_files
        removed = src_files - dst_files
        common = src_files & dst_files

        changed = set()
        for rel in common:
            if not filecmp.cmp(src / rel, dst / rel, shallow=False):
                changed.add(rel)

        return changed, added, removed

    def _git(self, *args: str) -> tuple[int, str, str]:
        """Run a git command in the repo directory."""
        result = subprocess.run(
            ["git", "-C", str(self.repo_path)] + list(args),
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr
