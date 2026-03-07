from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchclaw.config import ResearchClawConfig
    from researchclaw.models import TrialMeta


@dataclass
class PermissionManager:
    """Read-only permission manager for paths outside the project directory.

    Manages two tiers of allowed read paths:
    - project_paths: permanently allowed (persisted in project config)
    - trial_paths: allowed for the current trial only (persisted in TrialMeta)

    Write access is NEVER granted through this system.
    """

    project_paths: list[str] = field(default_factory=list)
    trial_paths: list[str] = field(default_factory=list)
    project_dir: Path = field(default_factory=lambda: Path("."))

    def is_readable(self, path: str | Path) -> bool:
        """Check if a path is readable.

        Returns True if the path is under project_dir, or in project_paths
        or trial_paths.
        """
        resolved = Path(path).resolve()
        project_resolved = self.project_dir.resolve()

        # Always allow paths under project_dir
        try:
            resolved.relative_to(project_resolved)
            return True
        except ValueError:
            pass

        # Check project-level allowed paths
        resolved_str = str(resolved)
        for allowed in self.project_paths:
            allowed_resolved = str(Path(allowed).resolve())
            if resolved_str == allowed_resolved or resolved_str.startswith(
                allowed_resolved + "/"
            ):
                return True

        # Check trial-level allowed paths
        for allowed in self.trial_paths:
            allowed_resolved = str(Path(allowed).resolve())
            if resolved_str == allowed_resolved or resolved_str.startswith(
                allowed_resolved + "/"
            ):
                return True

        return False

    def request_access(
        self, path: str | Path, chat_interface: Any
    ) -> str:
        """Request read access to a path via TUI approval.

        Shows a Rich Panel with 3 numbered options:
        (1) Allow permanent — adds to project_paths (persisted in config)
        (2) Allow this trial — adds to trial_paths (persisted in TrialMeta)
        (3) Deny

        Returns:
            'project', 'trial', or 'denied'
        """
        path_str = str(Path(path).resolve())

        panel_content = (
            f"**Read access requested**\n\n"
            f"Path: {path_str}\n\n"
            f"  **(1)** Allow permanent  — remember for this project\n"
            f"  **(2)** Allow this trial — allow for current trial only\n"
            f"  **(3)** Deny             — do not allow"
        )
        chat_interface.send(panel_content)

        from researchclaw.repl import SlashCommand, UserMessage

        user_input = chat_interface.receive()

        if isinstance(user_input, SlashCommand):
            return "denied"

        text = user_input.text.strip() if isinstance(user_input, UserMessage) else ""

        if text == "1":
            self.project_paths.append(path_str)
            return "project"
        elif text == "2":
            self.trial_paths.append(path_str)
            return "trial"
        else:
            return "denied"

    def get_allowed_paths_prompt(self) -> str:
        """Build a prompt snippet listing all allowed read paths.

        Returns an empty string if no additional paths are allowed.
        """
        all_paths = list(self.project_paths) + list(self.trial_paths)
        if not all_paths:
            return ""
        paths_list = "\n".join(f"- {p}" for p in all_paths)
        return (
            f"\n## Additional Read Access\n"
            f"You may read files from these additional paths:\n{paths_list}\n"
        )


def build_read_paths_section(config: ResearchClawConfig, meta: TrialMeta | None = None) -> str:
    """Build a system prompt section listing allowed read paths."""
    paths = list(config.read_only_paths)
    if meta is not None:
        paths.extend(meta.trial_read_paths)
    if not paths:
        return ""
    listing = "\n".join(f"- {p}" for p in paths)
    return f"\n\n## Additional Readable Paths\nYou may read files from these additional paths:\n{listing}\n"
