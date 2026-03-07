from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FakeChatInterface
from researchclaw.permissions import PermissionManager
from researchclaw.repl import ChatInput, SlashCommand, UserMessage


class TestIsReadable:
    """Tests for PermissionManager.is_readable."""

    def test_path_under_project_dir(self, tmp_path: Path) -> None:
        """Paths under project_dir are always readable."""
        pm = PermissionManager(project_dir=tmp_path)
        subfile = tmp_path / "src" / "main.py"
        subfile.parent.mkdir(parents=True, exist_ok=True)
        subfile.touch()
        assert pm.is_readable(subfile) is True

    def test_project_dir_itself(self, tmp_path: Path) -> None:
        """The project_dir itself is readable."""
        pm = PermissionManager(project_dir=tmp_path)
        assert pm.is_readable(tmp_path) is True

    def test_path_outside_project_dir_denied(self, tmp_path: Path) -> None:
        """Paths outside project_dir are not readable by default."""
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        pm = PermissionManager(project_dir=project)
        assert pm.is_readable(outside) is False

    def test_project_paths_allowed(self, tmp_path: Path) -> None:
        """Paths in project_paths are readable."""
        project = tmp_path / "project"
        project.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pm = PermissionManager(
            project_dir=project,
            project_paths=[str(data_dir)],
        )
        assert pm.is_readable(data_dir) is True

    def test_trial_paths_allowed(self, tmp_path: Path) -> None:
        """Paths in trial_paths are readable."""
        project = tmp_path / "project"
        project.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pm = PermissionManager(
            project_dir=project,
            trial_paths=[str(data_dir)],
        )
        assert pm.is_readable(data_dir) is True

    def test_subpath_of_allowed_path(self, tmp_path: Path) -> None:
        """Subpaths of allowed paths are also readable."""
        project = tmp_path / "project"
        project.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        subfile = data_dir / "file.csv"
        subfile.touch()
        pm = PermissionManager(
            project_dir=project,
            project_paths=[str(data_dir)],
        )
        assert pm.is_readable(subfile) is True

    def test_denied_path_not_readable(self, tmp_path: Path) -> None:
        """Paths not in any allowed list are denied."""
        project = tmp_path / "project"
        project.mkdir()
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        denied = tmp_path / "denied"
        denied.mkdir()
        pm = PermissionManager(
            project_dir=project,
            project_paths=[str(allowed)],
        )
        assert pm.is_readable(denied) is False


class TestRequestAccess:
    """Tests for PermissionManager.request_access."""

    def test_user_chooses_project(self, tmp_path: Path) -> None:
        """User entering '1' returns 'project' and adds to project_paths."""
        pm = PermissionManager(project_dir=tmp_path)
        target = tmp_path / "external" / "data"
        chat = FakeChatInterface(responses=[UserMessage("1")])
        result = pm.request_access(target, chat)
        assert result == "project"
        assert str(target.resolve()) in pm.project_paths

    def test_user_chooses_trial(self, tmp_path: Path) -> None:
        """User entering '2' returns 'trial' and adds to trial_paths."""
        pm = PermissionManager(project_dir=tmp_path)
        target = tmp_path / "external" / "data"
        chat = FakeChatInterface(responses=[UserMessage("2")])
        result = pm.request_access(target, chat)
        assert result == "trial"
        assert str(target.resolve()) in pm.trial_paths

    def test_user_chooses_deny(self, tmp_path: Path) -> None:
        """User entering '3' returns 'denied'."""
        pm = PermissionManager(project_dir=tmp_path)
        target = tmp_path / "external" / "data"
        chat = FakeChatInterface(responses=[UserMessage("3")])
        result = pm.request_access(target, chat)
        assert result == "denied"
        assert len(pm.project_paths) == 0
        assert len(pm.trial_paths) == 0

    def test_slash_command_returns_denied(self, tmp_path: Path) -> None:
        """Slash command input returns 'denied'."""
        pm = PermissionManager(project_dir=tmp_path)
        target = tmp_path / "external"
        chat = FakeChatInterface(responses=[SlashCommand("/abort", "")])
        result = pm.request_access(target, chat)
        assert result == "denied"

    def test_invalid_input_returns_denied(self, tmp_path: Path) -> None:
        """Invalid input (not 1, 2, or 3) returns 'denied'."""
        pm = PermissionManager(project_dir=tmp_path)
        target = tmp_path / "external"
        chat = FakeChatInterface(responses=[UserMessage("no")])
        result = pm.request_access(target, chat)
        assert result == "denied"

    def test_request_shows_panel(self, tmp_path: Path) -> None:
        """request_access sends a panel message to the chat interface."""
        pm = PermissionManager(project_dir=tmp_path)
        target = tmp_path / "external"
        chat = FakeChatInterface(responses=[UserMessage("3")])
        pm.request_access(target, chat)
        assert len(chat.sent) == 1
        assert "Read access requested" in chat.sent[0]
        assert "Allow permanent" in chat.sent[0]


class TestGetAllowedPathsPrompt:
    """Tests for PermissionManager.get_allowed_paths_prompt."""

    def test_no_paths_returns_empty(self) -> None:
        pm = PermissionManager()
        assert pm.get_allowed_paths_prompt() == ""

    def test_project_paths_included(self) -> None:
        pm = PermissionManager(project_paths=["/data/train"])
        result = pm.get_allowed_paths_prompt()
        assert "/data/train" in result
        assert "Additional Read Access" in result

    def test_trial_paths_included(self) -> None:
        pm = PermissionManager(trial_paths=["/tmp/eval"])
        result = pm.get_allowed_paths_prompt()
        assert "/tmp/eval" in result

    def test_both_paths_combined(self) -> None:
        pm = PermissionManager(
            project_paths=["/data/train"],
            trial_paths=["/tmp/eval"],
        )
        result = pm.get_allowed_paths_prompt()
        assert "/data/train" in result
        assert "/tmp/eval" in result


class TestPermissionManagerDataclass:
    """Tests for PermissionManager dataclass defaults."""

    def test_default_project_paths_empty(self) -> None:
        pm = PermissionManager()
        assert pm.project_paths == []

    def test_default_trial_paths_empty(self) -> None:
        pm = PermissionManager()
        assert pm.trial_paths == []

    def test_default_project_dir(self) -> None:
        pm = PermissionManager()
        assert pm.project_dir == Path(".")
