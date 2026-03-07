"""Tests for researchclaw.fsm._shared — shared FSM utilities and constants.

Created as part of US-R01.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm._shared import (
    EXCLUDED_DIRS,
    KEY_FILES,
    MAX_CONTEXT_CHARS,
    SYSTEM_PROMPT_NO_TERMINAL,
    SYSTEM_PROMPT_PROACTIVE,
    gather_project_context,
    get_provider_safe,
    noop_context,
    parse_llm_response,
    persist_autopilot,
)
from researchclaw.sandbox import SandboxManager


# ---------------------------------------------------------------------------
# noop_context
# ---------------------------------------------------------------------------

class TestNoopContext:
    def test_yields_none(self) -> None:
        with noop_context() as val:
            assert val is None

    def test_body_executes(self) -> None:
        executed = False
        with noop_context():
            executed = True
        assert executed


# ---------------------------------------------------------------------------
# get_provider_safe
# ---------------------------------------------------------------------------

class TestGetProviderSafe:
    def test_returns_provider_on_success(self) -> None:
        fake_provider = MagicMock()
        with patch(
            "researchclaw.llm.provider.get_provider", return_value=fake_provider
        ):
            result = get_provider_safe(ResearchClawConfig())
        assert result is fake_provider

    def test_returns_none_on_exception(self) -> None:
        with patch(
            "researchclaw.llm.provider.get_provider",
            side_effect=RuntimeError("no provider"),
        ):
            result = get_provider_safe(ResearchClawConfig())
        assert result is None


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_both_sections(self) -> None:
        response = (
            "### REQUIREMENTS\nnumpy\npandas\n\n### CODE\nprint('hello')\n"
        )
        reqs, code = parse_llm_response(response)
        assert "numpy" in reqs
        assert "pandas" in reqs
        assert "print('hello')" in code

    def test_code_only(self) -> None:
        response = "### CODE\nprint('hello')\n"
        reqs, code = parse_llm_response(response)
        assert reqs == ""
        assert "print('hello')" in code

    def test_no_sections(self) -> None:
        response = "print('hello')"
        reqs, code = parse_llm_response(response)
        assert reqs == ""
        assert code == "print('hello')"

    def test_requirements_none(self) -> None:
        response = "### REQUIREMENTS\nNONE\n\n### CODE\nprint('hello')\n"
        reqs, code = parse_llm_response(response)
        assert reqs == ""

    def test_code_block_markers_stripped(self) -> None:
        response = "```python\nprint('hello')\n```"
        reqs, code = parse_llm_response(response)
        assert "print('hello')" in code
        assert "```" not in code


# ---------------------------------------------------------------------------
# gather_project_context
# ---------------------------------------------------------------------------

class TestGatherProjectContext:
    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir"
        result = gather_project_context(missing)
        assert result == ""

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = gather_project_context(tmp_path)
        assert "## Project Structure" in result

    def test_includes_key_files(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Hello")
        result = gather_project_context(tmp_path)
        assert "README.md" in result
        assert "# Hello" in result

    def test_excludes_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "test.pyc").write_text("data")
        result = gather_project_context(tmp_path)
        assert "__pycache__" not in result

    def test_truncation(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("A" * 10000)
        result = gather_project_context(tmp_path)
        assert len(result) <= MAX_CONTEXT_CHARS + 50  # small margin for truncation suffix


# ---------------------------------------------------------------------------
# persist_autopilot
# ---------------------------------------------------------------------------

class TestPersistAutopilot:
    def test_creates_config(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config = ResearchClawConfig(autopilot=True)
        persist_autopilot(config, project_dir)

        config_path = (
            SandboxManager.sandbox_path(project_dir)
            / "project_settings"
            / "researchclaw.yaml"
        )
        assert config_path.exists()
        loaded = ResearchClawConfig.load_from_yaml(config_path)
        assert loaded.autopilot is True

    def test_updates_existing_config(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = (
            SandboxManager.sandbox_path(project_dir)
            / "project_settings"
            / "researchclaw.yaml"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        initial = ResearchClawConfig(autopilot=False, model="gpt-4")
        initial.save_to_yaml(config_path)

        config = ResearchClawConfig(autopilot=True)
        persist_autopilot(config, project_dir)

        loaded = ResearchClawConfig.load_from_yaml(config_path)
        assert loaded.autopilot is True


# ---------------------------------------------------------------------------
# String constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_no_terminal_content(self) -> None:
        assert "You do NOT have access to a terminal" in SYSTEM_PROMPT_NO_TERMINAL
        assert "Never say 'I ran this command'" in SYSTEM_PROMPT_NO_TERMINAL

    def test_proactive_content(self) -> None:
        assert "Be proactive" in SYSTEM_PROMPT_PROACTIVE
        assert "present 2-3 concrete options" in SYSTEM_PROMPT_PROACTIVE

    def test_excluded_dirs_type(self) -> None:
        assert isinstance(EXCLUDED_DIRS, set)
        assert "__pycache__" in EXCLUDED_DIRS

    def test_key_files_type(self) -> None:
        assert isinstance(KEY_FILES, tuple)
        assert "README.md" in KEY_FILES

    def test_max_context_chars(self) -> None:
        assert MAX_CONTEXT_CHARS == 8000
