"""Tests for US-023: PyPI publishability and final cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import FakeChatInterface
from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.merge import handle_merge_loop
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput


# --- MERGE_LOOP handler tests ---


class TestHandleMergeLoop:
    """Tests for the MERGE_LOOP stub handler."""

    def test_returns_decide(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir(parents=True)
        meta = TrialMeta()
        chat = FakeChatInterface()
        result = handle_merge_loop(trial_dir, meta, ResearchClawConfig(), chat)
        assert result == State.DECIDE

    def test_sends_not_implemented_message(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir(parents=True)
        meta = TrialMeta()
        chat = FakeChatInterface()
        handle_merge_loop(trial_dir, meta, ResearchClawConfig(), chat)
        assert len(chat.sent) == 1
        assert "not yet implemented" in chat.sent[0].lower()

    def test_works_with_none_chat(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "20260302_trial_001"
        trial_dir.mkdir(parents=True)
        meta = TrialMeta()
        result = handle_merge_loop(trial_dir, meta, ResearchClawConfig(), None)
        assert result == State.DECIDE


# --- pyproject.toml validation tests ---


class TestPyprojectToml:
    """Tests that pyproject.toml has all required fields for PyPI publishing."""

    @pytest.fixture()
    def pyproject(self) -> dict[str, Any]:
        import tomllib

        project_root = Path(__file__).parent.parent
        with open(project_root / "pyproject.toml", "rb") as f:
            return tomllib.load(f)

    def test_has_name(self, pyproject: dict[str, Any]) -> None:
        assert pyproject["project"]["name"] == "researchclaw"

    def test_has_version(self, pyproject: dict[str, Any]) -> None:
        assert "version" in pyproject["project"]

    def test_has_description(self, pyproject: dict[str, Any]) -> None:
        assert "description" in pyproject["project"]
        assert len(pyproject["project"]["description"]) > 0

    def test_has_readme(self, pyproject: dict[str, Any]) -> None:
        assert pyproject["project"]["readme"] == "README.md"

    def test_has_license(self, pyproject: dict[str, Any]) -> None:
        assert pyproject["project"]["license"] == "MIT"

    def test_has_requires_python(self, pyproject: dict[str, Any]) -> None:
        assert "requires-python" in pyproject["project"]

    def test_has_dependencies(self, pyproject: dict[str, Any]) -> None:
        deps = pyproject["project"]["dependencies"]
        dep_names = [d.split(">=")[0].split("[")[0].strip() for d in deps]
        assert "click" in dep_names
        assert "pyyaml" in dep_names
        assert "rich" in dep_names
        assert "jinja2" in dep_names
        assert "prompt-toolkit" in dep_names

    def test_has_entry_point(self, pyproject: dict[str, Any]) -> None:
        assert "researchclaw" in pyproject["project"]["scripts"]
        assert pyproject["project"]["scripts"]["researchclaw"] == "researchclaw.cli:main"

    def test_no_llm_sdks_in_hard_dependencies(self, pyproject: dict[str, Any]) -> None:
        deps = [d.lower() for d in pyproject["project"]["dependencies"]]
        for dep in deps:
            assert "anthropic" not in dep
            assert "openai" not in dep
            assert "claude-agent-sdk" not in dep

    def test_build_system(self, pyproject: dict[str, Any]) -> None:
        assert pyproject["build-system"]["build-backend"] == "hatchling.build"


# --- Package structure tests ---


class TestPackageStructure:
    """Tests that the package structure is correct for publishing."""

    @pytest.fixture()
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    def test_license_file_exists(self, project_root: Path) -> None:
        license_path = project_root / "LICENSE"
        assert license_path.exists()
        content = license_path.read_text()
        assert "MIT License" in content

    def test_readme_exists(self, project_root: Path) -> None:
        readme_path = project_root / "README.md"
        assert readme_path.exists()
        content = readme_path.read_text()
        assert "researchclaw" in content.lower()

    def test_readme_has_install_instructions(self, project_root: Path) -> None:
        content = (project_root / "README.md").read_text()
        assert "pipx install researchclaw" in content

    def test_readme_has_usage(self, project_root: Path) -> None:
        content = (project_root / "README.md").read_text()
        assert "researchclaw ." in content

    def test_init_has_version(self, project_root: Path) -> None:
        init_path = project_root / "src" / "researchclaw" / "__init__.py"
        content = init_path.read_text()
        assert "__version__" in content

    def test_main_module_exists(self, project_root: Path) -> None:
        main_path = project_root / "src" / "researchclaw" / "__main__.py"
        assert main_path.exists()

    def test_no_env_files_in_source(self, project_root: Path) -> None:
        src_dir = project_root / "src"
        env_files = list(src_dir.rglob(".env*"))
        assert len(env_files) == 0

    def test_no_secret_files_in_source(self, project_root: Path) -> None:
        src_dir = project_root / "src"
        secret_files = list(src_dir.rglob("*credential*"))
        secret_files += list(src_dir.rglob("*secret*"))
        assert len(secret_files) == 0


# --- _build_handlers includes MERGE_LOOP ---


class TestBuildHandlersIncludesMergeLoop:
    """Test that _build_handlers includes MERGE_LOOP handler."""

    def test_merge_loop_in_handlers(self) -> None:
        from researchclaw.cli import _build_handlers

        handlers = _build_handlers()
        assert State.MERGE_LOOP in handlers

    def test_all_states_except_internal_have_handlers(self) -> None:
        from researchclaw.cli import _build_handlers

        handlers = _build_handlers()
        # All states that are reachable should have handlers
        expected_states = {
            State.EXPERIMENT_PLAN,
            State.EXPERIMENT_IMPLEMENT,
            State.EXPERIMENT_EXECUTE,
            State.EVAL_IMPLEMENT,
            State.EVAL_EXECUTE,
            State.EXPERIMENT_REPORT,
            State.DECIDE,
            State.VIEW_SUMMARY,
            State.SETTINGS,
            State.MERGE_LOOP,
        }
        assert set(handlers.keys()) == expected_states
