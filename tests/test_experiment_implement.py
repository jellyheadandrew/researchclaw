from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.experiment import (
    CODING_AGENT_SYSTEM,
    _load_template,
    _parse_llm_response,
    _placeholder_code,
    _render_run_exp_sh,
    _write_experiment_files,
    handle_experiment_implement,
)
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager

import researchclaw.fsm.experiment as experiment_mod


# --- Fake helpers ---


class FakeChat:
    """Fake chat interface with pre-programmed responses."""

    def __init__(self, responses: list[ChatInput] | None = None) -> None:
        self.responses = list(responses) if responses else []
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def send_image(self, path: str, caption: str | None = None) -> None:
        pass

    def receive(self) -> ChatInput:
        if not self.responses:
            raise SystemExit("No more responses")
        return self.responses.pop(0)


class FakeProvider:
    """Fake LLM provider that returns pre-configured responses."""

    def __init__(
        self, responses: list[str] | None = None, error: Exception | None = None
    ) -> None:
        self._responses = list(responses) if responses else []
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        self.calls.append({"messages": list(messages), "system": system})
        if self._error:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return "### REQUIREMENTS\nNONE\n\n### CODE\nprint('hello')\n"

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Any:
        yield self.chat(messages, system)


def _setup_sandbox(project_dir: Path) -> Path:
    """Initialize sandbox and create a trial, return trial dir."""
    SandboxManager.initialize(project_dir)
    return SandboxManager.create_trial(project_dir)


# --- Tests for _parse_llm_response ---


class TestParseLLMResponse:
    def test_both_sections(self) -> None:
        response = (
            "### REQUIREMENTS\nnumpy\npandas\n\n### CODE\nprint('hello')\n"
        )
        requirements, code = _parse_llm_response(response)
        assert "numpy" in requirements
        assert "pandas" in requirements
        assert "print('hello')" in code

    def test_requirements_none(self) -> None:
        response = "### REQUIREMENTS\nNONE\n\n### CODE\nprint('hello')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('hello')" in code

    def test_only_code_section(self) -> None:
        response = "### CODE\nprint('hello')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('hello')" in code

    def test_no_sections_treats_as_code(self) -> None:
        response = "print('hello world')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('hello world')" in code

    def test_code_with_backticks(self) -> None:
        response = "### REQUIREMENTS\nNONE\n\n### CODE\n```python\nprint('hello')\n```"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('hello')" in code
        assert "```" not in code

    def test_requirements_with_versions(self) -> None:
        response = "### REQUIREMENTS\nnumpy>=1.24\ntorch==2.0\n\n### CODE\nimport numpy\n"
        requirements, code = _parse_llm_response(response)
        assert "numpy>=1.24" in requirements
        assert "torch==2.0" in requirements

    def test_case_insensitive_markers(self) -> None:
        response = "### Requirements\nNone\n\n### Code\nprint('hi')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('hi')" in code

    def test_requirements_filters_headers(self) -> None:
        response = "### REQUIREMENTS\n# comment line\nnumpy\n\n### CODE\nprint('hi')\n"
        requirements, code = _parse_llm_response(response)
        assert "# comment line" not in requirements
        assert "numpy" in requirements

    def test_empty_response(self) -> None:
        requirements, code = _parse_llm_response("")
        assert requirements == ""
        assert code == ""

    def test_code_with_triple_backtick_python(self) -> None:
        """Code wrapped in ```python ... ``` gets cleaned up."""
        response = "```python\nprint('hello')\n```"
        requirements, code = _parse_llm_response(response)
        assert "print('hello')" in code
        assert "```" not in code


# --- Tests for _render_run_exp_sh ---


class TestRenderRunExpSh:
    def test_renders_with_trial_dir(self, tmp_path: Path) -> None:
        config = ResearchClawConfig(python_command="python3")
        result = _render_run_exp_sh(tmp_path, config)
        assert str(tmp_path) in result
        assert "python3" in result

    def test_renders_with_custom_python(self, tmp_path: Path) -> None:
        config = ResearchClawConfig(python_command="/usr/bin/python3.11")
        result = _render_run_exp_sh(tmp_path, config)
        assert "/usr/bin/python3.11" in result

    def test_contains_venv_creation(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_exp_sh(tmp_path, config)
        assert "venv" in result.lower()

    def test_contains_pip_install(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_exp_sh(tmp_path, config)
        assert "pip" in result

    def test_contains_main_py_execution(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_exp_sh(tmp_path, config)
        assert "main.py" in result

    def test_is_bash_script(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_exp_sh(tmp_path, config)
        assert result.startswith("#!/usr/bin/env bash")


# --- Tests for _load_template ---


class TestLoadTemplate:
    def test_template_loads(self) -> None:
        template = _load_template()
        assert template is not None
        assert template.name == "run_exp.sh.jinja2"


# --- Tests for _write_experiment_files ---


class TestWriteExperimentFiles:
    def test_writes_main_py(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_exp").mkdir(parents=True)
        (trial_dir / "experiment").mkdir(exist_ok=True)
        config = ResearchClawConfig()
        _write_experiment_files(trial_dir, "print('hi')", "", config)
        assert (trial_dir / "experiment" / "codes_exp" / "main.py").exists()
        assert "print('hi')" in (trial_dir / "experiment" / "codes_exp" / "main.py").read_text()

    def test_writes_requirements(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_exp").mkdir(parents=True)
        config = ResearchClawConfig()
        _write_experiment_files(trial_dir, "code", "numpy\npandas", config)
        req = (trial_dir / "requirements.txt").read_text()
        assert "numpy" in req
        assert "pandas" in req

    def test_writes_empty_requirements(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_exp").mkdir(parents=True)
        config = ResearchClawConfig()
        _write_experiment_files(trial_dir, "code", "", config)
        assert (trial_dir / "requirements.txt").read_text() == ""

    def test_writes_run_exp_sh(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_exp").mkdir(parents=True)
        config = ResearchClawConfig()
        _write_experiment_files(trial_dir, "code", "", config)
        script = trial_dir / "experiment" / "run_exp.sh"
        assert script.exists()
        assert "main.py" in script.read_text()

    def test_run_exp_sh_is_executable(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_exp").mkdir(parents=True)
        config = ResearchClawConfig()
        _write_experiment_files(trial_dir, "code", "", config)
        script = trial_dir / "experiment" / "run_exp.sh"
        assert script.stat().st_mode & stat.S_IEXEC

    def test_creates_codes_dir_if_missing(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        config = ResearchClawConfig()
        _write_experiment_files(trial_dir, "code", "", config)
        assert (trial_dir / "experiment" / "codes_exp" / "main.py").exists()


# --- Tests for _placeholder_code ---


class TestPlaceholderCode:
    def test_generates_valid_python(self) -> None:
        code = _placeholder_code("test plan")
        assert "def main" in code
        assert '__name__' in code
        assert "placeholder" in code.lower()

    def test_includes_plan_snippet(self) -> None:
        code = _placeholder_code("My experiment: do something cool")
        assert "My experiment" in code

    def test_truncates_long_plan(self) -> None:
        long_plan = "A" * 500
        code = _placeholder_code(long_plan)
        # Should only include first 200 chars
        assert "A" * 200 in code
        assert "A" * 201 not in code


# --- Tests for handle_experiment_implement ---


class TestHandleExperimentImplement:
    def test_returns_experiment_execute(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan\nDo stuff.")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        result = handle_experiment_implement(trial_dir, meta, config, chat)
        assert result == State.EXPERIMENT_EXECUTE

    def test_writes_main_py(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_experiment_implement(trial_dir, meta, config, chat)
        assert (trial_dir / "experiment" / "codes_exp" / "main.py").exists()

    def test_writes_run_exp_sh(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_experiment_implement(trial_dir, meta, config, chat)
        script = trial_dir / "experiment" / "run_exp.sh"
        assert script.exists()
        assert script.stat().st_mode & stat.S_IEXEC

    def test_sends_status_messages(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_experiment_implement(trial_dir, meta, config, chat)

        assert any("EXPERIMENT_IMPLEMENT" in m for m in chat.sent)
        assert any("codes_exp/main.py" in m for m in chat.sent)

    def test_with_llm_provider(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan\nClassify images.")
        provider = FakeProvider(responses=[
            "### REQUIREMENTS\nnumpy\n\n### CODE\nimport numpy\nprint('experiment')\n"
        ])
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        result = handle_experiment_implement(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_EXECUTE
        code = (trial_dir / "experiment" / "codes_exp" / "main.py").read_text()
        assert "import numpy" in code
        reqs = (trial_dir / "requirements.txt").read_text()
        assert "numpy" in reqs
        assert len(provider.calls) == 1

    def test_llm_system_prompt_includes_plan(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Unique Plan XYZ")
        provider = FakeProvider()
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        handle_experiment_implement(trial_dir, meta, config, chat)

        assert len(provider.calls) == 1
        assert "Unique Plan XYZ" in provider.calls[0]["system"]

    def test_llm_error_fallback_to_placeholder(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        provider = FakeProvider(error=RuntimeError("API down"))
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        result = handle_experiment_implement(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_EXECUTE
        code = (trial_dir / "experiment" / "codes_exp" / "main.py").read_text()
        assert "placeholder" in code.lower()

    def test_no_plan_file(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        # Don't create PLAN.md
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        result = handle_experiment_implement(trial_dir, meta, config, chat)

        assert result == State.EXPERIMENT_EXECUTE
        code = (trial_dir / "experiment" / "codes_exp" / "main.py").read_text()
        assert "No plan available" in code

    def test_none_chat_interface(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_experiment_implement(trial_dir, meta, config, None)

        assert result == State.EXPERIMENT_EXECUTE
        assert (trial_dir / "experiment" / "codes_exp" / "main.py").exists()

    def test_run_exp_sh_references_trial_dir(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(experiment_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_experiment_implement(trial_dir, meta, config, chat)

        script = (trial_dir / "experiment" / "run_exp.sh").read_text()
        assert str(trial_dir) in script

    def test_no_provider_message(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        chat = FakeChat(responses=[UserMessage("s")])
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(experiment_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        handle_experiment_implement(trial_dir, meta, config, chat)

        assert any("No LLM provider" in m for m in chat.sent)


# --- Tests for CODING_AGENT_SYSTEM prompt ---


class TestCodingAgentSystem:
    def test_has_plan_placeholder(self) -> None:
        assert "{plan_content}" in CODING_AGENT_SYSTEM

    def test_has_requirements_section(self) -> None:
        assert "REQUIREMENTS" in CODING_AGENT_SYSTEM

    def test_has_code_section(self) -> None:
        assert "CODE" in CODING_AGENT_SYSTEM

    def test_mentions_main_py(self) -> None:
        assert "main.py" in CODING_AGENT_SYSTEM
