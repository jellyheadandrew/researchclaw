from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from conftest import FakeChat
from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.evaluate import (
    EVAL_CODING_AGENT_SYSTEM,
    _load_eval_template,
    _parse_llm_response,
    _placeholder_eval_code,
    _render_run_eval_sh,
    _write_eval_files,
    handle_eval_implement,
)
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.repl import ChatInput, SlashCommand, UserMessage
from researchclaw.sandbox import SandboxManager

import researchclaw.fsm.evaluate as evaluate_mod


# --- Fake helpers ---


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
        return "### REQUIREMENTS\nNONE\n\n### CODE\nprint('eval result')\n"

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
            "### REQUIREMENTS\nmatplotlib\nscipy\n\n### CODE\nimport matplotlib\nprint('eval')\n"
        )
        requirements, code = _parse_llm_response(response)
        assert "matplotlib" in requirements
        assert "scipy" in requirements
        assert "import matplotlib" in code

    def test_requirements_none(self) -> None:
        response = "### REQUIREMENTS\nNONE\n\n### CODE\nprint('eval')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('eval')" in code

    def test_only_code_section(self) -> None:
        response = "### CODE\nprint('eval result')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('eval result')" in code

    def test_no_sections_treats_as_code(self) -> None:
        response = "print('eval output')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('eval output')" in code

    def test_code_with_backticks(self) -> None:
        response = "### REQUIREMENTS\nNONE\n\n### CODE\n```python\nprint('eval')\n```"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('eval')" in code
        assert "```" not in code

    def test_case_insensitive_markers(self) -> None:
        response = "### Requirements\nNone\n\n### Code\nprint('hi')\n"
        requirements, code = _parse_llm_response(response)
        assert requirements == ""
        assert "print('hi')" in code

    def test_empty_response(self) -> None:
        requirements, code = _parse_llm_response("")
        assert requirements == ""
        assert code == ""


# --- Tests for _render_run_eval_sh ---


class TestRenderRunEvalSh:
    def test_renders_with_trial_dir(self, tmp_path: Path) -> None:
        config = ResearchClawConfig(python_command="python3")
        result = _render_run_eval_sh(tmp_path, config)
        assert str(tmp_path) in result
        assert "python3" in result

    def test_renders_with_custom_python(self, tmp_path: Path) -> None:
        config = ResearchClawConfig(python_command="/usr/bin/python3.11")
        result = _render_run_eval_sh(tmp_path, config)
        assert "/usr/bin/python3.11" in result

    def test_contains_venv_creation(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_eval_sh(tmp_path, config)
        assert "venv" in result.lower()

    def test_contains_codes_eval_dir(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_eval_sh(tmp_path, config)
        assert "codes_eval" in result

    def test_contains_main_py_execution(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_eval_sh(tmp_path, config)
        assert "main.py" in result

    def test_is_bash_script(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_eval_sh(tmp_path, config)
        assert result.startswith("#!/usr/bin/env bash")

    def test_passes_outputs_dir_env(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_eval_sh(tmp_path, config)
        assert "OUTPUTS_DIR" in result

    def test_passes_trial_dir_env(self, tmp_path: Path) -> None:
        config = ResearchClawConfig()
        result = _render_run_eval_sh(tmp_path, config)
        assert "TRIAL_DIR" in result


# --- Tests for _load_eval_template ---


class TestLoadEvalTemplate:
    def test_template_loads(self) -> None:
        template = _load_eval_template()
        assert template is not None
        assert template.name == "eval.sh.jinja2"


# --- Tests for _write_eval_files ---


class TestWriteEvalFiles:
    def test_writes_main_py(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_eval").mkdir(parents=True)
        config = ResearchClawConfig()
        _write_eval_files(trial_dir, "print('eval')", "", config)
        assert (trial_dir / "experiment" / "codes_eval" / "main.py").exists()
        assert "print('eval')" in (trial_dir / "experiment" / "codes_eval" / "main.py").read_text()

    def test_writes_run_eval_sh(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_eval").mkdir(parents=True)
        config = ResearchClawConfig()
        _write_eval_files(trial_dir, "code", "", config)
        script = trial_dir / "experiment" / "run_eval.sh"
        assert script.exists()
        assert "main.py" in script.read_text()

    def test_run_eval_sh_is_executable(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_eval").mkdir(parents=True)
        config = ResearchClawConfig()
        _write_eval_files(trial_dir, "code", "", config)
        script = trial_dir / "experiment" / "run_eval.sh"
        assert script.stat().st_mode & stat.S_IEXEC

    def test_creates_codes_dir_if_missing(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        config = ResearchClawConfig()
        _write_eval_files(trial_dir, "code", "", config)
        assert (trial_dir / "experiment" / "codes_eval" / "main.py").exists()

    def test_appends_eval_requirements(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_eval").mkdir(parents=True)
        (trial_dir / "requirements.txt").write_text("numpy\n")
        config = ResearchClawConfig()
        _write_eval_files(trial_dir, "code", "matplotlib", config)
        reqs = (trial_dir / "requirements.txt").read_text()
        assert "numpy" in reqs
        assert "matplotlib" in reqs

    def test_no_duplicate_requirements(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial"
        (trial_dir / "experiment" / "codes_eval").mkdir(parents=True)
        (trial_dir / "requirements.txt").write_text("numpy\n")
        config = ResearchClawConfig()
        _write_eval_files(trial_dir, "code", "numpy", config)
        reqs = (trial_dir / "requirements.txt").read_text()
        assert reqs.count("numpy") == 1


# --- Tests for _placeholder_eval_code ---


class TestPlaceholderEvalCode:
    def test_generates_valid_python(self) -> None:
        code = _placeholder_eval_code("test plan")
        assert "def main" in code
        assert '__name__' in code
        assert "placeholder" in code.lower()

    def test_includes_plan_snippet(self) -> None:
        code = _placeholder_eval_code("My evaluation: analyze results")
        assert "My evaluation" in code

    def test_truncates_long_plan(self) -> None:
        long_plan = "B" * 500
        code = _placeholder_eval_code(long_plan)
        assert "B" * 200 in code
        assert "B" * 201 not in code

    def test_uses_outputs_dir_env(self) -> None:
        code = _placeholder_eval_code("plan")
        assert "OUTPUTS_DIR" in code


# --- Tests for handle_eval_implement ---


class TestHandleEvalImplement:
    def test_returns_eval_execute(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan\nEvaluate stuff.")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        result = handle_eval_implement(trial_dir, meta, config, chat)
        assert result == State.EVAL_EXECUTE

    def test_writes_main_py(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_eval_implement(trial_dir, meta, config, chat)
        assert (trial_dir / "experiment" / "codes_eval" / "main.py").exists()

    def test_writes_run_eval_sh(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_eval_implement(trial_dir, meta, config, chat)
        script = trial_dir / "experiment" / "run_eval.sh"
        assert script.exists()
        assert script.stat().st_mode & stat.S_IEXEC

    def test_sends_status_messages(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_eval_implement(trial_dir, meta, config, chat)

        assert any("EVAL_IMPLEMENT" in m for m in chat.sent)
        assert any("codes_eval/main.py" in m for m in chat.sent)

    def test_with_llm_provider(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Test Plan\nAnalyze outputs.")
        provider = FakeProvider(responses=[
            "### REQUIREMENTS\nmatplotlib\n\n### CODE\nimport matplotlib\nprint('eval result')\n"
        ])
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        result = handle_eval_implement(trial_dir, meta, config, chat)

        assert result == State.EVAL_EXECUTE
        code = (trial_dir / "experiment" / "codes_eval" / "main.py").read_text()
        assert "import matplotlib" in code
        assert len(provider.calls) == 1

    def test_llm_system_prompt_includes_plan(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Unique Eval Plan ABC")
        provider = FakeProvider()
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        handle_eval_implement(trial_dir, meta, config, chat)

        assert len(provider.calls) == 1
        assert "Unique Eval Plan ABC" in provider.calls[0]["system"]

    def test_llm_system_prompt_includes_outputs_dir(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        provider = FakeProvider()
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]

        handle_eval_implement(trial_dir, meta, config, chat)

        assert "outputs" in provider.calls[0]["system"]

    def test_llm_error_fallback_to_placeholder(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        provider = FakeProvider(error=RuntimeError("API down"))
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: provider)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        result = handle_eval_implement(trial_dir, meta, config, chat)

        assert result == State.EVAL_EXECUTE
        code = (trial_dir / "experiment" / "codes_eval" / "main.py").read_text()
        assert "placeholder" in code.lower()

    def test_no_plan_file(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        result = handle_eval_implement(trial_dir, meta, config, chat)

        assert result == State.EVAL_EXECUTE
        code = (trial_dir / "experiment" / "codes_eval" / "main.py").read_text()
        assert "No plan available" in code

    def test_none_chat_interface(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]

        result = handle_eval_implement(trial_dir, meta, config, None)

        assert result == State.EVAL_EXECUTE
        assert (trial_dir / "experiment" / "codes_eval" / "main.py").exists()

    def test_no_provider_message(self, tmp_path: Path, monkeypatch: object) -> None:
        trial_dir = _setup_sandbox(tmp_path)
        (trial_dir / "PLAN.md").write_text("# Plan")
        chat = FakeChat()
        config = ResearchClawConfig()
        meta = TrialMeta()

        monkeypatch.setattr(evaluate_mod, "_get_provider_safe", lambda cfg: None)  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluate_mod, "_prompt_llm_unavailable", lambda ci, msg: "skip")  # type: ignore[attr-defined]

        handle_eval_implement(trial_dir, meta, config, chat)

        assert any("No LLM provider" in m or "EVAL_IMPLEMENT" in m for m in chat.sent)


# --- Tests for EVAL_CODING_AGENT_SYSTEM prompt ---


class TestEvalCodingAgentSystem:
    def test_has_plan_placeholder(self) -> None:
        assert "{plan_content}" in EVAL_CODING_AGENT_SYSTEM

    def test_has_requirements_section(self) -> None:
        assert "REQUIREMENTS" in EVAL_CODING_AGENT_SYSTEM

    def test_has_code_section(self) -> None:
        assert "CODE" in EVAL_CODING_AGENT_SYSTEM

    def test_mentions_main_py(self) -> None:
        assert "main.py" in EVAL_CODING_AGENT_SYSTEM

    def test_mentions_outputs_dir(self) -> None:
        assert "OUTPUTS_DIR" in EVAL_CODING_AGENT_SYSTEM

    def test_mentions_trial_dir(self) -> None:
        assert "TRIAL_DIR" in EVAL_CODING_AGENT_SYSTEM

    def test_mentions_visualizations(self) -> None:
        assert "visualization" in EVAL_CODING_AGENT_SYSTEM.lower()
