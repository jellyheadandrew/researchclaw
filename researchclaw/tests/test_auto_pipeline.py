"""Tests for the automatic execution pipeline.

After PLAN approval, the system should auto-implement → auto-execute → auto-eval → report
without user intervention. These tests mock the AI worker to simulate the auto-pipeline.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from researchclaw.ai_worker import AIWorker
from researchclaw.config import Config
from researchclaw.messenger import QueueMessenger
from researchclaw.orchestrator import ResearchClaw
from researchclaw.states import State


def _agent_with_ai(tmp_path: Path) -> tuple[ResearchClaw, QueueMessenger]:
    """Create agent with planner_use_claude=True so ai_worker is instantiated."""
    cfg = Config(base_dir=str(tmp_path), messenger_type="stdio", planner_use_claude=True)
    messenger = QueueMessenger()
    agent = ResearchClaw(cfg, messenger=messenger)
    return agent, messenger


def _stub_implement_experiment_ok(plan_path: Path, codes_dir: Path) -> tuple[bool, str]:
    return True, (
        '### FILE: run.sh\n```\n#!/usr/bin/env bash\nset -euo pipefail\n'
        'echo "ok" > "$RC_OUTPUTS_DIR/out.txt"\n```\n\n'
        '### FILE: codes/main.py\n```\nprint("hello")\n```'
    )


def _stub_implement_eval_ok(
    plan_path: Path, codes_dir: Path, outputs_dir: Path, eval_codes_dir: Path
) -> tuple[bool, str]:
    return True, (
        '### FILE: eval.sh\n```\n#!/usr/bin/env bash\nset -euo pipefail\n'
        'echo "metric=1" > "$RC_RESULTS_DIR/metrics.txt"\n```\n\n'
        '### FILE: eval_codes/check.py\n```\nprint("eval")\n```'
    )


def _stub_implement_experiment_fail(*args, **kwargs) -> tuple[bool, str]:
    return False, "AI unavailable"


def _stub_implement_eval_fail(*args, **kwargs) -> tuple[bool, str]:
    return False, "AI unavailable"


def _stub_fix_experiment_ok(*args, **kwargs) -> tuple[bool, str]:
    return True, (
        '### FILE: codes/main.py\n```\nprint("fixed")\n```'
    )


def _setup_plan_and_approve(agent: ResearchClaw) -> None:
    """Navigate to PLAN, select scratch, and approve a plan (bypassing AI planner)."""
    agent.handle_message("/plan")
    agent.handle_message("/plan project scratch")
    agent.handle_message("Run a simple experiment")
    agent.handle_message("/plan approve")


def test_auto_pipeline_full_success(tmp_path: Path) -> None:
    """After plan approval, AI generates code, experiment runs, eval runs, report generated."""
    agent, messenger = _agent_with_ai(tmp_path)

    # Patch AI worker methods
    agent.ai_worker.implement_experiment = _stub_implement_experiment_ok
    agent.ai_worker.implement_eval = _stub_implement_eval_ok
    # Planner.update_plan uses CLI too — patch to return fallback
    agent.planner.use_claude = False

    _setup_plan_and_approve(agent)

    # Should flow all the way to DECIDE automatically
    assert agent.state == State.DECIDE

    # Verify run.sh was generated
    sandbox = tmp_path / "sandbox"
    run_scripts = list(sandbox.rglob("run.sh"))
    assert run_scripts, "run.sh should exist"
    content = run_scripts[0].read_text(encoding="utf-8")
    assert "RC_OUTPUTS_DIR" in content

    # Verify eval.sh was generated
    eval_scripts = list(sandbox.rglob("eval.sh"))
    assert eval_scripts, "eval.sh should exist"

    # Verify codes/ files were created
    code_files = list(sandbox.rglob("codes/main.py"))
    assert code_files, "codes/main.py should be created by AI"

    # Verify eval_codes/ files were created
    eval_files = list(sandbox.rglob("eval_codes/check.py"))
    assert eval_files, "eval_codes/check.py should be created by AI"

    # Verify report
    reports = list((tmp_path / "results").rglob("REPORT.md"))
    assert reports, "REPORT.md should be generated"

    # Verify experiment logs
    logs = (tmp_path / "EXPERIMENT_LOGS.md").read_text(encoding="utf-8")
    assert "trial_" in logs


def test_auto_implement_experiment_ai_failure_falls_back(tmp_path: Path) -> None:
    """If AI implementation fails, agent stays in EXPERIMENT_IMPLEMENT for manual fallback."""
    agent, messenger = _agent_with_ai(tmp_path)

    agent.ai_worker.implement_experiment = _stub_implement_experiment_fail
    agent.planner.use_claude = False

    _setup_plan_and_approve(agent)

    # AI failed — should stay in EXPERIMENT_IMPLEMENT for manual /write fallback
    assert agent.state == State.EXPERIMENT_IMPLEMENT
    assert any("failed" in m.lower() or "unavailable" in m.lower() for m in messenger.sent)


def test_auto_implement_eval_ai_failure_falls_back(tmp_path: Path) -> None:
    """If AI eval implementation fails, agent stays in EVAL_IMPLEMENT for manual fallback."""
    agent, messenger = _agent_with_ai(tmp_path)

    agent.ai_worker.implement_experiment = _stub_implement_experiment_ok
    agent.ai_worker.implement_eval = _stub_implement_eval_fail
    agent.planner.use_claude = False

    _setup_plan_and_approve(agent)

    # Experiment should succeed, but eval AI fails → stays in EVAL_IMPLEMENT
    assert agent.state == State.EVAL_IMPLEMENT
    assert any("eval" in m.lower() and "failed" in m.lower() for m in messenger.sent)


def test_ralph_loop_experiment_retry(tmp_path: Path) -> None:
    """Experiment failure triggers Ralph loop (fresh AI fix), then retries."""
    agent, messenger = _agent_with_ai(tmp_path)
    agent.settings.experiment_max_iterations = 3
    agent.storage.save_settings(agent.settings)

    call_count = {"implement": 0}

    def _impl_experiment_first_run_bad(plan_path, codes_dir):
        call_count["implement"] += 1
        # Produce a run.sh that fails (exit 1)
        return True, (
            '### FILE: run.sh\n```\n#!/usr/bin/env bash\nset -euo pipefail\nexit 1\n```\n\n'
            '### FILE: codes/main.py\n```\nprint("v1")\n```'
        )

    def _fix_experiment_to_succeed(*args, **kwargs):
        # Fix produces a run.sh that succeeds
        return True, (
            '### FILE: run.sh\n```\n#!/usr/bin/env bash\nset -euo pipefail\n'
            'echo "fixed" > "$RC_OUTPUTS_DIR/out.txt"\n```'
        )

    agent.ai_worker.implement_experiment = _impl_experiment_first_run_bad
    agent.ai_worker.fix_experiment = _fix_experiment_to_succeed
    agent.ai_worker.implement_eval = _stub_implement_eval_ok
    agent.planner.use_claude = False

    _setup_plan_and_approve(agent)

    # Ralph loop should fix the experiment and continue to eval → report → DECIDE
    assert agent.state == State.DECIDE
    assert any("RALPH LOOP" in m for m in messenger.sent)


def test_ralph_loop_max_iterations_goes_to_report(tmp_path: Path) -> None:
    """If Ralph loop hits max iterations, it goes to REPORT_SUMMARY → DECIDE."""
    agent, messenger = _agent_with_ai(tmp_path)
    agent.settings.experiment_max_iterations = 1
    agent.storage.save_settings(agent.settings)

    def _impl_experiment_always_fails(plan_path, codes_dir):
        return True, (
            '### FILE: run.sh\n```\n#!/usr/bin/env bash\nset -euo pipefail\nexit 1\n```'
        )

    agent.ai_worker.implement_experiment = _impl_experiment_always_fails
    agent.planner.use_claude = False

    _setup_plan_and_approve(agent)

    # Max iterations reached → report → DECIDE
    assert agent.state == State.DECIDE
    reports = list((tmp_path / "results").rglob("REPORT.md"))
    assert reports, "Report should still be generated on max iterations"


def test_auto_implement_writes_through_authority(tmp_path: Path) -> None:
    """AI-generated files go through authority.validate_write_path."""
    agent, messenger = _agent_with_ai(tmp_path)

    agent.ai_worker.implement_experiment = _stub_implement_experiment_ok
    agent.ai_worker.implement_eval = _stub_implement_eval_ok
    agent.planner.use_claude = False

    original_validate = agent.authority.validate_write_path
    validated_paths: list[Path] = []

    def _tracking_validate(state, path, trial, **kwargs):
        validated_paths.append(path)
        return original_validate(state, path, trial, **kwargs)

    agent.authority.validate_write_path = _tracking_validate

    _setup_plan_and_approve(agent)

    # Should have validated writes for run.sh, codes/main.py, eval.sh, eval_codes/check.py
    assert len(validated_paths) >= 4


def test_no_user_confirmation_during_auto_execution(tmp_path: Path) -> None:
    """Verify messenger.confirm is never called during automatic execution."""
    agent, messenger = _agent_with_ai(tmp_path)

    agent.ai_worker.implement_experiment = _stub_implement_experiment_ok
    agent.ai_worker.implement_eval = _stub_implement_eval_ok
    agent.planner.use_claude = False

    original_confirm = messenger.confirm
    confirm_calls = []

    def _tracking_confirm(prompt):
        confirm_calls.append(prompt)
        return original_confirm(prompt)

    messenger.confirm = _tracking_confirm

    _setup_plan_and_approve(agent)

    # No confirmation should have been requested
    assert confirm_calls == [], "confirm() should not be called during auto-execution"


def test_parse_file_patches() -> None:
    """Test AIWorker.parse_file_patches correctly parses ### FILE format."""
    output = (
        "### FILE: run.sh\n"
        "```\n"
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'echo "test"\n'
        "```\n"
        "\n"
        "### FILE: codes/main.py\n"
        "```\n"
        "print('hello')\n"
        "```\n"
    )
    patches = AIWorker.parse_file_patches(output)
    assert "run.sh" in patches
    assert "codes/main.py" in patches
    assert "set -euo pipefail" in patches["run.sh"]
    assert "print('hello')" in patches["codes/main.py"]


def test_parse_file_patches_no_changes() -> None:
    """NO_CHANGES_NEEDED output returns empty patches."""
    output = "NO_CHANGES_NEEDED: The code is correct as-is."
    patches = AIWorker.parse_file_patches(output)
    assert patches == {}
