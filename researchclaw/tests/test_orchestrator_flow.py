from __future__ import annotations

from pathlib import Path

from researchclaw.config import Config
from researchclaw.messenger import QueueMessenger
from researchclaw.orchestrator import ResearchClaw
from researchclaw.states import State


def _agent(tmp_path: Path) -> tuple[ResearchClaw, QueueMessenger]:
    cfg = Config(base_dir=str(tmp_path), messenger_type="stdio", planner_use_claude=False)
    messenger = QueueMessenger()
    agent = ResearchClaw(cfg, messenger=messenger)
    return agent, messenger


def _setup_trial_with_plan(agent: ResearchClaw, messenger: QueueMessenger) -> None:
    """Navigate to PLAN, select scratch, provide an idea, and approve."""
    agent.handle_message("/plan")
    assert agent.state == State.PLAN

    agent.handle_message("/plan project scratch")
    agent.handle_message("Focus on basic smoke run")
    assert agent.plan_draft.strip()

    agent.handle_message("/plan approve")


def test_happy_path_plan_to_report(tmp_path: Path) -> None:
    """Full pipeline using /write fallback (no AI worker) + manual /exp run and /eval run."""
    agent, messenger = _agent(tmp_path)

    _setup_trial_with_plan(agent, messenger)
    # Without AI worker, auto-implement sends fallback message; user uses /write + /exp run.
    assert agent.state == State.EXPERIMENT_IMPLEMENT

    # Write run.sh manually via escape hatch
    agent.handle_message("/write run.sh")
    agent.handle_message("#!/usr/bin/env bash")
    agent.handle_message("set -euo pipefail")
    agent.handle_message('echo "ok" > "$RC_OUTPUTS_DIR/out.txt"')
    agent.handle_message("/endwrite")

    agent.handle_message("/exp run")
    assert agent.state == State.EVAL_IMPLEMENT

    # Write eval.sh manually via escape hatch
    agent.handle_message("/write eval.sh")
    agent.handle_message("#!/usr/bin/env bash")
    agent.handle_message("set -euo pipefail")
    agent.handle_message('echo "metric=1" > "$RC_RESULTS_DIR/metrics.txt"')
    agent.handle_message("/endwrite")

    agent.handle_message("/eval run")
    assert agent.state == State.DECIDE

    reports = list((tmp_path / "results").rglob("REPORT.md"))
    assert reports, "REPORT.md should be generated"
    logs = (tmp_path / "EXPERIMENT_LOGS.md").read_text(encoding="utf-8")
    assert "Full Doc:" in logs


def test_experiment_failure_hits_max_iterations(tmp_path: Path) -> None:
    """Experiment failure with max_iterations=1 and no AI worker goes to report then DECIDE."""
    agent, messenger = _agent(tmp_path)
    agent.settings.experiment_max_iterations = 1
    agent.storage.save_settings(agent.settings)

    _setup_trial_with_plan(agent, messenger)
    assert agent.state == State.EXPERIMENT_IMPLEMENT

    # Write a failing run.sh
    agent.handle_message("/write run.sh")
    agent.handle_message("#!/usr/bin/env bash")
    agent.handle_message("set -euo pipefail")
    agent.handle_message("exit 1")
    agent.handle_message("/endwrite")

    agent.handle_message("/exp run")
    # No AI worker → max iterations → report → DECIDE
    assert agent.state == State.DECIDE

    logs = (tmp_path / "EXPERIMENT_LOGS.md").read_text(encoding="utf-8")
    assert "trial_" in logs


def test_plan_natural_language_approval(tmp_path: Path) -> None:
    """Approve plan using natural language instead of /plan approve."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/plan")
    agent.handle_message("/plan project scratch")
    agent.handle_message("Do a simple test")
    agent.handle_message("looks good")
    assert agent.state == State.EXPERIMENT_IMPLEMENT


def test_plan_natural_language_project_selection(tmp_path: Path) -> None:
    """Select project via natural language ('scratch')."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/plan")
    agent.handle_message("start from scratch")
    assert agent.plan_project_selected is True
