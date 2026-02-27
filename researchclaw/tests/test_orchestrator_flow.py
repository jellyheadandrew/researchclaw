from __future__ import annotations

from pathlib import Path

from researchclaw.config import Config
from researchclaw.messenger import QueueMessenger
from researchclaw.orchestrator import ResearchClawV2
from researchclaw.states import State


def _agent(tmp_path: Path) -> tuple[ResearchClawV2, QueueMessenger]:
    cfg = Config(base_dir=str(tmp_path), messenger_type="stdio", planner_use_claude=False)
    messenger = QueueMessenger()
    agent = ResearchClawV2(cfg, messenger=messenger)
    return agent, messenger


def test_happy_path_plan_to_report(tmp_path: Path) -> None:
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/plan")
    assert agent.state == State.PLAN

    agent.handle_message("/plan project scratch")
    agent.handle_message("Focus on basic smoke run")
    agent.handle_message("/plan approve")
    assert agent.state == State.EXPERIMENT_IMPLEMENT

    # run.sh
    agent.handle_message("/write run.sh")
    agent.handle_message("#!/usr/bin/env bash")
    agent.handle_message("set -euo pipefail")
    agent.handle_message('echo "ok" > "$RC_OUTPUTS_DIR/out.txt"')
    agent.handle_message("/endwrite")

    messenger.push("yes")
    agent.handle_message("/exp run")
    assert agent.state == State.EVAL_IMPLEMENT

    # eval.sh
    agent.handle_message("/write eval.sh")
    agent.handle_message("#!/usr/bin/env bash")
    agent.handle_message("set -euo pipefail")
    agent.handle_message('echo "metric=1" > "$RC_RESULTS_DIR/metrics.txt"')
    agent.handle_message("/endwrite")

    messenger.push("yes")
    agent.handle_message("/eval run")
    assert agent.state == State.DECIDE

    reports = list((tmp_path / "results").rglob("REPORT.md"))
    assert reports, "REPORT.md should be generated"
    logs = (tmp_path / "EXPERIMENT_LOGS.md").read_text(encoding="utf-8")
    assert "Full Doc:" in logs


def test_experiment_failure_hits_max_iterations(tmp_path: Path) -> None:
    agent, messenger = _agent(tmp_path)
    agent.settings.experiment_max_iterations = 1
    agent.storage.save_settings(agent.settings)

    agent.handle_message("/plan")
    agent.handle_message("/plan project scratch")
    agent.handle_message("failure test")
    agent.handle_message("/plan approve")

    agent.handle_message("/write run.sh")
    agent.handle_message("#!/usr/bin/env bash")
    agent.handle_message("set -euo pipefail")
    agent.handle_message("exit 1")
    agent.handle_message("/endwrite")

    messenger.push("yes")
    agent.handle_message("/exp run")
    assert agent.state == State.DECIDE

    logs = (tmp_path / "EXPERIMENT_LOGS.md").read_text(encoding="utf-8")
    assert "trial_" in logs
