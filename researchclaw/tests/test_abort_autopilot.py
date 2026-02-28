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


def test_abort_generates_terminated_report_and_autopilot_to_plan(tmp_path: Path) -> None:
    agent, messenger = _agent(tmp_path)

    messenger.push("yes")
    agent.handle_message("/autopilot-start")
    assert agent.settings.autopilot_enabled is True

    agent.handle_message("/plan")
    agent.handle_message("/plan project scratch")
    agent.handle_message("quick smoke test")
    agent.handle_message("/plan approve")
    assert agent.state == State.EXPERIMENT_IMPLEMENT

    messenger.push("yes")
    agent.handle_message("/abort stopping now")

    # autopilot auto-plans + auto-approves, landing at EXPERIMENT_IMPLEMENT
    assert agent.state == State.EXPERIMENT_IMPLEMENT
    report_files = list((tmp_path / "results").rglob("REPORT.md"))
    assert report_files
    content = report_files[0].read_text(encoding="utf-8")
    assert "[TERMINATED-DURING-EXPERIMENT]" in content


def test_abort_ignored_in_report_summary(tmp_path: Path) -> None:
    agent, _ = _agent(tmp_path)
    agent.state = State.REPORT_SUMMARY
    agent.handle_message("/abort")
    assert agent.state == State.REPORT_SUMMARY
