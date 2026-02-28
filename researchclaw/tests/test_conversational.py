"""Tests for conversational UX — natural language intent detection and routing."""
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


# ------------------------------------------------------------------
# DECIDE intent classification
# ------------------------------------------------------------------

def test_decide_intent_plan(tmp_path: Path) -> None:
    """Natural language 'plan a new experiment' routes to PLAN."""
    agent, messenger = _agent(tmp_path)
    assert agent.state == State.DECIDE

    agent.handle_message("plan a new experiment")
    assert agent.state == State.PLAN


def test_decide_intent_view_summary(tmp_path: Path) -> None:
    """Natural language 'show me past results' routes to VIEW_SUMMARY."""
    agent, messenger = _agent(tmp_path)
    assert agent.state == State.DECIDE

    agent.handle_message("show me past results")
    assert agent.state == State.VIEW_SUMMARY


def test_decide_intent_research(tmp_path: Path) -> None:
    """Natural language 'explore research ideas' routes to RESEARCH."""
    agent, messenger = _agent(tmp_path)
    assert agent.state == State.DECIDE

    agent.handle_message("explore research ideas")
    assert agent.state == State.RESEARCH


def test_decide_intent_settings(tmp_path: Path) -> None:
    """Natural language 'change setting' routes to SETTINGS."""
    agent, messenger = _agent(tmp_path)
    assert agent.state == State.DECIDE

    agent.handle_message("change setting")
    assert agent.state == State.SETTINGS


def test_decide_intent_update_and_push(tmp_path: Path) -> None:
    """Natural language 'push code to my project' routes to UPDATE_AND_PUSH."""
    agent, messenger = _agent(tmp_path)
    assert agent.state == State.DECIDE

    agent.handle_message("push code to my project")
    assert agent.state == State.UPDATE_AND_PUSH


def test_decide_slash_commands(tmp_path: Path) -> None:
    """Slash commands still work in DECIDE."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/plan")
    assert agent.state == State.PLAN


# ------------------------------------------------------------------
# PLAN state: project selection + approval
# ------------------------------------------------------------------

def test_plan_natural_scratch_selection(tmp_path: Path) -> None:
    """Natural language 'from scratch' selects scratch project."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/plan")
    agent.handle_message("from scratch")
    assert agent.plan_project_selected is True


def test_plan_natural_scratch_variations(tmp_path: Path) -> None:
    """Various natural language scratch selections work."""
    for phrase in ["scratch", "start fresh", "no project"]:
        agent, messenger = _agent(tmp_path)
        agent.handle_message("/plan")
        agent.handle_message(phrase)
        assert agent.plan_project_selected is True, f"Failed for phrase: {phrase}"


def test_plan_numbered_project_selection(tmp_path: Path) -> None:
    """Numeric project selection works in PLAN state."""
    agent, messenger = _agent(tmp_path)

    # Create a project directory
    project_dir = Path(tmp_path) / "projects" / "myproject"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "README.md").write_text("# test\n", encoding="utf-8")

    agent.handle_message("/plan")
    agent.handle_message("1")  # select first project by number
    assert agent.plan_project_selected is True
    assert agent.current_trial.selected_project == "myproject"


def test_plan_named_project_selection(tmp_path: Path) -> None:
    """Project selection by name works in PLAN state."""
    agent, messenger = _agent(tmp_path)

    project_dir = Path(tmp_path) / "projects" / "demo"
    project_dir.mkdir(parents=True, exist_ok=True)

    agent.handle_message("/plan")
    agent.handle_message("demo")
    assert agent.plan_project_selected is True
    assert agent.current_trial.selected_project == "demo"


def test_plan_approval_phrases(tmp_path: Path) -> None:
    """Various approval phrases trigger plan approval."""
    for phrase in ["approve", "looks good", "go ahead", "lgtm", "ship it"]:
        agent, messenger = _agent(tmp_path)
        agent.handle_message("/plan")
        agent.handle_message("/plan project scratch")
        agent.handle_message("test idea")
        agent.handle_message(phrase)
        assert agent.state == State.EXPERIMENT_IMPLEMENT, f"Failed for phrase: {phrase}"


def test_plan_show_plan(tmp_path: Path) -> None:
    """Natural language 'show plan' displays the current plan."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/plan")
    agent.handle_message("/plan project scratch")
    agent.handle_message("build a simple model")
    messenger.sent.clear()

    agent.handle_message("show plan")
    # Should have sent the plan draft
    assert any(agent.plan_draft in m for m in messenger.sent)


# ------------------------------------------------------------------
# VIEW_SUMMARY conversational
# ------------------------------------------------------------------

def test_view_summary_natural_older(tmp_path: Path) -> None:
    """Natural language 'show older' works in VIEW_SUMMARY."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/view_summary")
    assert agent.state == State.VIEW_SUMMARY

    messenger.sent.clear()
    agent.handle_message("show me older trials")
    # Should respond (even if no dates available)
    assert len(messenger.sent) > 0


def test_view_summary_exit(tmp_path: Path) -> None:
    """'exit' or 'done' returns to DECIDE from VIEW_SUMMARY."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/view_summary")
    assert agent.state == State.VIEW_SUMMARY

    agent.handle_message("done")
    assert agent.state == State.DECIDE


# ------------------------------------------------------------------
# SETTINGS conversational
# ------------------------------------------------------------------

def test_settings_show(tmp_path: Path) -> None:
    """'/settings show' displays current settings."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/settings")
    assert agent.state == State.SETTINGS

    messenger.sent.clear()
    agent.handle_message("/settings show")
    assert len(messenger.sent) > 0


# ------------------------------------------------------------------
# Global commands work from any state
# ------------------------------------------------------------------

def test_status_command(tmp_path: Path) -> None:
    """/status works from any state."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/status")
    assert any("State=" in m for m in messenger.sent)


def test_help_command(tmp_path: Path) -> None:
    """/help works from any state."""
    agent, messenger = _agent(tmp_path)

    agent.handle_message("/help")
    assert any("help" in m.lower() or "Global" in m for m in messenger.sent)
