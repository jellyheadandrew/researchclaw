from pathlib import Path

import click

from researchclaw import __version__
from researchclaw.fsm.engine import FSMEngine
from researchclaw.onboarding import needs_onboarding, run_onboarding
from researchclaw.repl import TerminalChat
from researchclaw.sandbox import SandboxManager


class ResearchClawGroup(click.Group):
    """Custom group that allows `researchclaw .` and `researchclaw status` to coexist.

    If the first argument matches a registered subcommand, it's treated as a subcommand.
    Otherwise, it's treated as a project_dir for the default FSM run behavior.
    """

    def parse_args(self, ctx, args):
        # If first arg exists and is NOT a known command, store it as project_dir
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            ctx.ensure_object(dict)
            ctx.obj["project_dir"] = args.pop(0)
        else:
            ctx.ensure_object(dict)
            ctx.obj.setdefault("project_dir", ".")
        return super().parse_args(ctx, args)


def _build_handlers():
    """Build the State -> handler mapping for the FSM engine."""
    from researchclaw.fsm.decide import handle_decide
    from researchclaw.fsm.evaluate import handle_eval_execute, handle_eval_implement
    from researchclaw.fsm.experiment import (
        handle_experiment_execute,
        handle_experiment_implement,
    )
    from researchclaw.fsm.merge import handle_merge_loop
    from researchclaw.fsm.plan import handle_experiment_plan
    from researchclaw.fsm.report import handle_experiment_report
    from researchclaw.fsm.settings import handle_settings
    from researchclaw.fsm.states import State
    from researchclaw.fsm.view_summary import handle_view_summary

    return {
        State.EXPERIMENT_PLAN: handle_experiment_plan,
        State.EXPERIMENT_IMPLEMENT: handle_experiment_implement,
        State.EXPERIMENT_EXECUTE: handle_experiment_execute,
        State.EVAL_IMPLEMENT: handle_eval_implement,
        State.EVAL_EXECUTE: handle_eval_execute,
        State.EXPERIMENT_REPORT: handle_experiment_report,
        State.DECIDE: handle_decide,
        State.VIEW_SUMMARY: handle_view_summary,
        State.SETTINGS: handle_settings,
        State.MERGE_LOOP: handle_merge_loop,
    }


@click.group(cls=ResearchClawGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="researchclaw")
@click.pass_context
def main(ctx, **kwargs):
    """ResearchClaw - Automated research experiment orchestrator."""
    if ctx.invoked_subcommand is None:
        project_dir = ctx.obj.get("project_dir", ".")
        project_path = Path(project_dir).resolve()
        click.echo(f"ResearchClaw v{__version__}")
        click.echo(f"Target: {project_path}")

        # Onboarding: if no global config, run first-time setup
        if needs_onboarding():
            chat = TerminalChat()
            run_onboarding(chat)

        if not SandboxManager.is_initialized(project_path):
            click.echo("Initializing sandbox...")
            SandboxManager.initialize(project_path)
            click.echo("Sandbox initialized.")

        # Load config to display autopilot status
        from researchclaw.config import ResearchClawConfig
        config = ResearchClawConfig.load_merged_config(project_path)
        autopilot_label = "ON" if config.autopilot else "OFF"
        click.echo(f"Autopilot: {autopilot_label}")

        # Resume logic: check latest trial state
        latest = SandboxManager.get_latest_trial(project_path)
        if latest is None:
            click.echo("No trials yet. Starting first trial...")
        else:
            meta = SandboxManager.get_trial_meta(latest)
            # If the latest trial is completed/terminated, start at DECIDE
            if meta.status in ("completed", "terminated"):
                click.echo(
                    f"Last trial {latest.name} ({meta.status}). "
                    f"Entering DECIDE state."
                )
                meta.state = "decide"
                SandboxManager.save_trial_meta(latest, meta)
            else:
                click.echo(
                    f"Resuming trial {latest.name}: "
                    f"state={meta.state}, status={meta.status}"
                )

        # Build handler map and run the FSM
        chat = TerminalChat()
        handlers = _build_handlers()
        engine = FSMEngine(project_path, config, chat, handlers)

        try:
            engine.run()
        except SystemExit:
            click.echo("Goodbye!")


@main.command()
@click.pass_context
def status(ctx):
    """Print status table of all trials."""
    from rich.console import Console

    from researchclaw.status import build_status_table

    project_dir = Path(ctx.obj.get("project_dir", ".")).resolve()
    if not SandboxManager.is_initialized(project_dir):
        click.echo("No sandbox found. Run 'researchclaw .' first to initialize.")
        return

    table = build_status_table(project_dir)
    console = Console()
    console.print(table)
