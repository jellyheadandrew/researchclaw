from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

from .config import Config
from .git_manager import GitManager, ensure_project_config
from .messenger import Messenger, get_messenger
from .models import Settings, TrialRecord
from .planner import PlanEngine
from .policy import AuthorityError, AuthorityManager
from .reporter import Reporter
from .states import State, TrialStatus
from .storage import StorageManager


class ResearchClawV2:
    def __init__(self, config: Config, messenger: Messenger | None = None):
        self.config = config
        self.storage = StorageManager(config.base_dir)
        self.storage.ensure_layout()

        self.settings: Settings = self.storage.load_settings()
        self.authority = AuthorityManager(config.base_dir)
        self.git = GitManager(config.base_dir, self.authority)
        self.reporter = Reporter(config.base_dir)
        self.planner = PlanEngine(
            use_claude=config.planner_use_claude,
            cli_path=config.planner_claude_cli_path,
            model=config.planner_model,
        )
        self.messenger = messenger or get_messenger(config.messenger_type)

        self.state: State = State.DECIDE
        self.current_trial: TrialRecord | None = None

        self.plan_draft: str = ""
        self.plan_project_selected: bool = False

        self.write_session: dict[str, Any] | None = None

        self.update_selected_project: str | None = None
        self.update_trial_candidates: list[TrialRecord] = []

        self.research_next_ping: datetime | None = None

        self._restore_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.messenger.send("ResearchClaw V2 online.")
        self._on_enter_state(self.state)
        while True:
            self._maybe_send_research_nudge()
            message = self.messenger.receive(timeout=1.0)
            if message is None:
                continue
            self.handle_message(message.strip())

    def _restore_state(self) -> None:
        session = self.storage.load_session()
        state_raw = str(session.get("state", State.DECIDE.value))
        try:
            restored_state = State(state_raw)
        except ValueError:
            restored_state = State.DECIDE

        trial_id = session.get("current_trial_id")
        trial = self.storage.get_trial(str(trial_id)) if trial_id else None

        # Non-recoverable transient state.
        if restored_state == State.REPORT_SUMMARY:
            restored_state = State.DECIDE

        trial_required_states = {
            State.PLAN,
            State.EXPERIMENT_IMPLEMENT,
            State.EXPERIMENT_EXECUTE,
            State.EVAL_IMPLEMENT,
            State.EVAL_EXECUTE,
            State.REPORT_SUMMARY,
        }
        if restored_state in trial_required_states and not trial:
            restored_state = State.DECIDE

        self.state = restored_state
        self.current_trial = trial

    def _transition(self, new_state: State, reason: str) -> None:
        self.state = new_state
        if self.current_trial:
            self.current_trial.state = new_state
            self.storage.append_trial_record(self.current_trial)
        self.storage.save_session(self.state, self.current_trial.trial_id if self.current_trial else None)

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    def handle_message(self, msg: str) -> None:
        if not msg:
            return

        # Active write capture mode
        if self.write_session is not None:
            self._handle_write_session(msg)
            return

        # Global commands
        lower = msg.lower()
        if lower in {"/status", "status"}:
            self._send_status()
            return
        if lower in {"/help", "help"}:
            self._send_help()
            return
        if lower == "/autopilot-start":
            self._cmd_autopilot_start()
            return
        if lower == "/autopilot-stop":
            self.settings.autopilot_enabled = False
            self.storage.save_settings(self.settings)
            self.messenger.send("Autopilot disabled.")
            return
        if lower.startswith("/abort"):
            self._cmd_abort(msg)
            return
        if lower == "/exit":
            self._handle_exit()
            return

        if self.state == State.REPORT_SUMMARY:
            self.messenger.send("REPORT_SUMMARY is non-interruptible. Please wait.")
            return

        if self.state == State.DECIDE:
            self._handle_decide(msg)
        elif self.state == State.PLAN:
            self._handle_plan(msg)
        elif self.state == State.EXPERIMENT_IMPLEMENT:
            self._handle_experiment_implement(msg)
        elif self.state == State.EXPERIMENT_EXECUTE:
            self.messenger.send("Experiment execution is in progress.")
        elif self.state == State.EVAL_IMPLEMENT:
            self._handle_eval_implement(msg)
        elif self.state == State.EVAL_EXECUTE:
            self.messenger.send("Evaluation execution is in progress.")
        elif self.state == State.VIEW_SUMMARY:
            self._handle_view_summary(msg)
        elif self.state == State.UPDATE_AND_PUSH:
            self._handle_update_and_push(msg)
        elif self.state == State.SETTINGS:
            self._handle_settings(msg)
        elif self.state == State.RESEARCH:
            self._handle_research(msg)

    # ------------------------------------------------------------------
    # Global commands
    # ------------------------------------------------------------------

    def _cmd_autopilot_start(self) -> None:
        approved = self.messenger.confirm(
            "Enable autopilot mode? It will skip DECIDE and go directly to PLAN after each trial summary."
        )
        if not approved:
            self.messenger.send("Autopilot unchanged.")
            return
        self.settings.autopilot_enabled = True
        self.storage.save_settings(self.settings)
        self.messenger.send("Autopilot enabled.")

    def _cmd_abort(self, msg: str) -> None:
        if self.state == State.REPORT_SUMMARY:
            self.messenger.send("Abort ignored in REPORT_SUMMARY.")
            return
        if not self.current_trial:
            self.messenger.send("No active trial to abort.")
            return

        approved = self.messenger.confirm("Abort current trial and finalize with terminated report?")
        if not approved:
            self.messenger.send("Abort cancelled.")
            return

        reason = msg[len("/abort") :].strip() or "aborted by user"
        self.current_trial.terminated = True
        self.current_trial.status = TrialStatus.TERMINATED
        self.current_trial.termination_reason = reason
        self.storage.append_trial_record(self.current_trial)
        self._run_report_summary("aborted by user")

    def _handle_exit(self) -> None:
        if self.state in {State.VIEW_SUMMARY, State.SETTINGS, State.RESEARCH, State.UPDATE_AND_PUSH}:
            self._transition(State.DECIDE, "exit")
            self._on_enter_state(State.DECIDE)
            return
        self.messenger.send("/exit is only available in VIEW_SUMMARY, SETTINGS, RESEARCH, UPDATE_AND_PUSH.")

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_decide(self, msg: str) -> None:
        lower = msg.lower()
        if lower in {"/plan", "plan"}:
            self._transition(State.PLAN, "user selected PLAN")
            self._on_enter_state(State.PLAN)
            return
        if lower in {"/view_summary", "view_summary", "summary"}:
            self._transition(State.VIEW_SUMMARY, "user selected VIEW_SUMMARY")
            self._on_enter_state(State.VIEW_SUMMARY)
            return
        if lower in {"/update_and_push", "update_and_push", "push"}:
            self._transition(State.UPDATE_AND_PUSH, "user selected UPDATE_AND_PUSH")
            self._on_enter_state(State.UPDATE_AND_PUSH)
            return
        if lower in {"/settings", "settings"}:
            self._transition(State.SETTINGS, "user selected SETTINGS")
            self._on_enter_state(State.SETTINGS)
            return
        if lower in {"/research", "research"}:
            self._transition(State.RESEARCH, "user selected RESEARCH")
            self._on_enter_state(State.RESEARCH)
            return

        self._show_decide_menu()

    def _handle_plan(self, msg: str) -> None:
        if not self.current_trial:
            self.current_trial = self.storage.create_trial(selected_project=None)
            self.storage.save_session(self.state, self.current_trial.trial_id)

        if msg.startswith("/plan project "):
            selection = msg[len("/plan project ") :].strip()
            self._select_plan_project(selection)
            return

        if msg == "/plan projects":
            self._show_projects_for_plan()
            return

        if msg == "/plan show":
            if self.plan_draft.strip():
                self.messenger.send(self.plan_draft)
            else:
                self.messenger.send("No plan draft yet. Send guidance text to start drafting.")
            return

        if msg == "/plan approve":
            if not self.plan_project_selected:
                self.messenger.send("Select a starting project first: /plan project <number|name|scratch>")
                return
            self.current_trial.plan_approved = True
            self.storage.append_trial_record(self.current_trial)
            self._transition(State.EXPERIMENT_IMPLEMENT, "plan approved")
            self._on_enter_state(State.EXPERIMENT_IMPLEMENT)
            return

        updated = self.planner.update_plan(self.plan_draft, msg, self.current_trial.selected_project)
        self.plan_draft = updated
        plan_path = self.storage.base_dir / self.current_trial.sandbox_path / "PLAN.md"
        try:
            validated = self.authority.validate_write_path(self.state, plan_path, self.current_trial)
            validated.parent.mkdir(parents=True, exist_ok=True)
            validated.write_text(updated + "\n", encoding="utf-8")
        except AuthorityError as e:
            self.messenger.send(f"Plan write blocked: {e}")
            return

        self.messenger.send("Updated plan draft:\n" + updated)

    def _handle_experiment_implement(self, msg: str) -> None:
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        if msg.startswith("/write "):
            path = msg[len("/write ") :].strip()
            self._start_write_session(path)
            return

        if msg == "/exp run":
            self._run_experiment_once()
            return

        if msg == "/exp status":
            self.messenger.send(
                f"Trial {self.current_trial.trial_name} | exp_iter={self.current_trial.experiment_iter}/{self.settings.experiment_max_iterations}"
            )
            return

        self.messenger.send(
            "EXPERIMENT_IMPLEMENT commands:\n"
            "- /write <path>  (paths under codes/ or run.sh)\n"
            "- /exp run\n"
            "- /exp status"
        )

    def _handle_eval_implement(self, msg: str) -> None:
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        if msg.startswith("/write "):
            path = msg[len("/write ") :].strip()
            self._start_write_session(path)
            return

        if msg == "/eval run":
            self._run_eval_once()
            return

        if msg == "/eval status":
            self.messenger.send(
                f"Trial {self.current_trial.trial_name} | eval_iter={self.current_trial.eval_iter}/{self.settings.eval_max_iterations}"
            )
            return

        self.messenger.send(
            "EVAL_IMPLEMENT commands:\n"
            "- /write <path>  (paths under eval_codes/ or eval.sh)\n"
            "- /eval run\n"
            "- /eval status"
        )

    def _handle_view_summary(self, msg: str) -> None:
        if msg == "/older":
            dates = self.storage.list_dates()
            if not dates:
                self.messenger.send("No trial dates available.")
                return
            self.messenger.send("Available dates:\n" + "\n".join(f"- {d}" for d in dates))
            return

        if msg.startswith("/date "):
            date = msg[len("/date ") :].strip()
            self._send_trials_for_date(date)
            return

        self.messenger.send("VIEW_SUMMARY commands: /older, /date YYYYMMDD, /exit")

    def _handle_update_and_push(self, msg: str) -> None:
        if msg.startswith("/project add-clone "):
            self._cmd_project_add_clone(msg)
            return
        if msg.startswith("/project add-init "):
            self._cmd_project_add_init(msg)
            return
        if msg.startswith("/update select "):
            selector = msg[len("/update select ") :].strip()
            self._select_update_project(selector)
            return
        if msg.startswith("/update trial "):
            selector = msg[len("/update trial ") :].strip()
            self._select_update_trial(selector)
            return

        self.messenger.send(
            "UPDATE_AND_PUSH commands:\n"
            "- /update select <number|name>\n"
            "- /project add-clone <name> <remote_url> [branch]\n"
            "- /project add-init <name> <remote_url> [branch]\n"
            "- /update trial <index|YYYYMMDD trial_XXX>\n"
            "- /exit"
        )

    def _handle_settings(self, msg: str) -> None:
        if msg == "/settings show":
            self._show_settings()
            return

        if msg == "/settings explain":
            self.messenger.send(
                "Settings fields:\n"
                "- experiment_max_iterations: max retries in EXPERIMENT loop\n"
                "- eval_max_iterations: max retries in EVAL loop\n"
                "- view_summary_page_size: count of recent trials shown\n"
                "- autopilot_enabled: skip DECIDE after report\n"
                "- research_cadence: ask/disabled/hourly/6h/daily\n"
                "- project git fields: remote_url/default_branch/auth_source"
            )
            return

        if msg.startswith("/settings set-project "):
            try:
                self.authority.assert_settings_mutate(self.state)
                _, _, _, project, field, value = msg.split(" ", 5)
            except ValueError:
                self.messenger.send("Usage: /settings set-project <project> <remote_url|default_branch|auth_source> <value>")
                return
            cfg = ensure_project_config(self.settings, project)
            if field not in {"remote_url", "default_branch", "auth_source"}:
                self.messenger.send("Invalid field. Use remote_url/default_branch/auth_source.")
                return
            setattr(cfg, field, value)
            self.storage.save_settings(self.settings)
            self.messenger.send(f"Updated project {project} {field}.")
            return

        if msg.startswith("/settings set "):
            try:
                self.authority.assert_settings_mutate(self.state)
                _, _, key, value = msg.split(" ", 3)
            except ValueError:
                self.messenger.send("Usage: /settings set <key> <value>")
                return

            if key in {"experiment_max_iterations", "eval_max_iterations", "view_summary_page_size"}:
                try:
                    setattr(self.settings, key, int(value))
                except ValueError:
                    self.messenger.send(f"{key} must be an integer.")
                    return
            elif key == "autopilot_enabled":
                setattr(self.settings, key, value.lower() in {"1", "true", "yes", "on"})
            elif key == "research_cadence":
                if value not in {"ask", "disabled", "hourly", "6h", "daily"}:
                    self.messenger.send("research_cadence must be ask/disabled/hourly/6h/daily")
                    return
                self.settings.research_cadence = value
            else:
                self.messenger.send("Unknown key.")
                return

            self.storage.save_settings(self.settings)
            self.messenger.send(f"Updated {key}.")
            return

        self.messenger.send(
            "SETTINGS commands:\n"
            "- /settings show\n"
            "- /settings explain\n"
            "- /settings set <key> <value>\n"
            "- /settings set-project <project> <field> <value>\n"
            "- /exit"
        )

    def _handle_research(self, msg: str) -> None:
        if msg.startswith("/write "):
            path = msg[len("/write ") :].strip()
            self._start_write_session(path)
            return

        if msg.startswith("/research cadence "):
            cadence = msg[len("/research cadence ") :].strip()
            if cadence not in {"ask", "disabled", "hourly", "6h", "daily"}:
                self.messenger.send("Cadence must be ask/disabled/hourly/6h/daily")
                return
            self.settings.research_cadence = cadence
            self.storage.save_settings(self.settings)
            self.research_next_ping = self._calc_next_ping(cadence)
            self.messenger.send(f"Research cadence set to {cadence}.")
            return

        if msg.startswith("/research brainstorm "):
            note = msg[len("/research brainstorm ") :].strip()
            if not note:
                self.messenger.send("Provide note text.")
                return
            self._append_brainstorm(note)
            self.messenger.send("Brainstorm note appended.")
            return

        if msg.startswith("/research download "):
            url = msg[len("/research download ") :].strip()
            if not url:
                self.messenger.send("Provide URL.")
                return
            self._download_reference(url)
            return

        self.messenger.send(
            "RESEARCH commands:\n"
            "- /research cadence <ask|disabled|hourly|6h|daily>\n"
            "- /research brainstorm <text>\n"
            "- /research download <url>\n"
            "- /write <path>\n"
            "- /exit"
        )

    # ------------------------------------------------------------------
    # Core state operations
    # ------------------------------------------------------------------

    def _on_enter_state(self, state: State) -> None:
        if state == State.DECIDE:
            self._show_decide_menu()
            return

        if state == State.PLAN:
            if self.current_trial is None or self.current_trial.status != TrialStatus.ACTIVE:
                self.current_trial = self.storage.create_trial(selected_project=None)
                self.storage.save_session(self.state, self.current_trial.trial_id)
            self.plan_project_selected = False
            self.plan_draft = ""
            self.messenger.send(
                f"[PLAN] Trial {self.current_trial.trial_name} created for {self.current_trial.date}.\n"
                "Select a starting project first with /plan project <number|name|scratch>."
            )
            self._show_projects_for_plan()
            return

        if state == State.EXPERIMENT_IMPLEMENT:
            self.messenger.send(
                "[EXPERIMENT_IMPLEMENT] Edit codes and run.sh.\n"
                "Use /write <path> and finish with /endwrite, then /exp run."
            )
            return

        if state == State.EVAL_IMPLEMENT:
            self.messenger.send(
                "[EVAL_IMPLEMENT] Edit eval.sh and eval_codes.\n"
                "Use /write <path>, then /eval run."
            )
            return

        if state == State.VIEW_SUMMARY:
            self._send_recent_summaries()
            return

        if state == State.UPDATE_AND_PUSH:
            self.update_selected_project = None
            self.update_trial_candidates = []
            self._show_update_project_menu()
            return

        if state == State.SETTINGS:
            self._show_settings()
            return

        if state == State.RESEARCH:
            self.messenger.send(
                "[RESEARCH] Set cadence with /research cadence <ask|disabled|hourly|6h|daily>.\n"
                "Use /research brainstorm <text> to save ideation into references/YYYYMM/BRAINSTORM_DD.md."
            )
            self.research_next_ping = self._calc_next_ping(self.settings.research_cadence)
            return

    def _show_decide_menu(self) -> None:
        self.messenger.send(
            "[DECIDE] Choose next state:\n"
            "- /plan\n"
            "- /view_summary\n"
            "- /update_and_push\n"
            "- /settings\n"
            "- /research\n"
            "- /status"
        )

    def _show_projects_for_plan(self) -> None:
        projects = self.storage.list_project_names()
        if not projects:
            self.messenger.send("No managed projects found. Use /plan project scratch to start from scratch.")
            return

        lines = ["Available projects:"]
        for i, name in enumerate(projects, start=1):
            lines.append(f"({i}) {name}")
        lines.append("Use /plan project <number|name|scratch>")
        self.messenger.send("\n".join(lines))

    def _select_plan_project(self, selection: str) -> None:
        if not self.current_trial:
            self.messenger.send("No active PLAN trial.")
            return

        projects = self.storage.list_project_names()
        selected: str | None
        if selection.lower() == "scratch":
            selected = None
        elif selection.isdigit():
            idx = int(selection)
            if idx < 1 or idx > len(projects):
                self.messenger.send("Invalid project index.")
                return
            selected = projects[idx - 1]
        else:
            if selection not in projects:
                self.messenger.send("Project not found.")
                return
            selected = selection

        try:
            self.storage.replace_trial_codes_from_project(self.current_trial, selected)
        except Exception as e:
            self.messenger.send(f"Failed to set project: {e}")
            return

        self.plan_project_selected = True
        self.messenger.send(f"Plan source set to: {selected or 'scratch'}")

    def _start_write_session(self, user_path: str) -> None:
        if not self.current_trial and self.state not in {State.RESEARCH, State.SETTINGS, State.UPDATE_AND_PUSH}:
            self.messenger.send("No active trial.")
            return

        resolved = self._resolve_user_path(user_path)
        try:
            validated = self.authority.validate_write_path(
                self.state,
                resolved,
                self.current_trial,
                selected_project=self.update_selected_project,
            )
        except AuthorityError as e:
            self.messenger.send(f"Write blocked: {e}")
            return

        self.write_session = {
            "path": validated,
            "lines": [],
        }
        self.messenger.send("Write session started. Send file content lines, then /endwrite or /cancelwrite.")

    def _handle_write_session(self, msg: str) -> None:
        assert self.write_session is not None
        if msg == "/cancelwrite":
            self.write_session = None
            self.messenger.send("Write session cancelled.")
            return
        if msg == "/endwrite":
            path: Path = self.write_session["path"]
            content = "\n".join(self.write_session["lines"]) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self.write_session = None
            self.messenger.send(f"Wrote {path}")
            return

        self.write_session["lines"].append(msg)

    def _run_experiment_once(self) -> None:
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        script = self.storage.base_dir / self.current_trial.sandbox_path / "run.sh"
        try:
            self.authority.validate_execute_path(State.EXPERIMENT_EXECUTE, script, self.current_trial)
        except AuthorityError as e:
            self.messenger.send(f"Cannot execute run.sh: {e}")
            return

        if not self.messenger.confirm(f"Execute run.sh for {self.current_trial.trial_name}?"):
            self.messenger.send("Execution cancelled.")
            return

        self.current_trial.experiment_iter += 1
        self.storage.append_trial_record(self.current_trial)

        self._transition(State.EXPERIMENT_EXECUTE, "run.sh execution")
        self.messenger.send("[EXPERIMENT_EXECUTE] Running run.sh...")

        trial_root = self.storage.base_dir / self.current_trial.sandbox_path
        outputs_root = self.storage.base_dir / self.current_trial.outputs_path
        success, detail = self._execute_script(
            script_path=script,
            cwd=trial_root,
            stdout_path=outputs_root / "experiment_stdout.log",
            stderr_path=outputs_root / "experiment_stderr.log",
            watch_roots=[trial_root],
            allowed_write_roots=[outputs_root],
            extra_env={
                "RC_CODES_DIR": str(trial_root / "codes"),
                "RC_OUTPUTS_DIR": str(outputs_root),
                "RC_RESULTS_DIR": str(self.storage.base_dir / self.current_trial.results_path),
            },
        )

        if success:
            self.messenger.send("Experiment completed successfully. Moving to EVAL_IMPLEMENT.")
            self._transition(State.EVAL_IMPLEMENT, "experiment success")
            self._on_enter_state(State.EVAL_IMPLEMENT)
            return

        self.messenger.send(f"Experiment failed: {detail}")
        if self.current_trial.experiment_iter >= self.settings.experiment_max_iterations:
            self._run_report_summary("experiment max iterations reached")
            return

        self._transition(State.EXPERIMENT_IMPLEMENT, "experiment failure retry")
        self.messenger.send("Returning to EXPERIMENT_IMPLEMENT with failure logs. Update code and retry.")

    def _run_eval_once(self) -> None:
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        script = self.storage.base_dir / self.current_trial.sandbox_path / "eval.sh"
        try:
            self.authority.validate_execute_path(State.EVAL_EXECUTE, script, self.current_trial)
        except AuthorityError as e:
            self.messenger.send(f"Cannot execute eval.sh: {e}")
            return

        if not self.messenger.confirm(f"Execute eval.sh for {self.current_trial.trial_name}?"):
            self.messenger.send("Evaluation cancelled.")
            return

        self.current_trial.eval_iter += 1
        self.storage.append_trial_record(self.current_trial)

        self._transition(State.EVAL_EXECUTE, "eval.sh execution")
        self.messenger.send("[EVAL_EXECUTE] Running eval.sh...")

        trial_root = self.storage.base_dir / self.current_trial.sandbox_path
        results_root = self.storage.base_dir / self.current_trial.results_path
        success, detail = self._execute_script(
            script_path=script,
            cwd=trial_root,
            stdout_path=results_root / "eval_stdout.log",
            stderr_path=results_root / "eval_stderr.log",
            watch_roots=[trial_root, results_root],
            allowed_write_roots=[results_root],
            extra_env={
                "RC_CODES_DIR": str(trial_root / "codes"),
                "RC_OUTPUTS_DIR": str(self.storage.base_dir / self.current_trial.outputs_path),
                "RC_RESULTS_DIR": str(results_root),
            },
        )

        if success:
            self.messenger.send("Evaluation completed successfully.")
            self._run_report_summary("eval completed")
            return

        self.messenger.send(f"Evaluation failed: {detail}")
        if self.current_trial.eval_iter >= self.settings.eval_max_iterations:
            self._run_report_summary("eval max iterations reached")
            return

        self._transition(State.EVAL_IMPLEMENT, "eval failure retry")
        self.messenger.send("Returning to EVAL_IMPLEMENT with failure logs. Update eval code and retry.")

    def _run_report_summary(self, reason: str) -> None:
        if not self.current_trial:
            self._transition(State.DECIDE, "report with no trial")
            self._on_enter_state(State.DECIDE)
            return

        self._transition(State.REPORT_SUMMARY, reason)
        self.messenger.send("[REPORT_SUMMARY] Generating report. Interruptions are ignored until complete.")

        report_rel, summary = self.reporter.generate_report(self.current_trial, reason=reason)
        self.storage.append_experiment_log(self.current_trial, summary, report_rel)

        self.current_trial.finished_at = datetime.now().isoformat()
        if self.current_trial.terminated:
            self.current_trial.status = TrialStatus.TERMINATED
        else:
            self.current_trial.status = TrialStatus.COMPLETED
        self.storage.append_trial_record(self.current_trial)

        trial_name = self.current_trial.trial_name
        self.current_trial = None
        self.storage.save_session(State.DECIDE, None)

        self.messenger.send(
            f"Report completed for {trial_name}.\n"
            f"Summary: {summary}\n"
            f"REPORT: {report_rel}"
        )

        self._transition(State.DECIDE, "report complete")

        if self.settings.autopilot_enabled:
            self.messenger.send("Autopilot is enabled. Moving directly to PLAN.")
            self._transition(State.PLAN, "autopilot")
            self._on_enter_state(State.PLAN)
        else:
            self._on_enter_state(State.DECIDE)

    # ------------------------------------------------------------------
    # UPDATE_AND_PUSH helpers
    # ------------------------------------------------------------------

    def _show_update_project_menu(self) -> None:
        projects = self.storage.list_project_names()
        lines = []
        for i, p in enumerate(projects, start=1):
            lines.append(f"({i}) {p}")
        lines.append(f"({len(projects) + 1}) Add new project via git clone")
        lines.append(f"({len(projects) + 2}) Add new project via git init")
        lines.append("Select with /update select <number|name>")
        self.messenger.send("\n".join(lines))

    def _select_update_project(self, selector: str) -> None:
        projects = self.storage.list_project_names()
        selection = selector
        if selector.isdigit():
            idx = int(selector)
            if idx == len(projects) + 1:
                self.messenger.send("Use: /project add-clone <name> <remote_url> [branch]")
                return
            if idx == len(projects) + 2:
                self.messenger.send("Use: /project add-init <name> <remote_url> [branch]")
                return
            if idx < 1 or idx > len(projects):
                self.messenger.send("Invalid project selection.")
                return
            selection = projects[idx - 1]

        if selection not in projects:
            self.messenger.send("Unknown project name.")
            return

        self.update_selected_project = selection
        self.messenger.send(f"Selected project: {selection}")

        self.update_trial_candidates = self.storage.list_recent_trials(self.settings.view_summary_page_size)
        if not self.update_trial_candidates:
            self.messenger.send("No trials available to assimilate.")
            return

        lines = ["Recent trials:"]
        for i, t in enumerate(self.update_trial_candidates, start=1):
            lines.append(f"({i}) [{t.date}] [{t.trial_name}] status={t.status.value}")
        lines.append("Select trial with /update trial <index> or /update trial <YYYYMMDD trial_XXX>")
        self.messenger.send("\n".join(lines))

    def _select_update_trial(self, selector: str) -> None:
        if not self.update_selected_project:
            self.messenger.send("Select a project first with /update select ...")
            return

        trial: TrialRecord | None = None
        if selector.isdigit():
            idx = int(selector)
            if idx < 1 or idx > len(self.update_trial_candidates):
                self.messenger.send("Invalid trial index.")
                return
            trial = self.update_trial_candidates[idx - 1]
        else:
            parts = selector.split()
            if len(parts) != 2:
                self.messenger.send("Use /update trial <index> or /update trial <YYYYMMDD trial_XXX>")
                return
            date, trial_name = parts
            candidates = self.storage.list_trials_for_date(date)
            for item in candidates:
                if item.trial_name == trial_name:
                    trial = item
                    break
            if trial is None:
                self.messenger.send("Trial not found for date.")
                return

        self._perform_update_and_push(trial)

    def _perform_update_and_push(self, trial: TrialRecord) -> None:
        assert self.update_selected_project is not None
        project = self.update_selected_project

        try:
            copied = self.git.assimilate_trial_codes(self.state, trial, project)
            cfg = ensure_project_config(self.settings, project)
            message = f"Assimilate {trial.trial_id}"
            push_out = self.git.commit_and_push(self.state, project, cfg, message)
            self.storage.save_settings(self.settings)
        except Exception as e:
            self.messenger.send(f"UPDATE_AND_PUSH failed: {e}")
            return

        self.messenger.send(
            f"Assimilation completed for {project}.\n"
            f"Files updated: {len(copied)}\n"
            f"Push result: {push_out or 'ok'}"
        )
        self._transition(State.DECIDE, "update and push complete")
        self._on_enter_state(State.DECIDE)

    def _cmd_project_add_clone(self, msg: str) -> None:
        try:
            tokens = shlex.split(msg)
            # /project add-clone <name> <remote_url> [branch]
            _, _, name, remote, *rest = tokens
            branch = rest[0] if rest else "main"
            result = self.git.add_project_clone(self.state, name, remote, branch)
            cfg = ensure_project_config(self.settings, name)
            cfg.remote_url = remote
            cfg.default_branch = branch
            self.storage.save_settings(self.settings)
            self.messenger.send(result)
            self._show_update_project_menu()
        except Exception as e:
            self.messenger.send(f"Add clone failed: {e}")

    def _cmd_project_add_init(self, msg: str) -> None:
        try:
            tokens = shlex.split(msg)
            # /project add-init <name> <remote_url> [branch]
            _, _, name, remote, *rest = tokens
            branch = rest[0] if rest else "main"
            result = self.git.add_project_init(self.state, name, remote, branch)
            cfg = ensure_project_config(self.settings, name)
            cfg.remote_url = remote
            cfg.default_branch = branch
            self.storage.save_settings(self.settings)
            self.messenger.send(result)
            self._show_update_project_menu()
        except Exception as e:
            self.messenger.send(f"Add init failed: {e}")

    # ------------------------------------------------------------------
    # Summaries / settings / research helpers
    # ------------------------------------------------------------------

    def _send_recent_summaries(self) -> None:
        lines = self.storage.load_experiment_log_lines()
        if not lines:
            self.messenger.send("No experiment summaries yet.")
            return

        limit = self.settings.view_summary_page_size
        recent = lines[-limit:]
        self.messenger.send("Recent trials:\n" + "\n".join(recent[::-1]))
        self.messenger.send("Use /older to list dates, /date YYYYMMDD to drill down, /exit to leave.")

    def _send_trials_for_date(self, date: str) -> None:
        trials = self.storage.list_trials_for_date(date)
        if not trials:
            self.messenger.send("No trials for that date.")
            return

        summaries = self._summary_map()
        out = []
        for t in trials:
            key = f"{t.date}-{t.trial_name}"
            summary = summaries.get(key, "(summary unavailable)")
            out.append(f"[{t.date}] [{t.trial_name}] {summary}")
        self.messenger.send("\n".join(out))

    def _summary_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for line in self.storage.load_experiment_log_lines():
            m = re.match(r"^(\d{8}) - (trial_\d{3}): (.*)\. Full Doc: ", line)
            if m:
                date, trial_name, summary = m.groups()
                mapping[f"{date}-{trial_name}"] = summary
        return mapping

    def _show_settings(self) -> None:
        self.messenger.send(json.dumps(self.settings.to_dict(), indent=2))

    def _append_brainstorm(self, note: str) -> None:
        now = datetime.now()
        month_dir = self.storage.references_dir / now.strftime("%Y%m")
        month_dir.mkdir(parents=True, exist_ok=True)
        path = month_dir / f"BRAINSTORM_{now.strftime('%d')}.md"

        validated = self.authority.validate_write_path(self.state, path, self.current_trial)
        with open(validated, "a", encoding="utf-8") as f:
            f.write(f"- {now.isoformat()} | {note}\n")

    def _download_reference(self, url: str) -> None:
        try:
            self.authority.assert_network(self.state)
            now = datetime.now()
            month_dir = self.storage.references_dir / now.strftime("%Y%m")
            month_dir.mkdir(parents=True, exist_ok=True)

            parsed = urlparse(url)
            name = Path(parsed.path).name or f"download_{int(now.timestamp())}.bin"
            target = month_dir / name
            validated = self.authority.validate_write_path(self.state, target, self.current_trial)
            urlretrieve(url, validated)
            self.messenger.send(f"Downloaded: {validated}")
        except Exception as e:
            self.messenger.send(f"Download failed: {e}")

    def _calc_next_ping(self, cadence: str) -> datetime | None:
        now = datetime.now()
        if cadence in {"ask", "disabled"}:
            return None
        if cadence == "hourly":
            return now + timedelta(hours=1)
        if cadence == "6h":
            return now + timedelta(hours=6)
        if cadence == "daily":
            return now + timedelta(days=1)
        return None

    def _maybe_send_research_nudge(self) -> None:
        if self.state != State.RESEARCH:
            return
        if self.research_next_ping is None:
            self.research_next_ping = self._calc_next_ping(self.settings.research_cadence)
            return
        if datetime.now() >= self.research_next_ping:
            self.messenger.send(
                "[Research Nudge] Capture one new hypothesis in /research brainstorm and link it to a testable next trial."
            )
            self.research_next_ping = self._calc_next_ping(self.settings.research_cadence)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _resolve_user_path(self, user_path: str) -> Path:
        p = Path(user_path)
        if p.is_absolute():
            return p

        # Allow fully qualified relative paths as-is.
        root_prefixes = {"sandbox", "results", "references", "projects"}
        if p.parts and p.parts[0] in root_prefixes:
            return self.storage.base_dir / p

        if self.state in {State.PLAN, State.EXPERIMENT_IMPLEMENT, State.EVAL_IMPLEMENT} and self.current_trial:
            return self.storage.base_dir / self.current_trial.sandbox_path / p

        if self.state == State.RESEARCH:
            today = datetime.now()
            return self.storage.references_dir / today.strftime("%Y%m") / p

        if self.state == State.UPDATE_AND_PUSH and self.update_selected_project:
            return self.storage.projects_dir / self.update_selected_project / p

        return self.storage.base_dir / p

    def _snapshot_tree(self, root: Path) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        if not root.exists():
            return snapshot
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def _execute_script(
        self,
        script_path: Path,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        watch_roots: list[Path],
        allowed_write_roots: list[Path],
        extra_env: dict[str, str],
    ) -> tuple[bool, str]:
        before: dict[str, tuple[int, int]] = {}
        for root in watch_roots:
            before.update(self._snapshot_tree(root))

        completed = subprocess.run(
            ["bash", str(script_path.name)],
            cwd=str(cwd),
            env={**os.environ.copy(), **extra_env},
            capture_output=True,
            text=True,
        )

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        after: dict[str, tuple[int, int]] = {}
        for root in watch_roots:
            after.update(self._snapshot_tree(root))

        changed = []
        for path, meta in after.items():
            if path not in before or before[path] != meta:
                changed.append(Path(path).resolve())

        violations = [
            str(p)
            for p in changed
            if not any(self._is_within(p, allowed_root.resolve()) for allowed_root in allowed_write_roots)
        ]

        if completed.returncode != 0:
            return False, f"exit={completed.returncode}; stderr tail={completed.stderr[-300:]}"
        if violations:
            return False, f"write-scope violation: {violations[:10]}"

        return True, "ok"

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _send_status(self) -> None:
        if self.current_trial:
            trial = (
                f"{self.current_trial.trial_name} ({self.current_trial.date}) "
                f"status={self.current_trial.status.value}"
            )
        else:
            trial = "(none)"
        self.messenger.send(
            f"State={self.state.value}\n"
            f"Current trial={trial}\n"
            f"Autopilot={self.settings.autopilot_enabled}"
        )

    def _send_help(self) -> None:
        self.messenger.send(
            "Global: /status, /help, /abort, /autopilot-start, /autopilot-stop, /exit\n"
            "DECIDE: /plan, /view_summary, /update_and_push, /settings, /research"
        )
