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

from .ai_worker import AIWorker
from .config import Config
from .cron import CronJob, CronScheduler, cadence_to_seconds
from .git_manager import GitManager, ensure_project_config
from .messenger import Messenger, get_messenger
from .models import Settings, TrialRecord
from .planner import PlanEngine
from .policy import AuthorityError, AuthorityManager
from .reporter import Reporter
from .research_engine import ResearchEngine
from .states import State, TrialStatus
from .storage import StorageManager


class ResearchClaw:
    def __init__(self, config: Config, messenger: Messenger | None = None):
        self.config = config
        self.storage = StorageManager(config.base_dir)
        self.storage.ensure_layout()

        self.settings: Settings = self.storage.load_settings()
        self.authority = AuthorityManager(config.base_dir)
        self.git = GitManager(config.base_dir, self.authority, settings=self.settings)
        self.reporter = Reporter(
            config.base_dir,
            use_claude=config.planner_use_claude,
            cli_path=config.planner_claude_cli_path,
            model=config.planner_model,
        )
        self.planner = PlanEngine(
            use_claude=config.planner_use_claude,
            cli_path=config.planner_claude_cli_path,
            model=config.planner_model,
            base_dir=config.base_dir,
            web_search_enabled=config.planner_web_search,
        )
        self.ai_worker = AIWorker(
            cli_path=config.planner_claude_cli_path,
            model=config.planner_model,
        ) if config.planner_use_claude else None
        self.research_engine = ResearchEngine(
            base_dir=config.base_dir,
            cli_path=config.planner_claude_cli_path,
            model=config.planner_model,
            use_claude=config.planner_use_claude,
        )
        self.cron = CronScheduler(
            state_path=Path(config.base_dir) / ".researchclaw" / "cron_state.json"
        )
        self.messenger = messenger or get_messenger(
            config.messenger_type,
            bot_token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
        )

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
        self.messenger.send("ResearchClaw online.")
        self._setup_cron()
        self._on_enter_state(self.state)
        try:
            while True:
                self._maybe_send_research_nudge()
                message = self.messenger.receive(timeout=1.0)
                if message is None:
                    continue
                self.handle_message(message.strip())
        finally:
            self.cron.stop()

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
        lower = msg.lower().strip()
        if lower in {"/plan", "plan"}:
            self._transition(State.PLAN, "user selected PLAN")
            self._on_enter_state(State.PLAN)
            return
        if lower in {"/view_summary", "view_summary", "summary", "view summary"}:
            self._transition(State.VIEW_SUMMARY, "user selected VIEW_SUMMARY")
            self._on_enter_state(State.VIEW_SUMMARY)
            return
        if lower in {"/update_and_push", "update_and_push", "push", "update and push", "update"}:
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

        # Natural language intent detection
        if not msg.startswith("/"):
            intent = self._classify_decide_intent(lower)
            if intent:
                self._handle_decide(intent)
                return

        self._show_decide_menu()

    def _classify_decide_intent(self, text: str) -> str | None:
        """Keyword-based intent classification for DECIDE state."""
        plan_keywords = {
            "experiment", "try", "test", "run", "new trial", "start", "begin",
            "next", "plan", "new experiment", "let's go", "another",
        }
        summary_keywords = {
            "review", "look at", "past", "history", "results", "reports",
            "see", "what happened", "show me", "summary", "view",
        }
        update_keywords = {
            "push", "merge", "update", "assimilate", "commit", "deploy",
            "sync", "upload",
        }
        settings_keywords = {
            "config", "setting", "configure", "setup", "change setting",
            "preferences", "options",
        }
        research_keywords = {
            "paper", "search", "read", "brainstorm", "idea", "explore",
            "research", "literature", "survey", "find papers",
        }

        for kw in plan_keywords:
            if kw in text:
                return "/plan"
        for kw in research_keywords:
            if kw in text:
                return "/research"
        for kw in summary_keywords:
            if kw in text:
                return "/view_summary"
        for kw in update_keywords:
            if kw in text:
                return "/update_and_push"
        for kw in settings_keywords:
            if kw in text:
                return "/settings"
        return None

    def _handle_plan(self, msg: str) -> None:
        if not self.current_trial:
            self.current_trial = self.storage.create_trial(selected_project=None)
            self.storage.save_session(self.state, self.current_trial.trial_id)

        # Slash command shortcuts
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
                self.messenger.send("No plan draft yet. Describe your experiment idea to start.")
            return

        if msg == "/plan approve":
            self._approve_plan()
            return

        # Natural language: detect project selection if not yet selected
        if not self.plan_project_selected:
            project_match = self._match_project_from_text(msg)
            if project_match is not None:
                self._select_plan_project(project_match)
                return

        # Natural language: detect plan approval
        lower = msg.lower().strip()
        approval_phrases = {
            "approve", "approved", "looks good", "go ahead", "ship it",
            "lgtm", "let's go", "do it", "start", "proceed", "yes",
        }
        if lower in approval_phrases or lower.startswith("approve"):
            self._approve_plan()
            return

        # Natural language: detect show plan
        show_phrases = {"show plan", "see the plan", "current plan", "what's the plan"}
        if lower in show_phrases:
            if self.plan_draft.strip():
                self.messenger.send(self.plan_draft)
            else:
                self.messenger.send("No plan draft yet. Describe your experiment idea to start.")
            return

        # Everything else: update plan via AI
        experiment_logs = self._read_experiment_logs()
        prior_reports = self._gather_recent_reports()
        updated = self.planner.update_plan(
            self.plan_draft,
            msg,
            self.current_trial.selected_project,
            experiment_logs=experiment_logs,
            prior_reports=prior_reports,
            autopilot=self.settings.autopilot_enabled,
        )
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

    def _approve_plan(self) -> None:
        """Approve the current plan and transition to automatic implementation."""
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return
        if not self.plan_project_selected:
            self.messenger.send(
                "Please select a starting project first. "
                "Say a project name/number, or 'scratch'."
            )
            return
        if not self.plan_draft.strip():
            self.messenger.send("No plan drafted yet. Describe your experiment idea first.")
            return
        self.current_trial.plan_approved = True
        self.storage.append_trial_record(self.current_trial)
        self.messenger.send("Plan approved! Starting automatic implementation...")
        self._transition(State.EXPERIMENT_IMPLEMENT, "plan approved")
        self._on_enter_state(State.EXPERIMENT_IMPLEMENT)

    def _match_project_from_text(self, msg: str) -> str | None:
        """Try to match a project selection from natural language text."""
        lower = msg.lower().strip()
        projects = self.storage.list_project_names()

        # "scratch" or "from scratch" or "start fresh"
        scratch_phrases = {"scratch", "from scratch", "start fresh", "nothing", "no project", "empty"}
        if lower in scratch_phrases or any(p in lower for p in scratch_phrases):
            return "scratch"

        # Check for direct project name match
        for name in projects:
            if name.lower() in lower:
                return name

        # Check for number selection: "1", "project 1", "use 1", etc.
        import re as _re
        num_match = _re.search(r"\b(\d+)\b", lower)
        if num_match:
            idx = int(num_match.group(1))
            if 1 <= idx <= len(projects):
                return projects[idx - 1]

        return None

    def _handle_experiment_implement(self, msg: str) -> None:
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        # Escape hatch: manual write session
        if msg.startswith("/write "):
            path = msg[len("/write ") :].strip()
            self._start_write_session(path)
            return

        # Manual trigger: run experiment
        if msg == "/exp run":
            self._run_experiment_once()
            return

        if msg == "/exp status":
            self.messenger.send(
                f"Trial {self.current_trial.trial_name} | exp_iter={self.current_trial.experiment_iter}/{self.settings.experiment_max_iterations}"
            )
            return

        # During auto-execution, user messages are informational
        self.messenger.send(
            "Experiment implementation is in progress. Use /abort to cancel, "
            "or /write <path> for manual overrides."
        )

    def _handle_eval_implement(self, msg: str) -> None:
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        # Escape hatch: manual write session
        if msg.startswith("/write "):
            path = msg[len("/write ") :].strip()
            self._start_write_session(path)
            return

        # Manual trigger: run eval
        if msg == "/eval run":
            self._run_eval_once()
            return

        if msg == "/eval status":
            self.messenger.send(
                f"Trial {self.current_trial.trial_name} | eval_iter={self.current_trial.eval_iter}/{self.settings.eval_max_iterations}"
            )
            return

        # During auto-execution, user messages are informational
        self.messenger.send(
            "Evaluation implementation is in progress. Use /abort to cancel, "
            "or /write <path> for manual overrides."
        )

    def _handle_view_summary(self, msg: str) -> None:
        lower = msg.lower().strip()

        if lower == "/older" or any(kw in lower for kw in ["older", "more", "earlier", "previous", "all dates"]):
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

        # Natural language: detect date mention (YYYYMMDD pattern)
        import re as _re
        date_match = _re.search(r"\b(20\d{6})\b", msg)
        if date_match:
            self._send_trials_for_date(date_match.group(1))
            return

        if any(kw in lower for kw in ["exit", "back", "done", "leave"]):
            self._handle_exit()
            return

        self.messenger.send(
            "Say 'older' to see all dates, mention a date (YYYYMMDD) to drill down, "
            "or 'exit' to go back."
        )

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
                "- autopilot_max_consecutive_trials: max trials before autopilot stops\n"
                "- autopilot_max_consecutive_failures: max consecutive failures before autopilot stops\n"
                "- research_cadence: ask/disabled/hourly/6h/daily\n"
                "- git_user_name: git identity name for commits\n"
                "- git_user_email: git identity email for commits\n"
                "- git_auth_method: system (use local credentials) or token\n"
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

            int_keys = {
                "experiment_max_iterations",
                "eval_max_iterations",
                "view_summary_page_size",
                "autopilot_max_consecutive_trials",
                "autopilot_max_consecutive_failures",
            }
            bool_keys = {"autopilot_enabled"}
            str_keys = {"git_user_name", "git_user_email"}

            if key in int_keys:
                try:
                    setattr(self.settings, key, int(value))
                except ValueError:
                    self.messenger.send(f"{key} must be an integer.")
                    return
            elif key in bool_keys:
                setattr(self.settings, key, value.lower() in {"1", "true", "yes", "on"})
            elif key == "research_cadence":
                if value not in {"ask", "disabled", "hourly", "6h", "daily"}:
                    self.messenger.send("research_cadence must be ask/disabled/hourly/6h/daily")
                    return
                self.settings.research_cadence = value
            elif key in str_keys:
                setattr(self.settings, key, value)
            elif key == "git_auth_method":
                if value not in {"system", "token"}:
                    self.messenger.send("git_auth_method must be system or token")
                    return
                self.settings.git_auth_method = value
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
            # Update cron scheduler
            interval = cadence_to_seconds(cadence)
            if interval > 0:
                job = CronJob(
                    job_id="research_nudge",
                    interval_seconds=interval,
                    callback=self._cron_research_nudge,
                )
                self.cron.register(job)
                self.cron.start()
            else:
                self.cron.set_enabled("research_nudge", False)
            self.messenger.send(f"Research cadence set to {cadence}.")
            return

        if msg.startswith("/research brainstorm "):
            topic = msg[len("/research brainstorm ") :].strip()
            if not topic:
                self.messenger.send("Provide a topic.")
                return
            history = self._read_experiment_logs()
            result = self.research_engine.brainstorm(topic, history)
            self._append_brainstorm(result)
            self.messenger.send(result)
            return

        if msg.startswith("/research search "):
            query = msg[len("/research search ") :].strip()
            if not query:
                self.messenger.send("Provide a search query.")
                return
            result = self.research_engine.search_papers(query)
            self.messenger.send(result)
            return

        if msg.startswith("/research summarize "):
            text = msg[len("/research summarize ") :].strip()
            if not text:
                self.messenger.send("Provide text to summarize.")
                return
            result = self.research_engine.summarize(text)
            self.messenger.send(result)
            return

        if msg.startswith("/research download "):
            url = msg[len("/research download ") :].strip()
            if not url:
                self.messenger.send("Provide URL.")
                return
            self._download_reference(url)
            return

        # Conversational mode — messages that don't start with / are routed to AI chat
        if not msg.startswith("/"):
            history = self._read_experiment_logs()
            response = self.research_engine.chat(msg, history)
            self.messenger.send(response)
            return

        self.messenger.send(
            "RESEARCH commands:\n"
            "- /research search <query>  (search papers)\n"
            "- /research brainstorm <topic>  (AI-powered brainstorming)\n"
            "- /research summarize <text>  (summarize content)\n"
            "- /research cadence <ask|disabled|hourly|6h|daily>\n"
            "- /research download <url>\n"
            "- /write <path>\n"
            "- /exit\n"
            "Or just type naturally to chat about research ideas."
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

            projects = self.storage.list_project_names()
            if projects:
                project_list = "\n".join(f"  ({i}) {p}" for i, p in enumerate(projects, 1))
                self.messenger.send(
                    f"Planning trial {self.current_trial.trial_name} ({self.current_trial.date}).\n\n"
                    f"Which project should we start with?\n{project_list}\n\n"
                    "Say a project name/number, or 'scratch' to start from nothing.\n"
                    "Then describe your experiment idea — I'll help refine the plan."
                )
            else:
                self.messenger.send(
                    f"Planning trial {self.current_trial.trial_name} ({self.current_trial.date}).\n"
                    "No projects set up yet — starting from scratch.\n"
                    "Describe your experiment idea and I'll help build a plan."
                )
                self.plan_project_selected = True

            # Proactive search for recent research trends
            experiment_logs = self._read_experiment_logs()
            prior_reports = self._gather_recent_reports()
            search_results = self.planner.proactive_search(
                None,
                experiment_logs,
                prior_reports,
            )
            if search_results:
                self.messenger.send(
                    "Here are some recent research trends that might be relevant:\n"
                    + search_results
                )
            return

        if state == State.EXPERIMENT_IMPLEMENT:
            self.messenger.send("[EXPERIMENT_IMPLEMENT] Generating experiment code from plan...")
            self._auto_implement_experiment()
            return

        if state == State.EVAL_IMPLEMENT:
            self.messenger.send("[EVAL_IMPLEMENT] Generating evaluation code...")
            self._auto_implement_eval()
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
                "[RESEARCH] You can search papers, brainstorm, or just chat about ideas.\n"
                "Commands: /research search, /research brainstorm, /research summarize, /research cadence\n"
                "Or type naturally to discuss research ideas."
            )
            self.research_next_ping = self._calc_next_ping(self.settings.research_cadence)
            return

    def _show_decide_menu(self) -> None:
        self.messenger.send(
            "What would you like to do next?\n"
            "- Plan a new experiment\n"
            "- Review past trial results\n"
            "- Push results to a project\n"
            "- Explore research ideas\n"
            "- Adjust settings\n\n"
            "Just tell me what you'd like to do."
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

    def _read_experiment_logs(self) -> str:
        logs_path = self.storage.base_dir / "EXPERIMENT_LOGS.md"
        if logs_path.exists():
            try:
                return logs_path.read_text(encoding="utf-8")
            except Exception:
                pass
        return ""

    def _gather_recent_reports(self, max_reports: int = 5) -> list[str]:
        results_dir = self.storage.base_dir / "results"
        if not results_dir.exists():
            return []
        report_files = sorted(results_dir.rglob("REPORT.md"), reverse=True)
        reports: list[str] = []
        for rf in report_files[:max_reports]:
            try:
                reports.append(rf.read_text(encoding="utf-8"))
            except Exception:
                continue
        return reports

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

        self.current_trial.experiment_iter += 1
        self.storage.append_trial_record(self.current_trial)

        self._transition(State.EXPERIMENT_EXECUTE, "run.sh execution")
        self.messenger.send(
            f"[EXPERIMENT_EXECUTE] Running run.sh (iteration {self.current_trial.experiment_iter}/{self.settings.experiment_max_iterations})..."
        )

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

        # Ralph loop: auto-retry with fresh AI context (both autopilot and non-autopilot)
        if self.ai_worker:
            self._ralph_retry_experiment()
            return

        # No AI worker available — go to report summary
        self.messenger.send("No AI worker available for auto-fix. Moving to report.")
        self._run_report_summary("experiment failed, no AI worker for retry")

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

        self.current_trial.eval_iter += 1
        self.storage.append_trial_record(self.current_trial)

        self._transition(State.EVAL_EXECUTE, "eval.sh execution")
        self.messenger.send(
            f"[EVAL_EXECUTE] Running eval.sh (iteration {self.current_trial.eval_iter}/{self.settings.eval_max_iterations})..."
        )

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

        # Ralph loop: auto-retry with fresh AI context (both autopilot and non-autopilot)
        if self.ai_worker:
            self._ralph_retry_eval()
            return

        # No AI worker available — go to report summary
        self.messenger.send("No AI worker available for auto-fix. Moving to report.")
        self._run_report_summary("eval failed, no AI worker for retry")

    # ------------------------------------------------------------------
    # Auto-implementation (agent-driven code generation)
    # ------------------------------------------------------------------

    def _auto_implement_experiment(self) -> None:
        """Auto-generate experiment code from PLAN.md using AI worker."""
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        if not self.ai_worker:
            self.messenger.send(
                "AI worker unavailable. Use /write <path> to implement manually, "
                "then /exp run to execute."
            )
            return

        trial_root = self.storage.base_dir / self.current_trial.sandbox_path
        plan_path = trial_root / "PLAN.md"
        codes_dir = trial_root / "codes"

        if not plan_path.exists():
            self.messenger.send("No PLAN.md found. Cannot auto-implement.")
            return

        ok, output = self.ai_worker.implement_experiment(plan_path, codes_dir)

        if not ok:
            self.messenger.send(
                f"AI implementation failed: {output}\n"
                "Use /write <path> to implement manually, then /exp run."
            )
            return

        patches = AIWorker.parse_file_patches(output)
        if not patches:
            self.messenger.send(
                "AI produced no code files. Use /write <path> to implement manually, "
                "then /exp run."
            )
            return

        applied = 0
        run_sh_generated = False
        for rel_path, content in patches.items():
            if rel_path == "run.sh":
                target = trial_root / "run.sh"
                run_sh_generated = True
            else:
                clean = rel_path.removeprefix("codes/")
                target = codes_dir / clean
            try:
                validated = self.authority.validate_write_path(
                    State.EXPERIMENT_IMPLEMENT, target, self.current_trial
                )
                validated.parent.mkdir(parents=True, exist_ok=True)
                validated.write_text(content + "\n", encoding="utf-8")
                if rel_path == "run.sh":
                    validated.chmod(0o755)
                applied += 1
            except AuthorityError as e:
                self.messenger.send(f"Write blocked for {rel_path}: {e}")

        file_list = ", ".join(patches.keys())
        self.messenger.send(
            f"[EXPERIMENT_IMPLEMENT] Generated {applied} files: {file_list}"
        )

        if not run_sh_generated:
            run_sh = trial_root / "run.sh"
            if run_sh.read_text(encoding="utf-8").strip().endswith(
                'echo "Define experiment commands in run.sh"'
            ):
                self.messenger.send(
                    "Warning: run.sh was not generated by AI and still has placeholder content."
                )

        # Auto-transition to execution
        self._run_experiment_once()

    def _auto_implement_eval(self) -> None:
        """Auto-generate evaluation code from PLAN.md and experiment outputs using AI worker."""
        if not self.current_trial:
            self.messenger.send("No active trial.")
            return

        if not self.ai_worker:
            self.messenger.send(
                "AI worker unavailable. Use /write <path> to implement eval manually, "
                "then /eval run to execute."
            )
            return

        trial_root = self.storage.base_dir / self.current_trial.sandbox_path
        plan_path = trial_root / "PLAN.md"
        codes_dir = trial_root / "codes"
        outputs_dir = self.storage.base_dir / self.current_trial.outputs_path
        eval_codes_dir = trial_root / "eval_codes"

        ok, output = self.ai_worker.implement_eval(
            plan_path, codes_dir, outputs_dir, eval_codes_dir
        )

        if not ok:
            self.messenger.send(
                f"AI eval implementation failed: {output}\n"
                "Use /write <path> to implement manually, then /eval run."
            )
            return

        patches = AIWorker.parse_file_patches(output)
        if not patches:
            self.messenger.send(
                "AI produced no eval files. Use /write <path> to implement manually, "
                "then /eval run."
            )
            return

        applied = 0
        for rel_path, content in patches.items():
            if rel_path == "eval.sh":
                target = trial_root / "eval.sh"
            else:
                clean = rel_path.removeprefix("eval_codes/")
                target = eval_codes_dir / clean
            try:
                validated = self.authority.validate_write_path(
                    State.EVAL_IMPLEMENT, target, self.current_trial
                )
                validated.parent.mkdir(parents=True, exist_ok=True)
                validated.write_text(content + "\n", encoding="utf-8")
                if rel_path == "eval.sh":
                    validated.chmod(0o755)
                applied += 1
            except AuthorityError as e:
                self.messenger.send(f"Write blocked for {rel_path}: {e}")

        file_list = ", ".join(patches.keys())
        self.messenger.send(
            f"[EVAL_IMPLEMENT] Generated {applied} files: {file_list}"
        )

        # Auto-transition to execution
        self._run_eval_once()

    def _ralph_retry_experiment(self) -> None:
        """Fresh AI context retry for experiment failure (Ralph loop philosophy).

        Works in both autopilot and non-autopilot modes. Each retry spawns a
        fresh AI subprocess with no prior conversation context.
        """
        assert self.current_trial is not None
        assert self.ai_worker is not None

        trial_root = self.storage.base_dir / self.current_trial.sandbox_path
        outputs_root = self.storage.base_dir / self.current_trial.outputs_path

        self.messenger.send("[RALPH LOOP] Spawning fresh AI worker to fix experiment code...")
        ok, output = self.ai_worker.fix_experiment(
            plan_path=trial_root / "PLAN.md",
            codes_dir=trial_root / "codes",
            stdout_log=outputs_root / "experiment_stdout.log",
            stderr_log=outputs_root / "experiment_stderr.log",
        )

        if not ok:
            self.messenger.send(f"AI fix failed: {output}. Moving to report.")
            self._run_report_summary("experiment ralph fix failed")
            return

        patches = AIWorker.parse_file_patches(output)
        if not patches:
            self.messenger.send("AI determined no code changes needed. Moving to report.")
            self._run_report_summary("experiment ralph no changes")
            return

        codes_root = trial_root / "codes"
        applied = 0
        for rel_path, content in patches.items():
            clean = rel_path.removeprefix("codes/")
            target = codes_root / clean
            try:
                validated = self.authority.validate_write_path(
                    State.EXPERIMENT_IMPLEMENT, target, self.current_trial
                )
                validated.parent.mkdir(parents=True, exist_ok=True)
                validated.write_text(content + "\n", encoding="utf-8")
                applied += 1
            except AuthorityError:
                self.messenger.send(f"AI patch blocked for {clean}.")

        self.messenger.send(f"[RALPH LOOP] Applied {applied} file patches. Re-running experiment...")
        self._transition(State.EXPERIMENT_IMPLEMENT, "ralph fix applied")
        self._run_experiment_once()

    def _ralph_retry_eval(self) -> None:
        """Fresh AI context retry for eval failure (Ralph loop philosophy).

        Works in both autopilot and non-autopilot modes.
        """
        assert self.current_trial is not None
        assert self.ai_worker is not None

        trial_root = self.storage.base_dir / self.current_trial.sandbox_path
        outputs_root = self.storage.base_dir / self.current_trial.outputs_path
        results_root = self.storage.base_dir / self.current_trial.results_path

        self.messenger.send("[RALPH LOOP] Spawning fresh AI worker to fix eval code...")
        ok, output = self.ai_worker.fix_eval(
            eval_codes_dir=trial_root / "eval_codes",
            outputs_dir=outputs_root,
            stdout_log=results_root / "eval_stdout.log",
            stderr_log=results_root / "eval_stderr.log",
        )

        if not ok:
            self.messenger.send(f"AI fix failed: {output}. Moving to report.")
            self._run_report_summary("eval ralph fix failed")
            return

        patches = AIWorker.parse_file_patches(output)
        if not patches:
            self.messenger.send("AI determined no eval changes needed. Moving to report.")
            self._run_report_summary("eval ralph no changes")
            return

        eval_codes_root = trial_root / "eval_codes"
        applied = 0
        for rel_path, content in patches.items():
            clean = rel_path.removeprefix("eval_codes/")
            target = eval_codes_root / clean
            try:
                validated = self.authority.validate_write_path(
                    State.EVAL_IMPLEMENT, target, self.current_trial
                )
                validated.parent.mkdir(parents=True, exist_ok=True)
                validated.write_text(content + "\n", encoding="utf-8")
                applied += 1
            except AuthorityError:
                self.messenger.send(f"AI patch blocked for {clean}.")

        self.messenger.send(f"[RALPH LOOP] Applied {applied} eval patches. Re-running evaluation...")
        self._transition(State.EVAL_IMPLEMENT, "ralph fix applied")
        self._run_eval_once()

    def _should_stop_autopilot(self) -> bool:
        max_check = max(
            self.settings.autopilot_max_consecutive_trials,
            self.settings.autopilot_max_consecutive_failures,
        ) + 1
        trials = self.storage.list_recent_trials(max_check)
        if not trials:
            return False

        # Trials are sorted most-recent-first from list_recent_trials
        consecutive = 0
        consecutive_failures = 0
        for t in trials:
            if t.status in {TrialStatus.COMPLETED, TrialStatus.TERMINATED}:
                consecutive += 1
                if t.status == TrialStatus.TERMINATED:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
            else:
                break

        if consecutive >= self.settings.autopilot_max_consecutive_trials:
            self.messenger.send(
                f"Autopilot reached max consecutive trials ({self.settings.autopilot_max_consecutive_trials})."
            )
            return True

        if consecutive_failures >= self.settings.autopilot_max_consecutive_failures:
            self.messenger.send(
                f"Autopilot reached max consecutive failures ({self.settings.autopilot_max_consecutive_failures})."
            )
            return True

        return False

    def _autopilot_plan(self) -> None:
        """Auto-plan: select project, generate plan with AI, auto-approve."""
        # Create trial
        self.current_trial = self.storage.create_trial(selected_project=None)
        self.storage.save_session(self.state, self.current_trial.trial_id)

        # Auto-select project from most recent trial or first available
        selected: str | None = None
        recent = self.storage.list_recent_trials(10)
        for t in recent:
            if t.selected_project:
                selected = t.selected_project
                break
        if not selected:
            projects = self.storage.list_project_names()
            selected = projects[0] if projects else None

        try:
            self.storage.replace_trial_codes_from_project(self.current_trial, selected)
        except Exception:
            pass
        self.current_trial.selected_project = selected
        self.plan_project_selected = True

        self.messenger.send(
            f"[AUTOPILOT PLAN] Trial {self.current_trial.trial_name} | "
            f"Project: {selected or '(scratch)'}"
        )

        # Generate plan with AI (using autopilot=True for decisive prompt)
        experiment_logs = self._read_experiment_logs()
        prior_reports = self._gather_recent_reports()
        plan = self.planner.update_plan(
            "",
            "Generate the next experiment plan based on prior results.",
            selected,
            experiment_logs=experiment_logs,
            prior_reports=prior_reports,
            autopilot=True,
        )
        self.plan_draft = plan

        plan_path = self.storage.base_dir / self.current_trial.sandbox_path / "PLAN.md"
        try:
            validated = self.authority.validate_write_path(self.state, plan_path, self.current_trial)
            validated.parent.mkdir(parents=True, exist_ok=True)
            validated.write_text(plan + "\n", encoding="utf-8")
        except AuthorityError as e:
            self.messenger.send(f"Autopilot plan write blocked: {e}")

        self.messenger.send(f"[AUTOPILOT] Plan generated:\n{plan[:500]}...")

        # Auto-approve
        self.current_trial.plan_approved = True
        self.storage.append_trial_record(self.current_trial)
        self._transition(State.EXPERIMENT_IMPLEMENT, "autopilot plan approved")
        self._on_enter_state(State.EXPERIMENT_IMPLEMENT)

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
            if self._should_stop_autopilot():
                self.messenger.send("Autopilot safeguard triggered. Disabling autopilot.")
                self.settings.autopilot_enabled = False
                self.storage.save_settings(self.settings)
                self._on_enter_state(State.DECIDE)
            else:
                self.messenger.send("Autopilot is enabled. Auto-planning next trial...")
                self._transition(State.PLAN, "autopilot")
                self._autopilot_plan()
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
            # Try intelligent assimilation first, fall back to naive copy
            if self.ai_worker:
                self.messenger.send("Using AI-powered intelligent assimilation...")
                applied, detail = self.git.assimilate_intelligently(
                    self.state, trial, project, self.ai_worker
                )
                if applied:
                    self.messenger.send(f"AI assimilation: {detail}")
                else:
                    self.messenger.send(f"AI assimilation returned no changes ({detail}). Falling back to naive copy.")
                    applied = self.git.assimilate_trial_codes(self.state, trial, project)
            else:
                applied = self.git.assimilate_trial_codes(self.state, trial, project)

            cfg = ensure_project_config(self.settings, project)
            message = f"Assimilate {trial.trial_id}"
            push_out = self.git.commit_and_push(self.state, project, cfg, message)
            self.storage.save_settings(self.settings)
        except Exception as e:
            self.messenger.send(f"UPDATE_AND_PUSH failed: {e}")
            return

        file_count = len(applied) if isinstance(applied, list) else 0
        self.messenger.send(
            f"Assimilation completed for {project}.\n"
            f"Files updated: {file_count}\n"
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

    def _setup_cron(self) -> None:
        interval = cadence_to_seconds(self.settings.research_cadence)
        if interval > 0:
            job = CronJob(
                job_id="research_nudge",
                interval_seconds=interval,
                callback=self._cron_research_nudge,
            )
            self.cron.register(job)
            self.cron.start()

    def _cron_research_nudge(self) -> None:
        history = self._read_experiment_logs()
        nudge = self.research_engine.generate_nudge(history)
        if nudge:
            self.messenger.send(f"[Research Nudge] {nudge}")
        else:
            self.messenger.send(
                "[Research Nudge] Capture one new hypothesis in /research brainstorm "
                "and link it to a testable next trial."
            )

    def _maybe_send_research_nudge(self) -> None:
        if self.state != State.RESEARCH:
            return
        if self.research_next_ping is None:
            self.research_next_ping = self._calc_next_ping(self.settings.research_cadence)
            return
        if datetime.now() >= self.research_next_ping:
            self._cron_research_nudge()
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
