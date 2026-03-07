from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm import TrialAborted
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.sandbox import SandboxManager


HandlerFunc = Callable[
    [Path, TrialMeta, ResearchClawConfig, Any], State
]


class FSMEngine:
    """Core FSM dispatch loop that manages state transitions.

    The orchestrator is Python code, NOT an LLM agent. It dispatches to
    handler functions, manages state persistence, and handles transitions.
    """

    def __init__(
        self,
        project_dir: str | Path,
        config: ResearchClawConfig,
        chat_interface: Any,
        handlers: dict[State, HandlerFunc],
    ) -> None:
        self.project_dir = Path(project_dir)
        self.config = config
        self.chat_interface = chat_interface
        self.handlers = handlers

    def run(self, trial_dir: str | Path | None = None) -> None:
        """Run the FSM dispatch loop.

        Reads current state from meta.json, dispatches to the appropriate
        handler, saves updated state, and continues until a quit signal
        (SystemExit) or the handler returns None.

        Handles:
        - TrialAborted: triggers EXPERIMENT_REPORT with terminated status
        - KeyboardInterrupt (Ctrl+C): saves state and exits gracefully
        - Unexpected exceptions: saves state and re-raises

        Args:
            trial_dir: Optional trial directory to resume. If None, uses
                the latest trial from the sandbox.
        """
        if trial_dir is not None:
            current_trial = Path(trial_dir)
        else:
            current_trial_or_none = SandboxManager.get_latest_trial(self.project_dir)
            if current_trial_or_none is None:
                current_trial = SandboxManager.create_trial(self.project_dir)
            else:
                current_trial = current_trial_or_none

        meta = SandboxManager.get_trial_meta(current_trial)

        while True:
            current_state = State(meta.state)

            handler = self.handlers.get(current_state)
            if handler is None:
                break

            try:
                next_state = handler(
                    current_trial, meta, self.config, self.chat_interface
                )
            except TrialAborted:
                # User aborted — mark as terminated and route to EXPERIMENT_REPORT
                meta.status = "terminated"
                meta.updated_at = datetime.now(timezone.utc).isoformat()
                SandboxManager.save_trial_meta(current_trial, meta)

                if self.chat_interface is not None:
                    self.chat_interface.send(
                        "Trial aborted. Generating termination report..."
                    )

                # Route to EXPERIMENT_REPORT handler if available
                report_handler = self.handlers.get(State.EXPERIMENT_REPORT)
                if report_handler is not None:
                    try:
                        next_state = report_handler(
                            current_trial, meta, self.config, self.chat_interface
                        )
                    except Exception:
                        # If report generation also fails, go directly to DECIDE
                        next_state = State.DECIDE
                else:
                    next_state = State.DECIDE

                # Update meta and continue the loop
                meta.state = next_state.value
                meta.updated_at = datetime.now(timezone.utc).isoformat()
                SandboxManager.save_trial_meta(current_trial, meta)
                continue

            except KeyboardInterrupt:
                # Ctrl+C — save current state and exit gracefully
                meta.updated_at = datetime.now(timezone.utc).isoformat()
                SandboxManager.save_trial_meta(current_trial, meta)

                if self.chat_interface is not None:
                    self.chat_interface.send(
                        "\nInterrupted. State saved. You can resume later."
                    )
                return

            except SystemExit:
                # Propagate SystemExit (from /quit)
                raise

            except Exception as exc:
                # Unexpected error — save state so we can resume, then re-raise
                meta.updated_at = datetime.now(timezone.utc).isoformat()
                SandboxManager.save_trial_meta(current_trial, meta)

                if self.chat_interface is not None:
                    self.chat_interface.send(
                        f"Unexpected error: {exc}. State saved."
                    )
                raise

            # Log autopilot decision reasoning at every transition
            if self.config.autopilot:
                if not meta.decision_reasoning:
                    meta.decision_reasoning = (
                        f"[autopilot] {current_state.value} -> {next_state.value}"
                    )

            meta.state = next_state.value
            meta.updated_at = datetime.now(timezone.utc).isoformat()
            SandboxManager.save_trial_meta(current_trial, meta)
