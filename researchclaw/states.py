from __future__ import annotations

from enum import Enum


class State(str, Enum):
    DECIDE = "decide"
    PLAN = "plan"
    EXPERIMENT_IMPLEMENT = "experiment_implement"
    EXPERIMENT_EXECUTE = "experiment_execute"
    EVAL_IMPLEMENT = "eval_implement"
    EVAL_EXECUTE = "eval_execute"
    REPORT_SUMMARY = "report_summary"
    VIEW_SUMMARY = "view_summary"
    UPDATE_AND_PUSH = "update_and_push"
    SETTINGS = "settings"
    RESEARCH = "research"


class TrialStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    TERMINATED = "terminated"
