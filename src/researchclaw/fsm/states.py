from enum import Enum


class State(str, Enum):
    """FSM states for the ResearchClaw experiment lifecycle."""

    EXPERIMENT_PLAN = "experiment_plan"
    EXPERIMENT_IMPLEMENT = "experiment_implement"
    EXPERIMENT_EXECUTE = "experiment_execute"
    EVAL_IMPLEMENT = "eval_implement"
    EVAL_EXECUTE = "eval_execute"
    EXPERIMENT_REPORT = "experiment_report"
    DECIDE = "decide"
    VIEW_SUMMARY = "view_summary"
    SETTINGS = "settings"
    MERGE_LOOP = "merge_loop"
