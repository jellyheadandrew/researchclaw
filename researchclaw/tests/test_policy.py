from __future__ import annotations

from pathlib import Path

from researchclaw.models import TrialRecord
from researchclaw.policy import AuthorityManager, AuthorityError
from researchclaw.states import State, TrialStatus


def _trial(base_dir: Path) -> TrialRecord:
    return TrialRecord(
        trial_id="20260227-trial_001",
        date="20260227",
        trial_number=1,
        state=State.PLAN,
        status=TrialStatus.ACTIVE,
        selected_project=None,
        sandbox_path="sandbox/20260227/trial_001",
        outputs_path="sandbox/20260227/trial_001/outputs",
        results_path="results/20260227/trial_001",
    )


def test_state_count_is_11() -> None:
    assert len(State) == 11


def test_experiment_implement_write_scope(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))
    trial = _trial(tmp_path)

    allowed = tmp_path / "sandbox/20260227/trial_001/codes/train.py"
    auth.validate_write_path(State.EXPERIMENT_IMPLEMENT, allowed, trial)

    run_sh = tmp_path / "sandbox/20260227/trial_001/run.sh"
    auth.validate_write_path(State.EXPERIMENT_IMPLEMENT, run_sh, trial)

    denied = tmp_path / "sandbox/20260227/trial_001/outputs/out.txt"
    try:
        auth.validate_write_path(State.EXPERIMENT_IMPLEMENT, denied, trial)
        assert False, "expected write denial"
    except AuthorityError:
        pass


def test_eval_execute_write_scope(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))
    trial = _trial(tmp_path)

    allowed = tmp_path / "results/20260227/trial_001/metrics.json"
    auth.validate_write_path(State.EVAL_EXECUTE, allowed, trial)

    denied = tmp_path / "sandbox/20260227/trial_001/eval_codes/eval.py"
    try:
        auth.validate_write_path(State.EVAL_EXECUTE, denied, trial)
        assert False, "expected write denial"
    except AuthorityError:
        pass


def test_decide_write_denied(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))
    trial = _trial(tmp_path)

    with_denied = tmp_path / "results/20260227/trial_001/x.txt"
    try:
        auth.validate_write_path(State.DECIDE, with_denied, trial)
        assert False, "expected write denial"
    except AuthorityError:
        pass


def test_plan_allows_network(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))
    # Should not raise
    auth.assert_network(State.PLAN)


def test_experiment_execute_read_restricted_to_trial_sandbox(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))
    trial = _trial(tmp_path)

    # Allowed: within trial sandbox
    allowed = tmp_path / "sandbox/20260227/trial_001/codes/model.py"
    auth.validate_read_path(State.EXPERIMENT_EXECUTE, allowed, trial)

    # Denied: projects directory
    denied = tmp_path / "projects/myproject/main.py"
    try:
        auth.validate_read_path(State.EXPERIMENT_EXECUTE, denied, trial)
        assert False, "expected read denial"
    except AuthorityError:
        pass

    # Denied: other trial's sandbox
    denied2 = tmp_path / "sandbox/20260227/trial_002/codes/other.py"
    try:
        auth.validate_read_path(State.EXPERIMENT_EXECUTE, denied2, trial)
        assert False, "expected read denial"
    except AuthorityError:
        pass


def test_eval_execute_read_restricted(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))
    trial = _trial(tmp_path)

    # Allowed: outputs
    auth.validate_read_path(
        State.EVAL_EXECUTE,
        tmp_path / "sandbox/20260227/trial_001/outputs/out.log",
        trial,
    )

    # Allowed: eval_codes
    auth.validate_read_path(
        State.EVAL_EXECUTE,
        tmp_path / "sandbox/20260227/trial_001/eval_codes/eval.py",
        trial,
    )

    # Allowed: eval.sh
    auth.validate_read_path(
        State.EVAL_EXECUTE,
        tmp_path / "sandbox/20260227/trial_001/eval.sh",
        trial,
    )

    # Allowed: results path
    auth.validate_read_path(
        State.EVAL_EXECUTE,
        tmp_path / "results/20260227/trial_001/metrics.json",
        trial,
    )

    # Denied: codes directory
    try:
        auth.validate_read_path(
            State.EVAL_EXECUTE,
            tmp_path / "sandbox/20260227/trial_001/codes/model.py",
            trial,
        )
        assert False, "expected read denial for codes in EVAL_EXECUTE"
    except AuthorityError:
        pass


def test_report_summary_read_restricted(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))
    trial = _trial(tmp_path)

    # Allowed: trial sandbox
    auth.validate_read_path(
        State.REPORT_SUMMARY,
        tmp_path / "sandbox/20260227/trial_001/codes/model.py",
        trial,
    )

    # Allowed: results (any trial, for comparison)
    auth.validate_read_path(
        State.REPORT_SUMMARY,
        tmp_path / "results/20260226/trial_001/REPORT.md",
        trial,
    )

    # Allowed: EXPERIMENT_LOGS.md
    auth.validate_read_path(
        State.REPORT_SUMMARY,
        tmp_path / "EXPERIMENT_LOGS.md",
        trial,
    )

    # Denied: projects
    try:
        auth.validate_read_path(
            State.REPORT_SUMMARY,
            tmp_path / "projects/demo/main.py",
            trial,
        )
        assert False, "expected read denial for projects in REPORT_SUMMARY"
    except AuthorityError:
        pass


def test_research_read_restricted(tmp_path: Path) -> None:
    auth = AuthorityManager(str(tmp_path))

    # Allowed: references
    auth.validate_read_path(
        State.RESEARCH,
        tmp_path / "references/202602/BRAINSTORM_01.md",
        None,
    )

    # Allowed: EXPERIMENT_LOGS.md
    auth.validate_read_path(
        State.RESEARCH,
        tmp_path / "EXPERIMENT_LOGS.md",
        None,
    )

    # Allowed: results (for reviewing past trials)
    auth.validate_read_path(
        State.RESEARCH,
        tmp_path / "results/20260227/trial_001/REPORT.md",
        None,
    )

    # Denied: sandbox
    try:
        auth.validate_read_path(
            State.RESEARCH,
            tmp_path / "sandbox/20260227/trial_001/codes/model.py",
            None,
        )
        assert False, "expected read denial for sandbox in RESEARCH"
    except AuthorityError:
        pass
