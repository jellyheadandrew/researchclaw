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
