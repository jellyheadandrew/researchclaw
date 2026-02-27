"""
Comprehensive unit tests for PathValidator.

These tests must ALL pass before any other module is implemented.
The access control layer is the most security-critical component of ResearchClaw.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing from parent package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from researchclaw.models import TrialInfo
from researchclaw.access_control import PathValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_dir(tmp_path):
    """Create a realistic ResearchClaw directory tree in a temp directory."""
    (tmp_path / "github_codes").mkdir()
    (tmp_path / "github_codes" / "model.py").write_text("# model")
    (tmp_path / "github_codes" / "train.py").write_text("# train")
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "sandbox" / "20260101").mkdir()
    (tmp_path / "sandbox" / "20260101" / "trial_001").mkdir()
    (tmp_path / "sandbox" / "20260101" / "trial_001" / "train.py").write_text("# trial train")
    (tmp_path / "experiment_reports").mkdir()
    (tmp_path / "experiment_reports" / "20260101").mkdir()
    (tmp_path / "experiment_reports" / "20260101" / "trial_001").mkdir()
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "agent.py").write_text("# agent")
    (tmp_path / "config.yaml").write_text("base_dir: /tmp")
    (tmp_path / ".trials.jsonl").write_text("")
    return tmp_path


@pytest.fixture
def active_trial():
    """An active TrialInfo for today's date."""
    return TrialInfo(
        date="20260226",
        number=1,
        status="active",
        started_at="2026-02-26T10:00:00",
    )


@pytest.fixture
def approved_trial():
    return TrialInfo(
        date="20260226",
        number=1,
        status="approved",
        started_at="2026-02-26T10:00:00",
        finished_at="2026-02-26T13:00:00",
    )


@pytest.fixture
def rejected_trial():
    return TrialInfo(
        date="20260226",
        number=1,
        status="rejected",
        started_at="2026-02-26T10:00:00",
        finished_at="2026-02-26T13:00:00",
    )


@pytest.fixture
def validator_idle(base_dir):
    """PathValidator with no active trial."""
    return PathValidator(str(base_dir))


@pytest.fixture
def validator_active(base_dir, active_trial):
    """PathValidator with an active trial."""
    v = PathValidator(str(base_dir))
    # Create the trial directories
    sandbox = base_dir / active_trial.sandbox_path
    sandbox.mkdir(parents=True, exist_ok=True)
    report = base_dir / active_trial.report_path
    report.mkdir(parents=True, exist_ok=True)
    v.set_trial(active_trial)
    return v


# ---------------------------------------------------------------------------
# READ permission tests
# ---------------------------------------------------------------------------

class TestCanRead:
    def test_read_github_codes_file(self, validator_idle, base_dir):
        path = str(base_dir / "github_codes" / "model.py")
        assert validator_idle.can_read(path) is True

    def test_read_github_codes_subdir(self, validator_idle, base_dir):
        path = str(base_dir / "github_codes")
        assert validator_idle.can_read(path) is True

    def test_read_sandbox_old_trial(self, validator_idle, base_dir):
        path = str(base_dir / "sandbox" / "20260101" / "trial_001" / "train.py")
        assert validator_idle.can_read(path) is True

    def test_read_experiment_reports(self, validator_idle, base_dir):
        path = str(base_dir / "experiment_reports" / "20260101" / "trial_001")
        assert validator_idle.can_read(path) is True

    def test_read_config_yaml(self, validator_idle, base_dir):
        path = str(base_dir / "config.yaml")
        assert validator_idle.can_read(path) is True

    def test_read_trials_jsonl(self, validator_idle, base_dir):
        path = str(base_dir / ".trials.jsonl")
        assert validator_idle.can_read(path) is True

    def test_read_core_denied(self, validator_idle, base_dir):
        path = str(base_dir / "core" / "agent.py")
        assert validator_idle.can_read(path) is False

    def test_read_core_dir_denied(self, validator_idle, base_dir):
        path = str(base_dir / "core")
        assert validator_idle.can_read(path) is False

    def test_read_outside_base_dir_denied(self, validator_idle):
        assert validator_idle.can_read("/etc/passwd") is False

    def test_read_path_traversal_denied(self, validator_idle, base_dir):
        # Attempting to escape via ../
        traversal = str(base_dir / "github_codes" / ".." / ".." / "etc" / "passwd")
        assert validator_idle.can_read(traversal) is False

    def test_read_home_dir_denied(self, validator_idle):
        assert validator_idle.can_read("/home/someuser/.ssh/id_rsa") is False

    def test_read_current_trial_sandbox(self, validator_active, base_dir, active_trial):
        path = str(base_dir / active_trial.sandbox_path / "train.py")
        assert validator_active.can_read(path) is True


# ---------------------------------------------------------------------------
# WRITE permission tests
# ---------------------------------------------------------------------------

class TestCanWrite:
    def test_write_denied_with_no_trial(self, validator_idle, base_dir):
        """Without an active trial, all writes are denied."""
        path = str(base_dir / "sandbox" / "20260226" / "trial_001" / "x.py")
        assert validator_idle.can_write(path) is False

    def test_write_allowed_current_trial_sandbox(self, validator_active, base_dir, active_trial):
        path = str(base_dir / active_trial.sandbox_path / "new_file.py")
        assert validator_active.can_write(path) is True

    def test_write_allowed_current_trial_report(self, validator_active, base_dir, active_trial):
        path = str(base_dir / active_trial.report_path / "REPORT.md")
        assert validator_active.can_write(path) is True

    def test_write_denied_github_codes(self, validator_active, base_dir):
        """github_codes/ must never be writable — changes go through sandbox → approval."""
        path = str(base_dir / "github_codes" / "model.py")
        assert validator_active.can_write(path) is False

    def test_write_denied_other_trial(self, validator_active, base_dir):
        """Cannot write to a sibling trial, only to the currently active one."""
        path = str(base_dir / "sandbox" / "20260226" / "trial_002" / "x.py")
        assert validator_active.can_write(path) is False

    def test_write_denied_old_date_trial(self, validator_active, base_dir):
        path = str(base_dir / "sandbox" / "20260101" / "trial_001" / "x.py")
        assert validator_active.can_write(path) is False

    def test_write_denied_core(self, validator_active, base_dir):
        path = str(base_dir / "core" / "agent.py")
        assert validator_active.can_write(path) is False

    def test_write_denied_outside_base_dir(self, validator_active):
        assert validator_active.can_write("/etc/cron.d/evil") is False

    def test_write_denied_path_traversal(self, validator_active, base_dir, active_trial):
        traversal = str(base_dir / active_trial.sandbox_path / ".." / ".." / "github_codes" / "evil.py")
        assert validator_active.can_write(traversal) is False

    def test_write_denied_after_trial_approved(self, base_dir, approved_trial):
        sandbox = base_dir / approved_trial.sandbox_path
        sandbox.mkdir(parents=True, exist_ok=True)
        report = base_dir / approved_trial.report_path
        report.mkdir(parents=True, exist_ok=True)
        v = PathValidator(str(base_dir))
        v.set_trial(approved_trial)  # set with approved status
        path = str(base_dir / approved_trial.sandbox_path / "x.py")
        assert v.can_write(path) is False  # approved trial = read-only

    def test_write_denied_after_trial_rejected(self, base_dir, rejected_trial):
        sandbox = base_dir / rejected_trial.sandbox_path
        sandbox.mkdir(parents=True, exist_ok=True)
        report = base_dir / rejected_trial.report_path
        report.mkdir(parents=True, exist_ok=True)
        v = PathValidator(str(base_dir))
        v.set_trial(rejected_trial)
        path = str(base_dir / rejected_trial.sandbox_path / "x.py")
        assert v.can_write(path) is False

    def test_write_revoked_on_set_trial_none(self, validator_active, base_dir, active_trial):
        """Simulate trial finalization: set_trial(None) revokes write access."""
        path = str(base_dir / active_trial.sandbox_path / "x.py")
        assert validator_active.can_write(path) is True  # was allowed
        validator_active.set_trial(None)
        assert validator_active.can_write(path) is False  # now denied

    def test_write_denied_config_yaml(self, validator_active, base_dir):
        assert validator_active.can_write(str(base_dir / "config.yaml")) is False

    def test_write_allowed_research_trial_summary(self, validator_idle, base_dir):
        """RESEARCH_TRIAL_SUMMARY.md is always writable (no active trial required)."""
        path = str(base_dir / "RESEARCH_TRIAL_SUMMARY.md")
        assert validator_idle.can_write(path) is True

    def test_read_allowed_research_trial_summary(self, validator_idle, base_dir):
        """RESEARCH_TRIAL_SUMMARY.md is always readable."""
        (base_dir / "RESEARCH_TRIAL_SUMMARY.md").write_text("# test")
        path = str(base_dir / "RESEARCH_TRIAL_SUMMARY.md")
        assert validator_idle.can_read(path) is True


# ---------------------------------------------------------------------------
# validate_read / validate_write raise correctly
# ---------------------------------------------------------------------------

class TestValidateRaises:
    def test_validate_read_allowed(self, validator_idle, base_dir):
        path = str(base_dir / "github_codes" / "model.py")
        result = validator_idle.validate_read(path)
        assert result == Path(path).resolve()

    def test_validate_read_raises_on_deny(self, validator_idle):
        with pytest.raises(PermissionError, match="READ denied"):
            validator_idle.validate_read("/etc/passwd")

    def test_validate_write_raises_on_deny(self, validator_idle, base_dir):
        path = str(base_dir / "github_codes" / "model.py")
        with pytest.raises(PermissionError, match="WRITE denied"):
            validator_idle.validate_write(path)

    def test_validate_write_allowed(self, validator_active, base_dir, active_trial):
        path = str(base_dir / active_trial.sandbox_path / "new.py")
        result = validator_active.validate_write(path)
        assert result == Path(path).resolve()


# ---------------------------------------------------------------------------
# Shell command validation tests
# ---------------------------------------------------------------------------

class TestValidateShellCommand:
    def test_redirect_to_sandbox_allowed(self, validator_active, base_dir, active_trial):
        sandbox = str(base_dir / active_trial.sandbox_path)
        cmd = f"python train.py > {sandbox}/output.log"
        # Should not raise
        validator_active.validate_shell_command(cmd)

    def test_redirect_outside_sandbox_denied(self, validator_active, base_dir):
        cmd = "python train.py > /tmp/leaked.log"
        with pytest.raises(PermissionError):
            validator_active.validate_shell_command(cmd)

    def test_tee_to_sandbox_allowed(self, validator_active, base_dir, active_trial):
        report = str(base_dir / active_trial.report_path)
        cmd = f"python train.py | tee {report}/out.log"
        validator_active.validate_shell_command(cmd)

    def test_tee_outside_sandbox_denied(self, validator_active):
        cmd = "python train.py | tee /tmp/evil.log"
        with pytest.raises(PermissionError):
            validator_active.validate_shell_command(cmd)

    def test_cp_to_github_codes_denied(self, validator_active, base_dir):
        cmd = f"cp results.pt {base_dir}/github_codes/model.pt"
        with pytest.raises(PermissionError):
            validator_active.validate_shell_command(cmd)

    def test_mv_inside_sandbox_allowed(self, validator_active, base_dir, active_trial):
        sandbox = str(base_dir / active_trial.sandbox_path)
        cmd = f"mv {sandbox}/old.py {sandbox}/new.py"
        validator_active.validate_shell_command(cmd)

    def test_output_flag_outside_sandbox_denied(self, validator_active):
        cmd = "python eval.py -o /tmp/results.json"
        with pytest.raises(PermissionError):
            validator_active.validate_shell_command(cmd)

    def test_no_output_paths_passes(self, validator_active):
        cmd = "python train.py --epochs 50 --lr 0.001"
        validator_active.validate_shell_command(cmd)


# ---------------------------------------------------------------------------
# TrialInfo model tests
# ---------------------------------------------------------------------------

class TestTrialInfo:
    def test_trial_name_formatting(self):
        t = TrialInfo(date="20260226", number=1, status="active", started_at="")
        assert t.trial_name == "trial_001"

    def test_trial_name_three_digit_padding(self):
        t = TrialInfo(date="20260226", number=42, status="active", started_at="")
        assert t.trial_name == "trial_042"

    def test_sandbox_path(self):
        t = TrialInfo(date="20260226", number=3, status="active", started_at="")
        assert str(t.sandbox_path) == "sandbox/20260226/trial_003"

    def test_report_path(self):
        t = TrialInfo(date="20260226", number=3, status="active", started_at="")
        assert str(t.report_path) == "experiment_reports/20260226/trial_003"

    def test_is_writable_active(self):
        t = TrialInfo(date="20260226", number=1, status="active", started_at="")
        assert t.is_writable is True

    def test_is_writable_review(self):
        t = TrialInfo(date="20260226", number=1, status="review", started_at="")
        assert t.is_writable is False

    def test_is_writable_approved(self):
        t = TrialInfo(date="20260226", number=1, status="approved", started_at="")
        assert t.is_writable is False

    def test_is_writable_rejected(self):
        t = TrialInfo(date="20260226", number=1, status="rejected", started_at="")
        assert t.is_writable is False

    def test_round_trip_serialization(self):
        t = TrialInfo(
            date="20260226", number=5, status="approved",
            started_at="2026-02-26T10:00:00",
            finished_at="2026-02-26T13:00:00",
            command_history=["python train.py"],
            goal="test cosine annealing",
        )
        assert TrialInfo.from_dict(t.to_dict()) == t

    def test_is_reviewable_review(self):
        t = TrialInfo(date="20260226", number=1, status="review", started_at="")
        assert t.is_reviewable is True

    def test_is_reviewable_active(self):
        t = TrialInfo(date="20260226", number=1, status="active", started_at="")
        assert t.is_reviewable is False

    def test_is_reviewable_approved(self):
        t = TrialInfo(date="20260226", number=1, status="approved", started_at="")
        assert t.is_reviewable is False

    def test_trial_status_backward_compat(self):
        """TrialStatus(str, Enum) should compare equal to plain strings."""
        from researchclaw.models import TrialStatus
        assert TrialStatus.ACTIVE == "active"
        assert TrialStatus.REVIEW == "review"
        assert TrialStatus.APPROVED == "approved"
        assert TrialStatus.REJECTED == "rejected"

    def test_trial_status_from_dict_round_trip(self):
        """from_dict should handle both raw strings and TrialStatus enum values."""
        from researchclaw.models import TrialStatus
        t = TrialInfo(
            date="20260226", number=1, status=TrialStatus.ACTIVE, started_at="now",
        )
        d = t.to_dict()
        assert d["status"] == "active"  # serialized as plain string
        t2 = TrialInfo.from_dict(d)
        assert t2.status == TrialStatus.ACTIVE
        assert t2.is_writable is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
