"""Tests for researchclaw/sandbox_manager.py"""

from __future__ import annotations

import json

import pytest

from researchclaw.access_control import PathValidator
from researchclaw.models import TrialInfo
from researchclaw.sandbox_manager import SandboxManager


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_dir(tmp_path):
    """Set up a base directory with github_codes/ and an empty .trials.jsonl."""
    github = tmp_path / "github_codes"
    github.mkdir()
    (github / "train.py").write_text("# training script")
    (github / "model.py").write_text("# model definition")
    (tmp_path / ".trials.jsonl").write_text("")
    return tmp_path


@pytest.fixture
def validator(base_dir):
    return PathValidator(str(base_dir))


@pytest.fixture
def mgr(base_dir, validator):
    return SandboxManager(str(base_dir), validator)


# ──────────────────────────────────────────────────────────────────────────────
# TestCreateTrial
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateTrial:
    def test_creates_sandbox_directory(self, base_dir, mgr):
        trial = mgr.create_trial()
        assert (base_dir / trial.sandbox_path).exists()
        assert (base_dir / trial.sandbox_path).is_dir()

    def test_copies_github_codes_into_sandbox(self, base_dir, mgr):
        trial = mgr.create_trial()
        assert (base_dir / trial.sandbox_path / "train.py").exists()
        assert (base_dir / trial.sandbox_path / "model.py").exists()

    def test_creates_log_directory(self, base_dir, mgr):
        trial = mgr.create_trial()
        assert (base_dir / trial.report_path / "log").is_dir()

    def test_creates_eval_results_directory(self, base_dir, mgr):
        trial = mgr.create_trial()
        assert (base_dir / trial.report_path / "eval_results").is_dir()

    def test_returns_active_status(self, base_dir, mgr):
        trial = mgr.create_trial()
        assert trial.status == "active"

    def test_trial_number_is_one_for_first_trial(self, base_dir, mgr):
        trial = mgr.create_trial()
        assert trial.number == 1

    def test_trial_number_increments(self, base_dir, mgr):
        t1 = mgr.create_trial()
        # Finalize first so we can create a second (same date directory needs different number)
        mgr.finalize_trial(t1, "rejected")
        t2 = mgr.create_trial()
        assert t2.number == t1.number + 1

    def test_persists_to_trials_jsonl(self, base_dir, mgr):
        trial = mgr.create_trial()
        lines = (base_dir / ".trials.jsonl").read_text().strip().splitlines()
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["number"] == trial.number
        assert data["status"] == "active"

    def test_goal_stored_in_trial(self, base_dir, mgr):
        trial = mgr.create_trial(goal="test cosine LR schedule")
        assert trial.goal == "test cosine LR schedule"

    def test_raises_if_github_codes_missing(self, tmp_path):
        v = PathValidator(str(tmp_path))
        m = SandboxManager(str(tmp_path), v)
        with pytest.raises(FileNotFoundError):
            m.create_trial()

    def test_grants_write_access(self, base_dir, mgr, validator):
        trial = mgr.create_trial()
        sandbox_file = str(base_dir / trial.sandbox_path / "new_file.py")
        assert validator.can_write(sandbox_file)


# ──────────────────────────────────────────────────────────────────────────────
# TestFinalizeTrial
# ──────────────────────────────────────────────────────────────────────────────

class TestFinalizeTrial:
    def test_finalize_approved_sets_status(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.finalize_trial(trial, "approved")
        assert trial.status == "approved"

    def test_finalize_rejected_sets_status(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.finalize_trial(trial, "rejected")
        assert trial.status == "rejected"

    def test_finalize_sets_finished_at(self, base_dir, mgr):
        trial = mgr.create_trial()
        assert trial.finished_at is None
        mgr.finalize_trial(trial, "approved")
        assert trial.finished_at is not None

    def test_finalize_revokes_write_access(self, base_dir, mgr, validator):
        trial = mgr.create_trial()
        mgr.finalize_trial(trial, "approved")
        sandbox_file = str(base_dir / trial.sandbox_path / "new_file.py")
        assert not validator.can_write(sandbox_file)

    def test_finalize_invalid_status_raises(self, base_dir, mgr):
        trial = mgr.create_trial()
        with pytest.raises(ValueError):
            mgr.finalize_trial(trial, "invalid_status")

    def test_finalize_persists_state(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.finalize_trial(trial, "approved")
        # The last line in .trials.jsonl should reflect the approved status
        lines = (base_dir / ".trials.jsonl").read_text().strip().splitlines()
        last_entry = json.loads(lines[-1])
        assert last_entry["status"] == "approved"


# ──────────────────────────────────────────────────────────────────────────────
# TestMarkReview
# ──────────────────────────────────────────────────────────────────────────────

class TestMarkReview:
    def test_mark_review_transitions_status(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        assert trial.status == "review"

    def test_mark_review_revokes_write_access(self, base_dir, mgr, validator):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        # is_writable is False for "review" status — write access is revoked
        sandbox_file = str(base_dir / trial.sandbox_path / "file.py")
        assert not validator.can_write(sandbox_file)


# ──────────────────────────────────────────────────────────────────────────────
# TestListTrials
# ──────────────────────────────────────────────────────────────────────────────

class TestListTrials:
    def test_list_all_trials(self, base_dir, mgr):
        t1 = mgr.create_trial()
        mgr.finalize_trial(t1, "rejected")
        t2 = mgr.create_trial()
        mgr.finalize_trial(t2, "rejected")
        trials = mgr.list_trials()
        assert len(trials) == 2

    def test_list_filter_by_date_no_match(self, base_dir, mgr):
        mgr.create_trial()
        trials = mgr.list_trials(date="29991231")
        assert trials == []

    def test_deduplication_keeps_latest_state(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        # Now .trials.jsonl has two entries for the same trial; latest should win
        trials = mgr.list_trials()
        assert len(trials) == 1
        assert trials[0].status == "review"


# ──────────────────────────────────────────────────────────────────────────────
# TestGetActiveTrial
# ──────────────────────────────────────────────────────────────────────────────

class TestGetActiveTrial:
    def test_returns_active_trial(self, base_dir, mgr):
        trial = mgr.create_trial()
        active = mgr.get_active_trial()
        assert active is not None
        assert active.number == trial.number

    def test_returns_none_after_finalize(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.finalize_trial(trial, "approved")
        assert mgr.get_active_trial() is None

    def test_returns_none_on_empty_jsonl(self, tmp_path):
        v = PathValidator(str(tmp_path))
        m = SandboxManager(str(tmp_path), v)
        (tmp_path / ".trials.jsonl").write_text("")
        assert m.get_active_trial() is None

    def test_returns_none_when_no_jsonl_exists(self, tmp_path):
        v = PathValidator(str(tmp_path))
        m = SandboxManager(str(tmp_path), v)
        assert m.get_active_trial() is None


# ──────────────────────────────────────────────────────────────────────────────
# TestReactivateTrial
# ──────────────────────────────────────────────────────────────────────────────

class TestReactivateTrial:
    def test_reactivate_transitions_review_to_active(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        assert trial.status == "review"
        mgr.reactivate_trial(trial)
        assert trial.status == "active"

    def test_reactivate_restores_write_access(self, base_dir, mgr, validator):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        sandbox_file = str(base_dir / trial.sandbox_path / "new_file.py")
        assert not validator.can_write(sandbox_file)  # review = not writable
        mgr.reactivate_trial(trial)
        assert validator.can_write(sandbox_file)  # reactivated = writable

    def test_reactivate_raises_if_not_review(self, base_dir, mgr):
        trial = mgr.create_trial()
        # Trial is "active", not "review" — should raise
        with pytest.raises(ValueError, match="Can only reactivate"):
            mgr.reactivate_trial(trial)

    def test_reactivate_raises_if_approved(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.finalize_trial(trial, "approved")
        with pytest.raises(ValueError, match="Can only reactivate"):
            mgr.reactivate_trial(trial)

    def test_reactivate_persists_state(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        mgr.reactivate_trial(trial)
        lines = (base_dir / ".trials.jsonl").read_text().strip().splitlines()
        last_entry = json.loads(lines[-1])
        assert last_entry["status"] == "active"


# ──────────────────────────────────────────────────────────────────────────────
# TestGetReviewTrial
# ──────────────────────────────────────────────────────────────────────────────

class TestGetReviewTrial:
    def test_returns_review_trial(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        review = mgr.get_review_trial()
        assert review is not None
        assert review.status == "review"

    def test_returns_none_after_finalize(self, base_dir, mgr):
        trial = mgr.create_trial()
        mgr.mark_review(trial)
        mgr.finalize_trial(trial, "approved")
        assert mgr.get_review_trial() is None

    def test_returns_none_on_empty_jsonl(self, tmp_path):
        v = PathValidator(str(tmp_path))
        m = SandboxManager(str(tmp_path), v)
        (tmp_path / ".trials.jsonl").write_text("")
        assert m.get_review_trial() is None
