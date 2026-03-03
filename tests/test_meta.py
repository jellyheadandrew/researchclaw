from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from researchclaw.cli import main
from researchclaw.models import TrialMeta
from researchclaw.sandbox import SandboxManager


class TestTrialMetaJson:
    """Tests for TrialMeta.to_json() and from_json()."""

    def test_to_json_creates_file(self, tmp_path: Path) -> None:
        meta = TrialMeta(trial_number=1)
        path = tmp_path / "meta.json"
        meta.to_json(path)
        assert path.is_file()

    def test_to_json_content(self, tmp_path: Path) -> None:
        meta = TrialMeta(
            trial_number=3,
            status="running",
            state="experiment_execute",
            created_at="2026-03-01T10:00:00+00:00",
            updated_at="2026-03-01T10:35:00+00:00",
            plan_approved_at="2026-03-01T10:05:00+00:00",
            experiment_exit_code=None,
            eval_exit_code=None,
            decision=None,
            decision_reasoning=None,
        )
        path = tmp_path / "meta.json"
        meta.to_json(path)

        with open(path) as f:
            data = json.load(f)

        assert data["trial_number"] == 3
        assert data["status"] == "running"
        assert data["state"] == "experiment_execute"
        assert data["plan_approved_at"] == "2026-03-01T10:05:00+00:00"

    def test_from_json_roundtrip(self, tmp_path: Path) -> None:
        original = TrialMeta(
            trial_number=2,
            status="running",
            state="eval_implement",
            created_at="2026-03-01T10:00:00+00:00",
            updated_at="2026-03-01T11:00:00+00:00",
            plan_approved_at="2026-03-01T10:05:00+00:00",
            experiment_exit_code=0,
            eval_exit_code=None,
            decision=None,
            decision_reasoning=None,
        )
        path = tmp_path / "meta.json"
        original.to_json(path)
        loaded = TrialMeta.from_json(path)

        assert loaded.trial_number == original.trial_number
        assert loaded.status == original.status
        assert loaded.state == original.state
        assert loaded.created_at == original.created_at
        assert loaded.updated_at == original.updated_at
        assert loaded.plan_approved_at == original.plan_approved_at
        assert loaded.experiment_exit_code == original.experiment_exit_code
        assert loaded.eval_exit_code == original.eval_exit_code
        assert loaded.decision == original.decision
        assert loaded.decision_reasoning == original.decision_reasoning

    def test_to_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        meta = TrialMeta()
        path = tmp_path / "nested" / "dir" / "meta.json"
        meta.to_json(path)
        assert path.is_file()

    def test_from_json_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TrialMeta.from_json(tmp_path / "nonexistent.json")


class TestSandboxManagerMeta:
    """Tests for SandboxManager meta persistence methods."""

    def test_save_trial_meta(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)

        meta = TrialMeta(
            trial_number=1,
            status="running",
            state="experiment_implement",
        )
        SandboxManager.save_trial_meta(trial_dir, meta)

        with open(trial_dir / "meta.json") as f:
            data = json.load(f)
        assert data["status"] == "running"
        assert data["state"] == "experiment_implement"

    def test_get_trial_meta(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)

        meta = SandboxManager.get_trial_meta(trial_dir)
        assert meta.trial_number == 1
        assert meta.status == "pending"
        assert meta.state == "experiment_plan"

    def test_save_and_get_roundtrip(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)

        meta = TrialMeta(
            trial_number=1,
            status="running",
            state="experiment_execute",
            created_at="2026-03-02T10:00:00+00:00",
            updated_at="2026-03-02T10:30:00+00:00",
            plan_approved_at="2026-03-02T10:05:00+00:00",
            experiment_exit_code=None,
            eval_exit_code=None,
            decision=None,
            decision_reasoning=None,
        )
        SandboxManager.save_trial_meta(trial_dir, meta)
        loaded = SandboxManager.get_trial_meta(trial_dir)

        assert loaded.trial_number == 1
        assert loaded.status == "running"
        assert loaded.state == "experiment_execute"
        assert loaded.plan_approved_at == "2026-03-02T10:05:00+00:00"

    def test_get_latest_trial_none_when_empty(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        assert SandboxManager.get_latest_trial(tmp_path) is None

    def test_get_latest_trial_returns_most_recent(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        SandboxManager.create_trial(tmp_path, date=date)
        trial2 = SandboxManager.create_trial(tmp_path, date=date)

        latest = SandboxManager.get_latest_trial(tmp_path)
        assert latest == trial2

    def test_get_latest_trial_across_dates(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date1 = datetime(2026, 3, 2, tzinfo=timezone.utc)
        date2 = datetime(2026, 3, 3, tzinfo=timezone.utc)
        SandboxManager.create_trial(tmp_path, date=date1)
        trial2 = SandboxManager.create_trial(tmp_path, date=date2)

        latest = SandboxManager.get_latest_trial(tmp_path)
        assert latest == trial2

    def test_get_latest_trial_no_experiments_dir(self, tmp_path: Path) -> None:
        # No sandbox at all
        assert SandboxManager.get_latest_trial(tmp_path) is None


class TestCLIResume:
    """Tests for CLI resume logic."""

    def _patch_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Common patches to avoid real terminal interaction in CLI tests."""
        import researchclaw.cli as cli_mod
        monkeypatch.setattr(cli_mod, "needs_onboarding", lambda: False)
        monkeypatch.setattr(cli_mod, "TerminalChat", lambda **kw: None)
        monkeypatch.setattr(cli_mod, "_build_handlers", lambda: {})

        class FakeEngine:
            def __init__(self, *a, **kw):
                pass
            def run(self):
                raise SystemExit("done")
        monkeypatch.setattr(cli_mod, "FSMEngine", FakeEngine)

    def test_cli_no_trials_yet(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_cli(monkeypatch)
        SandboxManager.initialize(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "No trials yet." in result.output

    def test_cli_resume_shows_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_cli(monkeypatch)
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)

        # Update meta to a specific state
        meta = TrialMeta(
            trial_number=1,
            status="running",
            state="experiment_implement",
        )
        SandboxManager.save_trial_meta(trial_dir, meta)

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Resuming trial 20260302_trial_001" in result.output
        assert "state=experiment_implement" in result.output
        assert "status=running" in result.output

    def test_cli_completed_trial_shows_decide(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Completed trial is resumed at DECIDE state."""
        self._patch_cli(monkeypatch)
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)

        meta = TrialMeta(trial_number=1, status="completed", state="experiment_report")
        SandboxManager.save_trial_meta(trial_dir, meta)

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path)])
        assert "Entering DECIDE state" in result.output

        # Verify meta.json was updated to decide state
        loaded_meta = SandboxManager.get_trial_meta(trial_dir)
        assert loaded_meta.state == "decide"
