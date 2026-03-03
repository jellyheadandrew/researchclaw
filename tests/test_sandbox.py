from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from researchclaw.sandbox import SandboxManager


class TestSandboxInitialize:
    """Tests for SandboxManager.initialize()."""

    def test_initialize_creates_directory_structure(self, tmp_path: Path) -> None:
        sandbox = SandboxManager.initialize(tmp_path)

        assert sandbox == tmp_path / "sandbox_researchclaw"
        assert (sandbox / "project_settings").is_dir()
        assert (sandbox / "project_settings" / "researchclaw.yaml").is_file()
        assert (sandbox / "project_settings" / "PROJECT_MEMORY.md").is_file()
        assert (sandbox / "EXPERIMENT_LOGS.md").is_file()
        assert (sandbox / "experiments").is_dir()

    def test_initialize_creates_default_config(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        config_path = tmp_path / "sandbox_researchclaw" / "project_settings" / "researchclaw.yaml"

        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert data["model"] == "claude-opus-4-6"
        assert data["max_trials"] == 20
        assert data["max_retries"] == 5
        assert data["autopilot"] is False
        assert data["experiment_timeout_seconds"] == 3600
        assert data["python_command"] == "python3"

    def test_initialize_empty_files(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)

        memory = tmp_path / "sandbox_researchclaw" / "project_settings" / "PROJECT_MEMORY.md"
        assert memory.read_text() == ""

        logs = tmp_path / "sandbox_researchclaw" / "EXPERIMENT_LOGS.md"
        assert logs.read_text() == ""

    def test_initialize_idempotent(self, tmp_path: Path) -> None:
        """Calling initialize twice should not overwrite existing files."""
        SandboxManager.initialize(tmp_path)

        # Write custom content to an existing file
        logs = tmp_path / "sandbox_researchclaw" / "EXPERIMENT_LOGS.md"
        logs.write_text("existing content")

        config = tmp_path / "sandbox_researchclaw" / "project_settings" / "researchclaw.yaml"
        original_config = config.read_text()

        # Initialize again
        SandboxManager.initialize(tmp_path)

        # Files should not be overwritten
        assert logs.read_text() == "existing content"
        assert config.read_text() == original_config


class TestSandboxIsInitialized:
    """Tests for SandboxManager.is_initialized()."""

    def test_not_initialized(self, tmp_path: Path) -> None:
        assert SandboxManager.is_initialized(tmp_path) is False

    def test_initialized(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        assert SandboxManager.is_initialized(tmp_path) is True


class TestSandboxCreateTrial:
    """Tests for SandboxManager.create_trial()."""

    def test_create_first_trial(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)

        assert trial_dir.name == "20260302_trial_001"
        assert trial_dir.is_dir()
        assert (trial_dir / "experiment" / "codes_exp").is_dir()
        assert (trial_dir / "experiment" / "codes_eval").is_dir()
        assert (trial_dir / "experiment" / "outputs").is_dir()
        assert (trial_dir / "requirements.txt").is_file()
        assert (trial_dir / "meta.json").is_file()

    def test_create_sequential_trials_same_date(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)

        trial1 = SandboxManager.create_trial(tmp_path, date=date)
        trial2 = SandboxManager.create_trial(tmp_path, date=date)
        trial3 = SandboxManager.create_trial(tmp_path, date=date)

        assert trial1.name == "20260302_trial_001"
        assert trial2.name == "20260302_trial_002"
        assert trial3.name == "20260302_trial_003"

    def test_trial_number_resets_per_date(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date1 = datetime(2026, 3, 2, tzinfo=timezone.utc)
        date2 = datetime(2026, 3, 3, tzinfo=timezone.utc)

        trial1 = SandboxManager.create_trial(tmp_path, date=date1)
        trial2 = SandboxManager.create_trial(tmp_path, date=date1)
        trial3 = SandboxManager.create_trial(tmp_path, date=date2)

        assert trial1.name == "20260302_trial_001"
        assert trial2.name == "20260302_trial_002"
        assert trial3.name == "20260303_trial_001"

    def test_meta_json_created(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)
        trial_dir = SandboxManager.create_trial(tmp_path, date=date)

        meta_path = trial_dir / "meta.json"
        with open(meta_path) as f:
            data = json.load(f)

        assert data["trial_number"] == 1
        assert data["status"] == "pending"
        assert data["state"] == "experiment_plan"
        assert data["plan_approved_at"] is None
        assert data["experiment_exit_code"] is None
        assert data["eval_exit_code"] is None
        assert data["decision"] is None
        assert data["decision_reasoning"] is None

    def test_meta_json_trial_number_matches(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        date = datetime(2026, 3, 2, tzinfo=timezone.utc)

        SandboxManager.create_trial(tmp_path, date=date)
        trial2 = SandboxManager.create_trial(tmp_path, date=date)

        with open(trial2 / "meta.json") as f:
            data = json.load(f)

        assert data["trial_number"] == 2

    def test_requirements_txt_empty(self, tmp_path: Path) -> None:
        SandboxManager.initialize(tmp_path)
        trial_dir = SandboxManager.create_trial(tmp_path)

        assert (trial_dir / "requirements.txt").read_text() == ""
