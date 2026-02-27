"""Tests for researchclaw/summarizer.py"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from researchclaw.access_control import PathValidator
from researchclaw.models import TrialInfo
from researchclaw.summarizer import Summarizer
from researchclaw.watcher import ExperimentEvent, ExperimentStatus


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def trial(tmp_path):
    t = TrialInfo(
        date="20260225",
        number=1,
        status="active",
        started_at="2026-02-25T10:00:00",
        goal="test cosine LR schedule",
    )
    (tmp_path / t.report_path / "log").mkdir(parents=True)
    (tmp_path / t.sandbox_path).mkdir(parents=True)
    return t


@pytest.fixture
def validator(tmp_path, trial):
    v = PathValidator(str(tmp_path))
    v.set_trial(trial)
    return v


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete.return_value = (
        "OBSERVATIONS:\n"
        "Loss decreased steadily from 0.5 to 0.09. "
        "The cosine schedule helped stabilize training.\n\n"
        "SUGGESTED NEXT DIRECTIONS:\n"
        "Try a lower learning rate of 1e-4 for the final epochs."
    )
    return llm


@pytest.fixture
def summarizer(tmp_path, mock_llm, validator):
    return Summarizer(mock_llm, validator, str(tmp_path), log_tail_lines=50)


def _make_status(event, trial_name="trial_001", duration="5m 30s",
                 log_tail="some log output", gpu_info=None,
                 new_files=None, returncode=None, message=""):
    return ExperimentStatus(
        event=event,
        trial_name=trial_name,
        duration=duration,
        log_tail=log_tail,
        gpu_info=gpu_info or {},
        new_files=new_files or [],
        returncode=returncode,
        message=message,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TestFormatStatusMessage
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatStatusMessage:
    def test_finished_contains_trial_name(self, summarizer):
        status = _make_status(ExperimentEvent.FINISHED, trial_name="trial_042")
        msg = summarizer.format_status_message(status)
        assert "trial_042" in msg

    def test_finished_contains_runtime(self, summarizer):
        status = _make_status(ExperimentEvent.FINISHED, duration="3h 14m")
        msg = summarizer.format_status_message(status)
        assert "3h 14m" in msg

    def test_finished_with_gpu_info(self, summarizer):
        status = _make_status(
            ExperimentEvent.FINISHED,
            gpu_info={"utilization": 80.0, "memory_used_mb": 4000.0, "memory_total_mb": 8000.0},
        )
        msg = summarizer.format_status_message(status)
        assert "80%" in msg

    def test_finished_no_gpu_info(self, summarizer):
        status = _make_status(ExperimentEvent.FINISHED, gpu_info={})
        msg = summarizer.format_status_message(status)
        assert "not available" in msg

    def test_finished_contains_checkmark(self, summarizer):
        status = _make_status(ExperimentEvent.FINISHED)
        msg = summarizer.format_status_message(status)
        assert "✅" in msg

    def test_crashed_contains_exit_code(self, summarizer):
        status = _make_status(ExperimentEvent.CRASHED, returncode=1)
        msg = summarizer.format_status_message(status)
        assert "1" in msg
        assert "CRASHED" in msg or "❌" in msg

    def test_hung_asks_to_kill(self, summarizer):
        status = _make_status(
            ExperimentEvent.HUNG,
            message="No log output for 5m 0s. Process may be hung.",
        )
        msg = summarizer.format_status_message(status)
        assert "kill" in msg.lower() or "⚠️" in msg

    def test_nan_detected_message(self, summarizer):
        status = _make_status(
            ExperimentEvent.NAN_DETECTED,
            message="NaN or Inf detected in training output.",
        )
        msg = summarizer.format_status_message(status)
        assert "⚠️" in msg or "NaN" in msg or "unstable" in msg.lower()

    def test_status_update_contains_running(self, summarizer):
        status = _make_status(
            ExperimentEvent.STATUS_UPDATE,
            gpu_info={"utilization": 75.0},
        )
        msg = summarizer.format_status_message(status)
        assert "running" in msg.lower() or "📊" in msg


# ──────────────────────────────────────────────────────────────────────────────
# TestParseMetrics
# ──────────────────────────────────────────────────────────────────────────────

class TestParseMetrics:
    def test_delegates_to_utils(self, summarizer):
        metrics = summarizer.parse_metrics("loss=0.0891")
        assert "loss" in metrics
        assert abs(metrics["loss"] - 0.0891) < 1e-6

    def test_empty_log(self, summarizer):
        assert summarizer.parse_metrics("") == {}


# ──────────────────────────────────────────────────────────────────────────────
# TestGenerateReport
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_report_file_written(self, tmp_path, summarizer, trial):
        summarizer.generate_report(trial, diff="+ lr=0.001")
        assert (tmp_path / trial.report_path / "REPORT.md").exists()

    def test_report_contains_trial_name(self, tmp_path, summarizer, trial):
        report = summarizer.generate_report(trial, diff="")
        assert "trial_001" in report

    def test_report_contains_goal(self, tmp_path, summarizer, trial):
        report = summarizer.generate_report(trial, diff="")
        assert "test cosine LR schedule" in report

    def test_report_contains_observations_from_llm(self, tmp_path, summarizer, trial):
        report = summarizer.generate_report(trial, diff="+ lr=0.001")
        assert "Loss decreased steadily" in report

    def test_report_contains_suggestions_from_llm(self, tmp_path, summarizer, trial):
        report = summarizer.generate_report(trial, diff="+ lr=0.001")
        assert "lower learning rate" in report

    def test_report_includes_metrics_from_log(self, tmp_path, summarizer, trial):
        stdout_log = tmp_path / trial.report_path / "log" / "stdout.log"
        stdout_log.write_text("epoch 10 | loss=0.05 | val_acc=0.95\n")
        report = summarizer.generate_report(trial, diff="")
        # Metrics section should reference parsed metrics
        assert "loss" in report

    def test_report_no_log_file_does_not_raise(self, tmp_path, summarizer, trial):
        # No stdout.log — should not raise
        report = summarizer.generate_report(trial, diff="")
        assert "trial_001" in report

    def test_llm_failure_fallback(self, tmp_path, mock_llm, validator, trial):
        mock_llm.complete.side_effect = RuntimeError("API is down")
        s = Summarizer(mock_llm, validator, str(tmp_path), log_tail_lines=50)
        report = s.generate_report(trial, diff="")
        assert "LLM analysis failed" in report
        assert (tmp_path / trial.report_path / "REPORT.md").exists()

    def test_write_denied_raises(self, tmp_path, mock_llm, trial):
        # Validator with no active trial → writes to report_path are denied
        v = PathValidator(str(tmp_path))
        # No set_trial() called, so no write access
        s = Summarizer(mock_llm, v, str(tmp_path), log_tail_lines=50)
        with pytest.raises(PermissionError):
            s.generate_report(trial, diff="")
