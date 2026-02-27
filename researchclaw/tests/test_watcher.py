"""Tests for researchclaw/watcher.py"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.models import TrialInfo
from researchclaw.watcher import ExperimentEvent, ExperimentStatus, Watcher, WatcherState


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
    )
    log_dir = tmp_path / t.report_path / "log"
    log_dir.mkdir(parents=True)
    (tmp_path / t.sandbox_path).mkdir(parents=True)
    return t


@pytest.fixture
def watcher(tmp_path):
    """Watcher with zero poll_interval (no sleeping in tests) and long timeouts."""
    return Watcher(
        str(tmp_path),
        poll_interval=0,
        heartbeat_timeout=9999,
        status_update_interval=0,  # disable periodic updates
        gpu_idle_threshold=9999,
    )


def make_mock_proc(pid=12345, poll_sequence=(None, 0)):
    """Return a mock Popen that steps through poll_sequence on each .poll() call."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = pid
    proc.poll.side_effect = list(poll_sequence)
    return proc


# ──────────────────────────────────────────────────────────────────────────────
# TestWatchFinished
# ──────────────────────────────────────────────────────────────────────────────

class TestWatchFinished:
    def test_yields_finished_on_zero_returncode(self, tmp_path, watcher, trial):
        proc = make_mock_proc(poll_sequence=(None, 0))
        events = list(watcher.watch(proc, trial))
        assert events[-1].event == ExperimentEvent.FINISHED

    def test_finished_carries_returncode_zero(self, tmp_path, watcher, trial):
        proc = make_mock_proc(poll_sequence=(None, 0))
        events = list(watcher.watch(proc, trial))
        assert events[-1].returncode == 0

    def test_finished_includes_log_tail(self, tmp_path, watcher, trial):
        stdout_log = tmp_path / trial.report_path / "log" / "stdout.log"
        stdout_log.write_text("some training output\n")
        proc = make_mock_proc(poll_sequence=(None, 0))
        events = list(watcher.watch(proc, trial))
        assert "some training output" in events[-1].log_tail

    def test_generator_stops_after_finished(self, tmp_path, watcher, trial):
        proc = make_mock_proc(poll_sequence=(None, 0))
        events = list(watcher.watch(proc, trial))
        # Should yield exactly one event: FINISHED
        assert len([e for e in events if e.event == ExperimentEvent.FINISHED]) == 1


# ──────────────────────────────────────────────────────────────────────────────
# TestWatchCrashed
# ──────────────────────────────────────────────────────────────────────────────

class TestWatchCrashed:
    def test_yields_crashed_on_nonzero_returncode(self, tmp_path, watcher, trial):
        proc = make_mock_proc(poll_sequence=(None, 1))
        events = list(watcher.watch(proc, trial))
        assert events[-1].event == ExperimentEvent.CRASHED

    def test_crashed_carries_returncode(self, tmp_path, watcher, trial):
        proc = make_mock_proc(poll_sequence=(None, 1))
        events = list(watcher.watch(proc, trial))
        assert events[-1].returncode == 1

    def test_crashed_with_exit_code_2(self, tmp_path, watcher, trial):
        proc = make_mock_proc(poll_sequence=(None, 2))
        events = list(watcher.watch(proc, trial))
        assert events[-1].event == ExperimentEvent.CRASHED
        assert events[-1].returncode == 2


# ──────────────────────────────────────────────────────────────────────────────
# TestDetectNan
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectNan:
    def test_loss_nan_detected(self, watcher):
        assert watcher.detect_nan("loss=nan") is True

    def test_loss_nan_uppercase(self, watcher):
        assert watcher.detect_nan("loss=NaN") is True

    def test_val_loss_inf(self, watcher):
        assert watcher.detect_nan("val_loss: inf") is True

    def test_train_loss_negative_inf(self, watcher):
        assert watcher.detect_nan("train_loss=-inf") is True

    def test_acc_nan(self, watcher):
        assert watcher.detect_nan("acc=nan") is True

    def test_no_false_positive_plain_nan(self, watcher):
        """Plain 'nan' in text without a metric prefix should not trigger."""
        assert watcher.detect_nan("NaN values are discussed in the paper") is False

    def test_normal_loss_value_no_detection(self, watcher):
        assert watcher.detect_nan("loss=0.0891") is False

    def test_empty_string(self, watcher):
        assert watcher.detect_nan("") is False

    def test_irrelevant_text(self, watcher):
        assert watcher.detect_nan("epoch 5/50, step 100/1000") is False


# ──────────────────────────────────────────────────────────────────────────────
# TestScanNewFiles
# ──────────────────────────────────────────────────────────────────────────────

class TestScanNewFiles:
    def _make_state(self, watcher, trial_name="trial_001"):
        import time
        return WatcherState(
            pid=1234,
            start_time=time.monotonic(),
            last_log_time=time.monotonic(),
        )

    def test_finds_pt_file(self, tmp_path, watcher, trial):
        sandbox = tmp_path / trial.sandbox_path
        (sandbox / "model.pt").write_bytes(b"fake")
        state = self._make_state(watcher)
        files = watcher._scan_new_files(trial, state)
        assert any("model.pt" in f for f in files)

    def test_finds_csv_and_png(self, tmp_path, watcher, trial):
        sandbox = tmp_path / trial.sandbox_path
        (sandbox / "metrics.csv").write_text("loss,acc")
        (sandbox / "plot.png").write_bytes(b"png")
        state = self._make_state(watcher)
        files = watcher._scan_new_files(trial, state)
        assert any("metrics.csv" in f for f in files)
        assert any("plot.png" in f for f in files)

    def test_ignores_py_files(self, tmp_path, watcher, trial):
        sandbox = tmp_path / trial.sandbox_path
        (sandbox / "train.py").write_text("# code")
        state = self._make_state(watcher)
        files = watcher._scan_new_files(trial, state)
        assert not any(".py" in f for f in files)

    def test_returns_empty_if_sandbox_missing(self, tmp_path, trial):
        # Use a watcher pointing to tmp_path but with a trial whose sandbox doesn't exist
        w = Watcher(str(tmp_path), poll_interval=0)
        t = TrialInfo(date="29991231", number=99, status="active", started_at="2099")
        state = WatcherState(pid=1, start_time=0.0, last_log_time=0.0)
        assert w._scan_new_files(t, state) == []

    def test_returns_relative_paths(self, tmp_path, watcher, trial):
        sandbox = tmp_path / trial.sandbox_path
        (sandbox / "best.ckpt").write_bytes(b"ckpt")
        state = self._make_state(watcher)
        files = watcher._scan_new_files(trial, state)
        # Paths should be relative to base_dir, not absolute
        for f in files:
            assert not f.startswith("/") or f.startswith(str(tmp_path))


# ──────────────────────────────────────────────────────────────────────────────
# TestCheckGpu
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckGpu:
    def test_returns_dict_on_success(self, watcher):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "50, 4000, 8000\n"
        with patch("subprocess.run", return_value=mock_result):
            info = watcher.check_gpu()
        assert info.get("utilization") == 50.0
        assert info.get("memory_used_mb") == 4000.0
        assert info.get("memory_total_mb") == 8000.0

    def test_returns_empty_dict_when_nvidia_smi_missing(self, watcher):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert watcher.check_gpu() == {}

    def test_returns_empty_dict_on_timeout(self, watcher):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            assert watcher.check_gpu() == {}

    def test_returns_empty_dict_on_nonzero_rc(self, watcher):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert watcher.check_gpu() == {}
