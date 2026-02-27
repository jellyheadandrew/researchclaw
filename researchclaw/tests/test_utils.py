"""Tests for researchclaw/utils.py"""

from __future__ import annotations

import subprocess

import pytest

from researchclaw.utils import format_duration, parse_metrics_from_log, run_subprocess, tail_file


# ──────────────────────────────────────────────────────────────────────────────
# format_duration
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatDuration:
    def test_zero_seconds(self):
        assert format_duration(0) == "0s"

    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_one_minute_boundary(self):
        assert format_duration(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert format_duration(125) == "2m 5s"

    def test_one_hour_boundary(self):
        assert format_duration(3600) == "1h 0m"

    def test_hours_and_minutes(self):
        # 5400s = 90 min = 1h 30m
        assert format_duration(5400) == "1h 30m"

    def test_fractional_seconds_truncated(self):
        # float input — should truncate to int seconds
        result = format_duration(1.9)
        assert result == "1s"


# ──────────────────────────────────────────────────────────────────────────────
# tail_file
# ──────────────────────────────────────────────────────────────────────────────

class TestTailFile:
    def test_returns_last_n_lines(self, tmp_path):
        p = tmp_path / "log.txt"
        lines = [f"line {i}" for i in range(100)]
        p.write_text("\n".join(lines))
        result = tail_file(str(p), n=10)
        result_lines = result.splitlines()
        assert len(result_lines) == 10
        assert result_lines[-1] == "line 99"
        assert result_lines[0] == "line 90"

    def test_fewer_than_n_lines_returns_all(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("a\nb\nc")
        result = tail_file(str(p), n=50)
        assert result.splitlines() == ["a", "b", "c"]

    def test_nonexistent_file_returns_empty_string(self, tmp_path):
        result = tail_file(str(tmp_path / "missing.log"), n=10)
        assert result == ""

    def test_empty_file_returns_empty_string(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("")
        result = tail_file(str(p), n=10)
        assert result == ""

    def test_exact_n_lines(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("x\ny\nz")
        result = tail_file(str(p), n=3)
        assert result.splitlines() == ["x", "y", "z"]


# ──────────────────────────────────────────────────────────────────────────────
# parse_metrics_from_log
# ──────────────────────────────────────────────────────────────────────────────

class TestParseMetricsFromLog:
    def test_loss_equals_pattern(self):
        metrics = parse_metrics_from_log("loss=0.0891")
        assert "loss" in metrics
        assert abs(metrics["loss"] - 0.0891) < 1e-6

    def test_loss_colon_pattern(self):
        metrics = parse_metrics_from_log("loss: 0.5")
        assert abs(metrics["loss"] - 0.5) < 1e-6

    def test_val_acc_percent(self):
        metrics = parse_metrics_from_log("val_acc: 74.2%")
        assert "val_acc" in metrics
        assert abs(metrics["val_acc"] - 0.742) < 1e-4

    def test_pipe_format(self):
        metrics = parse_metrics_from_log("epoch 10/50 | loss: 0.089 | acc: 0.742")
        assert "loss" in metrics
        assert "acc" in metrics

    def test_multiple_metrics(self):
        log = "loss=0.1\nval_loss=0.2\nacc=0.9"
        metrics = parse_metrics_from_log(log)
        assert "loss" in metrics
        assert "val_loss" in metrics
        assert "acc" in metrics

    def test_last_value_wins(self):
        # When a metric appears multiple times, the last value should be kept
        log = "loss=0.5\nloss=0.2"
        metrics = parse_metrics_from_log(log)
        assert abs(metrics["loss"] - 0.2) < 1e-6

    def test_empty_log_returns_empty_dict(self):
        assert parse_metrics_from_log("") == {}

    def test_no_metrics_in_log(self):
        assert parse_metrics_from_log("Starting training...") == {}

    def test_scientific_notation(self):
        metrics = parse_metrics_from_log("lr=1e-3")
        assert "lr" in metrics
        assert abs(metrics["lr"] - 0.001) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# run_subprocess
# ──────────────────────────────────────────────────────────────────────────────

class TestRunSubprocess:
    def test_simple_echo(self, tmp_path):
        rc, stdout, stderr = run_subprocess(["echo", "hello"], cwd=str(tmp_path))
        assert rc == 0
        assert "hello" in stdout
        assert stderr == ""

    def test_nonzero_returncode(self, tmp_path):
        rc, stdout, stderr = run_subprocess(["false"], cwd=str(tmp_path))
        assert rc != 0

    def test_timeout_raises(self, tmp_path):
        with pytest.raises(subprocess.TimeoutExpired):
            run_subprocess(["sleep", "10"], cwd=str(tmp_path), timeout=1)

    def test_stderr_captured(self, tmp_path):
        rc, stdout, stderr = run_subprocess(
            ["bash", "-c", "echo errtext >&2"],
            cwd=str(tmp_path),
        )
        assert "errtext" in stderr
