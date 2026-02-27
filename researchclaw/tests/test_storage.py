from __future__ import annotations

from pathlib import Path

from researchclaw.storage import StorageManager


def test_create_trial_copies_selected_project(tmp_path: Path) -> None:
    storage = StorageManager(str(tmp_path))
    storage.ensure_layout()

    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "train.py").write_text("print('ok')\n", encoding="utf-8")

    trial = storage.create_trial(selected_project="demo")

    assert (tmp_path / trial.sandbox_path / "codes" / "train.py").exists()
    assert (tmp_path / trial.sandbox_path / "outputs").exists()
    assert (tmp_path / trial.sandbox_path / "eval_codes").exists()
    assert (tmp_path / trial.results_path).exists()


def test_trial_number_increments_per_date(tmp_path: Path) -> None:
    storage = StorageManager(str(tmp_path))
    storage.ensure_layout()

    t1 = storage.create_trial(selected_project=None)
    t2 = storage.create_trial(selected_project=None)

    assert t1.date == t2.date
    assert t1.trial_number == 1
    assert t2.trial_number == 2


def test_experiment_log_format(tmp_path: Path) -> None:
    storage = StorageManager(str(tmp_path))
    storage.ensure_layout()
    trial = storage.create_trial(selected_project=None)
    storage.append_experiment_log(trial, "iters(exp=1, eval=1)", "results/x/REPORT.md")

    lines = storage.load_experiment_log_lines()
    assert any("Full Doc:" in line for line in lines)
    assert any(trial.date in line and trial.trial_name in line for line in lines)
