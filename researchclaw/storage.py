from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Settings, TrialRecord
from .states import State, TrialStatus


class StorageManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.projects_dir = self.base_dir / "projects"
        self.sandbox_dir = self.base_dir / "sandbox"
        self.results_dir = self.base_dir / "results"
        self.references_dir = self.base_dir / "references"
        self.logs_path = self.base_dir / "EXPERIMENT_LOGS.md"
        self.trials_file = self.base_dir / "trials.jsonl"
        self.settings_file = self.base_dir / "settings.json"
        self.session_file = self.base_dir / "session_state.json"

    def ensure_layout(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)

        if not self.logs_path.exists():
            self.logs_path.write_text("# EXPERIMENT_LOGS\n\n", encoding="utf-8")
        if not self.trials_file.exists():
            self.trials_file.write_text("", encoding="utf-8")
        if not self.settings_file.exists():
            self.save_settings(Settings())

    def load_settings(self) -> Settings:
        if not self.settings_file.exists():
            return Settings()
        raw = json.loads(self.settings_file.read_text(encoding="utf-8"))
        return Settings.from_dict(raw)

    def save_settings(self, settings: Settings) -> None:
        self.settings_file.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")

    def load_session(self) -> dict[str, Any]:
        if not self.session_file.exists():
            return {}
        try:
            return json.loads(self.session_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save_session(self, state: State, current_trial_id: str | None) -> None:
        payload = {
            "state": state.value,
            "current_trial_id": current_trial_id,
            "updated_at": datetime.now().isoformat(),
        }
        self.session_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def list_project_names(self) -> list[str]:
        if not self.projects_dir.exists():
            return []
        return sorted([d.name for d in self.projects_dir.iterdir() if d.is_dir()])

    def project_path(self, project_name: str) -> Path:
        return self.projects_dir / project_name

    def create_trial(self, selected_project: str | None) -> TrialRecord:
        date = datetime.now().strftime("%Y%m%d")
        trial_number = self._next_trial_number(date)
        trial_name = f"trial_{trial_number:03d}"
        trial_id = f"{date}-{trial_name}"

        sandbox_root = self.sandbox_dir / date / trial_name
        codes_root = sandbox_root / "codes"
        outputs_root = sandbox_root / "outputs"
        eval_codes_root = sandbox_root / "eval_codes"

        results_root = self.results_dir / date / trial_name

        codes_root.mkdir(parents=True, exist_ok=True)
        outputs_root.mkdir(parents=True, exist_ok=True)
        eval_codes_root.mkdir(parents=True, exist_ok=True)
        results_root.mkdir(parents=True, exist_ok=True)

        run_sh = sandbox_root / "run.sh"
        eval_sh = sandbox_root / "eval.sh"

        if not run_sh.exists():
            run_sh.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "echo \"Define experiment commands in run.sh\"\n",
                encoding="utf-8",
            )
            run_sh.chmod(0o755)

        if not eval_sh.exists():
            eval_sh.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "echo \"Define evaluation commands in eval.sh\"\n",
                encoding="utf-8",
            )
            eval_sh.chmod(0o755)

        if selected_project:
            project_root = self.project_path(selected_project)
            if not project_root.exists():
                raise FileNotFoundError(f"project does not exist: {selected_project}")
            self._copy_tree(project_root, codes_root)

        trial = TrialRecord(
            trial_id=trial_id,
            date=date,
            trial_number=trial_number,
            state=State.PLAN,
            status=TrialStatus.ACTIVE,
            selected_project=selected_project,
            sandbox_path=str(sandbox_root.relative_to(self.base_dir)),
            outputs_path=str(outputs_root.relative_to(self.base_dir)),
            results_path=str(results_root.relative_to(self.base_dir)),
        )
        self.append_trial_record(trial)
        return trial

    def replace_trial_codes_from_project(self, trial: TrialRecord, selected_project: str | None) -> None:
        sandbox_root = self.base_dir / trial.sandbox_path
        codes_root = sandbox_root / "codes"
        if codes_root.exists():
            shutil.rmtree(codes_root)
        codes_root.mkdir(parents=True, exist_ok=True)

        if selected_project:
            project_root = self.project_path(selected_project)
            if not project_root.exists():
                raise FileNotFoundError(f"project not found: {selected_project}")
            self._copy_tree(project_root, codes_root)

        trial.selected_project = selected_project
        trial.touch()
        self.append_trial_record(trial)

    def append_trial_record(self, trial: TrialRecord) -> None:
        trial.touch()
        with open(self.trials_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trial.to_dict()) + "\n")

    def load_trials(self) -> list[TrialRecord]:
        if not self.trials_file.exists():
            return []

        latest: dict[str, TrialRecord] = {}
        with open(self.trials_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    trial = TrialRecord.from_dict(item)
                    latest[trial.trial_id] = trial
                except Exception:
                    continue

        return sorted(
            latest.values(),
            key=lambda t: (t.date, t.trial_number, t.updated_at),
        )

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        for trial in reversed(self.load_trials()):
            if trial.trial_id == trial_id:
                return trial
        return None

    def get_latest_active_trial(self) -> TrialRecord | None:
        trials = self.load_trials()
        for trial in reversed(trials):
            if trial.status == TrialStatus.ACTIVE:
                return trial
        return None

    def list_recent_trials(self, limit: int) -> list[TrialRecord]:
        trials = self.load_trials()
        trials.sort(key=lambda t: (t.date, t.trial_number), reverse=True)
        return trials[:limit]

    def list_dates(self) -> list[str]:
        dates = {t.date for t in self.load_trials()}
        return sorted(dates, reverse=True)

    def list_trials_for_date(self, date: str) -> list[TrialRecord]:
        trials = [t for t in self.load_trials() if t.date == date]
        trials.sort(key=lambda t: t.trial_number)
        return trials

    def append_experiment_log(self, trial: TrialRecord, summary: str, report_rel_path: str) -> None:
        entry = (
            f"{trial.date} - {trial.trial_name}: {summary}. "
            f"Full Doc: {report_rel_path}\n"
        )
        with open(self.logs_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def load_experiment_log_lines(self) -> list[str]:
        if not self.logs_path.exists():
            return []
        lines = [l.strip() for l in self.logs_path.read_text(encoding="utf-8").splitlines()]
        return [l for l in lines if l and not l.startswith("#")]

    def _next_trial_number(self, date: str) -> int:
        date_dir = self.sandbox_dir / date
        if not date_dir.exists():
            return 1

        max_num = 0
        for child in date_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("trial_"):
                continue
            suffix = child.name.split("trial_")[-1]
            if suffix.isdigit():
                max_num = max(max_num, int(suffix))
        return max_num + 1

    @staticmethod
    def _copy_tree(src: Path, dst: Path) -> None:
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
