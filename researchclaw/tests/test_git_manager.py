from __future__ import annotations

import subprocess
from pathlib import Path

from researchclaw.git_manager import GitManager
from researchclaw.models import ProjectGitConfig, TrialRecord
from researchclaw.policy import AuthorityManager
from researchclaw.states import State, TrialStatus


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_assimilate_and_push(tmp_path: Path) -> None:
    base = tmp_path
    projects = base / "projects"
    projects.mkdir(parents=True)

    remote = base / "remote.git"
    _run(["git", "init", "--bare", str(remote)])

    proj = projects / "demo"
    proj.mkdir()
    _run(["git", "init"], cwd=proj)
    _run(["git", "checkout", "-B", "main"], cwd=proj)
    _run(["git", "config", "user.email", "test@example.com"], cwd=proj)
    _run(["git", "config", "user.name", "Tester"], cwd=proj)
    (proj / "a.txt").write_text("old\n", encoding="utf-8")
    _run(["git", "add", "-A"], cwd=proj)
    _run(["git", "commit", "-m", "init"], cwd=proj)
    _run(["git", "remote", "add", "origin", str(remote)], cwd=proj)
    _run(["git", "push", "-u", "origin", "main"], cwd=proj)

    codes = base / "sandbox" / "20260227" / "trial_001" / "codes"
    outputs = base / "sandbox" / "20260227" / "trial_001" / "outputs"
    results = base / "results" / "20260227" / "trial_001"
    codes.mkdir(parents=True)
    outputs.mkdir(parents=True)
    results.mkdir(parents=True)

    (codes / "a.txt").write_text("new\n", encoding="utf-8")
    (codes / "b.txt").write_text("added\n", encoding="utf-8")

    trial = TrialRecord(
        trial_id="20260227-trial_001",
        date="20260227",
        trial_number=1,
        state=State.UPDATE_AND_PUSH,
        status=TrialStatus.COMPLETED,
        selected_project="demo",
        sandbox_path="sandbox/20260227/trial_001",
        outputs_path="sandbox/20260227/trial_001/outputs",
        results_path="results/20260227/trial_001",
    )

    auth = AuthorityManager(str(base))
    gm = GitManager(str(base), auth)
    copied = gm.assimilate_trial_codes(State.UPDATE_AND_PUSH, trial, "demo")
    assert set(copied) == {"a.txt", "b.txt"}

    out = gm.commit_and_push(
        State.UPDATE_AND_PUSH,
        "demo",
        ProjectGitConfig(remote_url=str(remote), default_branch="main", auth_source="system"),
        "assimilation",
    )
    assert out is not None

    check = subprocess.run(["git", "--git-dir", str(remote), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert check.returncode == 0
