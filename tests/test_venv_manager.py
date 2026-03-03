"""Tests for per-trial venv management (US-017)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from researchclaw.sandbox.venv_manager import VenvManager


# ---------------------------------------------------------------------------
# ensure_venv
# ---------------------------------------------------------------------------


class TestEnsureVenv:
    """Tests for VenvManager.ensure_venv()."""

    def test_creates_venv_when_not_exists(self, tmp_path: Path) -> None:
        """Calls python -m venv when env/ doesn't exist."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            result = VenvManager.ensure_venv(trial_dir, "python3")

        assert result == trial_dir / "env"
        mock_run.assert_called_once_with(
            ["python3", "-m", "venv", str(trial_dir / "env")],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_skips_creation_when_venv_exists(self, tmp_path: Path) -> None:
        """Returns existing venv path without calling subprocess."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "env").mkdir()

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            result = VenvManager.ensure_venv(trial_dir, "python3")

        assert result == trial_dir / "env"
        mock_run.assert_not_called()

    def test_uses_custom_python_command(self, tmp_path: Path) -> None:
        """Uses the specified python_command for venv creation."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            VenvManager.ensure_venv(trial_dir, "/usr/bin/python3.11")

        args = mock_run.call_args[0][0]
        assert args[0] == "/usr/bin/python3.11"

    def test_default_python_command(self, tmp_path: Path) -> None:
        """Defaults to 'python3' when no python_command specified."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            VenvManager.ensure_venv(trial_dir)

        args = mock_run.call_args[0][0]
        assert args[0] == "python3"

    def test_raises_on_venv_creation_failure(self, tmp_path: Path) -> None:
        """Raises CalledProcessError when venv creation fails."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()

        with patch(
            "researchclaw.sandbox.venv_manager.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "python3"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                VenvManager.ensure_venv(trial_dir, "python3")

    def test_returns_path_object(self, tmp_path: Path) -> None:
        """Return value is a Path object."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "env").mkdir()

        result = VenvManager.ensure_venv(trial_dir)
        assert isinstance(result, Path)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Accepts string path for trial_dir."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "env").mkdir()

        result = VenvManager.ensure_venv(str(trial_dir))
        assert result == trial_dir / "env"


# ---------------------------------------------------------------------------
# install_requirements
# ---------------------------------------------------------------------------


class TestInstallRequirements:
    """Tests for VenvManager.install_requirements()."""

    def test_installs_requirements(self, tmp_path: Path) -> None:
        """Runs pip install -r requirements.txt in the trial venv."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "requirements.txt").write_text("numpy\npandas\n")
        pip_path = trial_dir / "env" / "bin" / "pip"
        pip_path.parent.mkdir(parents=True)
        pip_path.touch()

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            result = VenvManager.install_requirements(trial_dir)

        assert result is True
        mock_run.assert_called_once_with(
            [str(pip_path), "install", "-r", str(trial_dir / "requirements.txt"), "-q"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_skips_empty_requirements(self, tmp_path: Path) -> None:
        """Skips installation when requirements.txt is empty."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "requirements.txt").write_text("")

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            result = VenvManager.install_requirements(trial_dir)

        assert result is True
        mock_run.assert_not_called()

    def test_skips_whitespace_only_requirements(self, tmp_path: Path) -> None:
        """Skips installation when requirements.txt has only whitespace."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "requirements.txt").write_text("   \n\n  ")

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            result = VenvManager.install_requirements(trial_dir)

        assert result is True
        mock_run.assert_not_called()

    def test_skips_missing_requirements(self, tmp_path: Path) -> None:
        """Skips installation when requirements.txt doesn't exist."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            result = VenvManager.install_requirements(trial_dir)

        assert result is True
        mock_run.assert_not_called()

    def test_returns_false_when_venv_missing(self, tmp_path: Path) -> None:
        """Returns False when venv pip doesn't exist."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "requirements.txt").write_text("numpy\n")
        # No env/ directory created

        result = VenvManager.install_requirements(trial_dir)
        assert result is False

    def test_returns_false_on_pip_failure(self, tmp_path: Path) -> None:
        """Returns False when pip install fails."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "requirements.txt").write_text("nonexistent-package-xyz\n")
        pip_path = trial_dir / "env" / "bin" / "pip"
        pip_path.parent.mkdir(parents=True)
        pip_path.touch()

        with patch(
            "researchclaw.sandbox.venv_manager.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "pip"),
        ):
            result = VenvManager.install_requirements(trial_dir)

        assert result is False

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Accepts string path for trial_dir."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "requirements.txt").write_text("")

        result = VenvManager.install_requirements(str(trial_dir))
        assert result is True


# ---------------------------------------------------------------------------
# get_venv_python
# ---------------------------------------------------------------------------


class TestGetVenvPython:
    """Tests for VenvManager.get_venv_python()."""

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        """Returns path to env/bin/python."""
        trial_dir = tmp_path / "trial_001"
        result = VenvManager.get_venv_python(trial_dir)
        assert result == trial_dir / "env" / "bin" / "python"

    def test_returns_path_object(self, tmp_path: Path) -> None:
        """Return value is a Path object."""
        result = VenvManager.get_venv_python(tmp_path)
        assert isinstance(result, Path)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Accepts string path for trial_dir."""
        result = VenvManager.get_venv_python(str(tmp_path))
        assert result == tmp_path / "env" / "bin" / "python"


# ---------------------------------------------------------------------------
# venv_exists
# ---------------------------------------------------------------------------


class TestVenvExists:
    """Tests for VenvManager.venv_exists()."""

    def test_returns_true_when_exists(self, tmp_path: Path) -> None:
        """Returns True when env/ directory exists."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "env").mkdir()

        assert VenvManager.venv_exists(trial_dir) is True

    def test_returns_false_when_missing(self, tmp_path: Path) -> None:
        """Returns False when env/ directory doesn't exist."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()

        assert VenvManager.venv_exists(trial_dir) is False

    def test_returns_false_for_file_not_dir(self, tmp_path: Path) -> None:
        """Returns False when env is a file, not a directory."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "env").touch()  # file, not dir

        assert VenvManager.venv_exists(trial_dir) is False

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Accepts string path for trial_dir."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()

        assert VenvManager.venv_exists(str(trial_dir)) is False


# ---------------------------------------------------------------------------
# Template integration
# ---------------------------------------------------------------------------


class TestTemplateIntegration:
    """Verify VenvManager paths match what templates expect."""

    def test_venv_path_matches_run_exp_template(self, tmp_path: Path) -> None:
        """VenvManager.ensure_venv returns path matching template's VENV_DIR."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        (trial_dir / "env").mkdir()

        venv_dir = VenvManager.ensure_venv(trial_dir)
        # Template: VENV_DIR="${TRIAL_DIR}/env"
        assert venv_dir == trial_dir / "env"

    def test_requirements_path_matches_template(self, tmp_path: Path) -> None:
        """VenvManager reads requirements.txt from same path as template."""
        trial_dir = tmp_path / "trial_001"
        trial_dir.mkdir()
        req_path = trial_dir / "requirements.txt"
        req_path.write_text("numpy\n")
        pip_path = trial_dir / "env" / "bin" / "pip"
        pip_path.parent.mkdir(parents=True)
        pip_path.touch()

        with patch("researchclaw.sandbox.venv_manager.subprocess.run") as mock_run:
            VenvManager.install_requirements(trial_dir)

        # Template: REQUIREMENTS="${TRIAL_DIR}/requirements.txt"
        call_args = mock_run.call_args[0][0]
        assert str(req_path) in call_args

    def test_venv_python_matches_template(self, tmp_path: Path) -> None:
        """get_venv_python path matches template's Python path."""
        trial_dir = tmp_path / "trial_001"
        python_path = VenvManager.get_venv_python(trial_dir)
        # Template: "${VENV_DIR}/bin/python"
        assert python_path == trial_dir / "env" / "bin" / "python"

    def test_never_activates_venv(self) -> None:
        """VenvManager never sources activate or modifies PATH.

        This is a design verification: ensure_venv and install_requirements
        use subprocess, not os.environ or importlib.
        """
        import inspect
        # Check method bodies only, not class docstring
        for method_name in ("ensure_venv", "install_requirements", "get_venv_python", "venv_exists"):
            source = inspect.getsource(getattr(VenvManager, method_name))
            assert "activate" not in source, f"{method_name} references 'activate'"
            assert "os.environ" not in source, f"{method_name} modifies os.environ"
            assert "sys.path" not in source, f"{method_name} modifies sys.path"
