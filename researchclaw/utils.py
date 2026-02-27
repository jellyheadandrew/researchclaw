"""
utils.py — Shared helper functions used across multiple ResearchClaw modules.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path


def setup_logging(base_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with both a console handler and a rotating file handler.
    Log file is written to {base_dir}/core/researchclaw.log.
    Returns the root 'researchclaw' logger.
    """
    logger = logging.getLogger("researchclaw")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_path = Path(base_dir) / "core" / "researchclaw.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def run_subprocess(
    cmd: list[str],
    cwd: str,
    env: dict | None = None,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    """
    Run a command synchronously. Returns (returncode, stdout, stderr).
    Raises subprocess.TimeoutExpired if timeout is exceeded.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def tail_file(path: str, n: int = 50) -> str:
    """Return the last n lines of a file as a single string."""
    p = Path(path)
    if not p.exists():
        return ""
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string, e.g. '3h 14m' or '45s'."""
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    minutes = secs // 60
    if minutes < 60:
        return f"{minutes}m {secs % 60}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def parse_metrics_from_log(log_text: str) -> dict[str, float]:
    """
    Extract numeric key=value metrics from experiment log output.
    Handles common patterns like:
      - loss=0.0891
      - val_acc: 74.2%
      - epoch 41/50 | loss: 0.089 | acc: 0.742
    Returns a dict mapping metric name → last seen value.
    """
    metrics: dict[str, float] = {}

    # Pattern 1: key=value or key: value (float)
    pattern = re.compile(
        r'\b((?:val_|train_|test_)?(?:loss|acc|accuracy|f1|auc|perplexity|ppl|'
        r'lr|learning_rate|epoch|step|score|metric|mse|rmse|mae|r2|bleu|rouge))'
        r'[\s:=]+([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)',
        re.IGNORECASE,
    )
    for match in pattern.finditer(log_text):
        key = match.group(1).lower().replace(" ", "_")
        try:
            metrics[key] = float(match.group(2))
        except ValueError:
            pass

    # Pattern 2: percentage values like "74.2%"
    pct_pattern = re.compile(
        r'\b((?:val_|train_)?(?:acc|accuracy|f1))[:\s=]+([0-9]+(?:\.[0-9]+)?)%',
        re.IGNORECASE,
    )
    for match in pct_pattern.finditer(log_text):
        key = match.group(1).lower()
        try:
            metrics[key] = float(match.group(2)) / 100.0
        except ValueError:
            pass

    return metrics
