"""Test helpers shared across test modules (importable because tests/ is on sys.path)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def snapshot_files(root: Path) -> set[str]:
    """Relative paths of every file under ``root`` (empty if it does not exist)."""
    if not root.exists():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def run_module(*args: str, env_extra: dict[str, str] | None = None, timeout: int = 600):
    """Run ``python -m fewshotlab`` in a subprocess from the repository root."""
    env = os.environ.copy()
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "fewshotlab", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=timeout,
    )
