"""Project locations.

Each location has a repository-relative default and an environment-variable override, so tests
and other machines never need code edits. Values are resolved on every call (no hidden state).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fewshotlab.errors import FewShotLabError

ENV_PROJECT_ROOT = "FEWSHOTLAB_PROJECT_ROOT"
ENV_DATA_ROOT = "FEWSHOTLAB_DATA_ROOT"
ENV_MANIFESTS_DIR = "FEWSHOTLAB_MANIFESTS_DIR"
ENV_RUNS_DIR = "FEWSHOTLAB_RUNS_DIR"
ENV_REPORTS_DIR = "FEWSHOTLAB_REPORTS_DIR"
ENV_CACHE_DIR = "FEWSHOTLAB_CACHE_DIR"
ENV_DEMO_ASSETS_DIR = "FEWSHOTLAB_DEMO_ASSETS_DIR"


def _is_project_root(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        return 'name = "fewshotlab"' in pyproject.read_text(encoding="utf-8")
    except OSError:
        return False


def find_project_root() -> Path:
    """Locate the repository root (env override, then the cwd, then this file's location)."""
    override = os.environ.get(ENV_PROJECT_ROOT)
    if override:
        return Path(override).expanduser().resolve()
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if _is_project_root(candidate):
                return candidate
    raise FewShotLabError(
        "could not locate the FewShotLab project root (a pyproject.toml with "
        f'name = "fewshotlab"); run from inside the repository or set {ENV_PROJECT_ROOT}'
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    manifests: Path
    runs: Path
    reports: Path
    cache: Path
    demo_assets: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def ledger(self) -> Path:
        return self.runs / "ledger.jsonl"

    @property
    def weights_dir(self) -> Path:
        return self.cache / "torch" / "checkpoints"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def get_paths() -> ProjectPaths:
    root = find_project_root()
    return ProjectPaths(
        root=root,
        data_root=_env_path(ENV_DATA_ROOT, root / "data" / "raw"),
        manifests=_env_path(ENV_MANIFESTS_DIR, root / "data" / "manifests"),
        runs=_env_path(ENV_RUNS_DIR, root / "runs"),
        reports=_env_path(ENV_REPORTS_DIR, root / "reports"),
        cache=_env_path(ENV_CACHE_DIR, root / ".cache"),
        demo_assets=_env_path(ENV_DEMO_ASSETS_DIR, root / "demo_assets"),
    )
