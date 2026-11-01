"""Shared fixtures. Every test runs offline on CPU against temporary directories."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

ENV_KEYS = {
    "data_root": "FEWSHOTLAB_DATA_ROOT",
    "manifests": "FEWSHOTLAB_MANIFESTS_DIR",
    "runs": "FEWSHOTLAB_RUNS_DIR",
    "reports": "FEWSHOTLAB_REPORTS_DIR",
    "cache": "FEWSHOTLAB_CACHE_DIR",
    "demo_assets": "FEWSHOTLAB_DEMO_ASSETS_DIR",
}


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Point every FewShotLab artifact location at a fresh temporary tree.

    Mirrors the repository: runs/, reports/, data/manifests/ and demo_assets/ exist, while
    data/raw/ and .cache/ do not (they are created on first use).
    """
    root = tmp_path / "project"
    locations = {
        "data_root": root / "data" / "raw",
        "manifests": root / "data" / "manifests",
        "runs": root / "runs",
        "reports": root / "reports",
        "cache": root / ".cache",
        "demo_assets": root / "demo_assets",
    }
    for key in ("manifests", "runs", "reports", "demo_assets"):
        locations[key].mkdir(parents=True)
    for key, env_name in ENV_KEYS.items():
        monkeypatch.setenv(env_name, str(locations[key]))
    env = {ENV_KEYS[key]: str(path) for key, path in locations.items()}
    return SimpleNamespace(root=root, env=env, **locations)
