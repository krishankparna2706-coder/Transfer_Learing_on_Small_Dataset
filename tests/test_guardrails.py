"""Repository guardrails that must hold at every phase."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from helpers import REPO_ROOT


def test_test_split_outputs_exist_only_under_an_intact_seal(monkeypatch):
    """Before the seal no test output may exist; after it, each must be recorded by the seal."""
    from fewshotlab.engine.seal import audit_test_outputs
    from fewshotlab.paths import get_paths

    for name in (
        "FEWSHOTLAB_DATA_ROOT",
        "FEWSHOTLAB_MANIFESTS_DIR",
        "FEWSHOTLAB_RUNS_DIR",
        "FEWSHOTLAB_REPORTS_DIR",
        "FEWSHOTLAB_CACHE_DIR",
        "FEWSHOTLAB_DEMO_ASSETS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FEWSHOTLAB_PROJECT_ROOT", str(REPO_ROOT))
    assert audit_test_outputs(get_paths()) == []


def test_training_code_cannot_reach_test_images_or_corruptions():
    """Static check: the training path never imports the test gate or the simulation package."""
    import ast

    source_root = REPO_ROOT / "src" / "fewshotlab"
    for module in ("engine/trainer.py", "engine/runner.py", "engine/study.py", "data/datasets.py"):
        tree = ast.parse((source_root / module).read_text(encoding="utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert "fewshotlab.engine.test_eval" not in imported, module
        assert not [m for m in imported if m and m.startswith("fewshotlab.simulation")], module


def test_large_and_private_inputs_are_ignored():
    rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/raw/", ".cache/", "*.ckpt", "*.pth", "*.npz", "/FewShotLab_0*.pdf"):
        assert pattern in rules, pattern


def test_no_pdfs_or_raw_data_are_tracked():
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert not [p for p in tracked if p.lower().endswith(".pdf")]
    assert not [p for p in tracked if p.startswith("data/raw/")]
