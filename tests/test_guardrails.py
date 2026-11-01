"""Repository guardrails that must hold at every phase."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from helpers import REPO_ROOT

TEST_SPLIT_OUTPUTS = ("test_metrics.json", "predictions_test.csv")


def test_no_test_split_outputs_exist_before_the_gate():
    hits = [
        path
        for folder in ("runs", "reports", "demo_assets")
        for path in (REPO_ROOT / folder).rglob("*")
        if path.name in TEST_SPLIT_OUTPUTS
    ]
    assert hits == [], f"test-split outputs found before the sealed P5 gate: {hits}"


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
