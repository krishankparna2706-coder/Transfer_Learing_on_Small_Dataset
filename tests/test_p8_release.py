"""P8: the release matches its inventory, its documented commands and its sealed numbers."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import pytest

from fewshotlab.cli import main
from fewshotlab.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
COMMANDS = (
    "data prepare",
    "data audit",
    "support build",
    "weights fetch",
    "doctor",
    "train",
    "evaluate",
    "benchmark",
    "lock-test",
    "report",
    "predict",
    "demo",
)


def test_inventory_hashes_match_the_tracked_files():
    rows = [
        line.split("|")[1:-1]
        for line in (REPO / "docs/RELEASE_INVENTORY.md").read_text().splitlines()
        if line.startswith("| ") and "`" in line
    ]
    checked = 0
    for name, path, digest, _check, in_git in (map(str.strip, r) for r in rows):
        path, digest = path.strip("`"), digest.strip("`")
        if in_git == "yes" and "*" not in path and "<" not in path:
            assert sha256_file(REPO / path) == digest, f"{name}: {path} changed after the inventory"
            checked += 1
    assert checked >= 15


def _help(command: tuple[str, ...], capsys) -> str:
    with pytest.raises(SystemExit):
        main([*command, "--help"])
    return capsys.readouterr().out


def test_every_documented_command_and_flag_exists(capsys):
    text = (REPO / "README.md").read_text(encoding="utf-8")
    found = 0
    for snippet in re.findall(r"`([a-z][^`]*)`", text):
        snippet = snippet.replace("python -m fewshotlab ", "")
        command = next((c for c in COMMANDS if snippet == c or snippet.startswith(c + " ")), None)
        if command is None:
            continue
        help_text = _help(tuple(command.split()), capsys)
        for flag in re.findall(r"(--[a-z][a-z-]*)", snippet):
            assert flag in help_text, f"README uses `{snippet}` but {command} has no {flag}"
        found += 1
    assert found >= 15


def _sealed_means() -> dict[tuple[str, int, str], float]:
    with (REPO / "reports/fewshotlab-v1-full/test/aggregate_test.csv").open() as handle:
        return {
            (r["dataset"], int(r["k"]), r["method"]): float(r["mean"])
            for r in csv.DictReader(handle)
            if r["metric"] == "macro_f1"
        }


def test_numbers_quoted_in_the_docs_equal_the_sealed_report():
    means, methods = _sealed_means(), ("scratch", "frozen", "partial", "prototypes")
    names = {"Flowers": "flowers102", "NEU": "neu_cls"}
    script = (REPO / "docs/JUDGE_SCRIPT.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (Flowers|NEU), K=(\d+) \|(.+)\|$", script, flags=re.M)
    assert len(rows) == 4
    for name, k, cells in rows:
        for method, cell in zip(methods, cells.split("|"), strict=True):
            quoted = float(cell.split()[0])
            assert round(means[(names[name], int(k), method)], 3) == quoted, (name, k, method)
    card = (REPO / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    for method in methods:
        row = re.search(rf"^\| {method} \|(.+)\|$", card, flags=re.M).group(1).split("|")
        quoted = [float(c.split()[0]) for c in row[-4:]]
        keys = [("flowers102", 1), ("flowers102", 10), ("neu_cls", 1), ("neu_cls", 10)]
        assert quoted == [round(means[(d, k, method)], 3) for d, k in keys], method


def test_git_ships_no_dataset_images_checkpoints_or_weights():
    try:
        files = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    banned = (
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".ckpt",
        ".pth",
        ".npz",
        ".pdf",
        ".rar",
        ".tgz",
        ".mat",
    )
    assert not [f for f in files if f.lower().endswith(banned)]
    pngs = [f for f in files if f.lower().endswith(".png")]
    assert all(f.startswith("reports/") for f in pngs), pngs  # charts only, never dataset images
    assert not [f for f in files if f.endswith("rehearsal_events.csv")]
