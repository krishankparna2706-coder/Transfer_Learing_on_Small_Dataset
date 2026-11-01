"""P1: data commands report blocked prerequisites with exit code 4 and write nothing."""

from __future__ import annotations

from fewshotlab.cli import main
from helpers import REPO_ROOT, snapshot_files


def test_prepare_without_permission_is_blocked(isolated_paths, capsys):
    before = snapshot_files(isolated_paths.root)
    assert main(["data", "prepare", "--dataset", "flowers102"]) == 4
    assert "--allow-download" in capsys.readouterr().err
    assert main(["data", "prepare", "--dataset", "neu_cls"]) == 4
    assert snapshot_files(isolated_paths.root) == before


def test_flowers_only_options_are_rejected_for_neu(isolated_paths, capsys):
    assert main(["data", "prepare", "--dataset", "neu_cls", "--mirror", "internet-archive"]) == 1
    assert "flowers102 only" in capsys.readouterr().err


def test_audit_requires_a_prepared_dataset(isolated_paths, capsys):
    assert main(["data", "audit", "--dataset", "neu_cls"]) == 4
    assert "NOT_PREPARED" in capsys.readouterr().err


def test_support_build_reports_unprepared_datasets(isolated_paths, capsys):
    config = str(REPO_ROOT / "configs" / "protocol-v1.yaml")
    assert main(["support", "build", "--config", config]) == 4
    out = capsys.readouterr().out
    assert "flowers102: NOT_PREPARED" in out
    assert "neu_cls: NOT_PREPARED" in out
    assert (isolated_paths.reports / "label_budget.md").is_file()
