"""P0: the CLI surface exists, documents its flags, and unfinished commands fail clearly."""

from __future__ import annotations

import argparse

import pytest

from fewshotlab.cli import main
from fewshotlab.errors import NotImplementedPhaseError
from helpers import run_module, snapshot_files

DOCUMENTED_FLAGS = {
    ("doctor",): ["--json", "--require-device"],
    ("data", "prepare"): ["--dataset", "--source", "--allow-download", "--mirror", "--cross-check"],
    ("data", "audit"): ["--dataset", "--near-dup-threshold"],
    ("support", "build"): ["--config", "--dataset"],
    ("train",): ["--config", "--device", "--allow-download", "--resume"],
    ("evaluate",): ["--run", "--split", "--study", "--device"],
    ("benchmark",): ["--config", "--dry-run", "--device", "--allow-download"],
    ("lock-test",): ["--plan", "--dry-run", "--allow-partial", "--skip-reproduction", "--device"],
    ("report",): ["--ledger", "--out", "--study"],
    ("predict",): ["--run", "--attempt", "--image", "--top-k", "--device", "--json"],
    ("demo",): ["--port", "--host", "--study", "--check", "--rehearse", "--device"],
}


def test_root_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in (
        "doctor",
        "data",
        "support",
        "train",
        "evaluate",
        "benchmark",
        "lock-test",
        "report",
        "predict",
        "demo",
    ):
        assert command in out
    assert "exit codes" in out


@pytest.mark.parametrize("command", sorted(DOCUMENTED_FLAGS), ids=" ".join)
def test_subcommand_help_documents_its_flags(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*command, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in DOCUMENTED_FLAGS[command]:
        assert flag in out, f"{flag} missing from `{' '.join(command)} --help`"


def test_missing_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_unknown_dataset_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["data", "prepare", "--dataset", "cifar10"])
    assert exc.value.code == 2


def test_an_unfinished_phase_command_exits_3_and_writes_nothing(isolated_paths, monkeypatch):
    """Every command exists after P6; the stub mechanism stays for later phases."""
    import fewshotlab.cli as cli

    build = cli.build_parser

    def with_stub():
        parser = build()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        cli._command(
            sub, "later", "later", "A later feature.", cli._not_implemented("later", "P9", "x")
        )
        return parser

    monkeypatch.setattr(cli, "build_parser", with_stub)
    before = snapshot_files(isolated_paths.root)
    with pytest.raises(NotImplementedPhaseError) as exc:
        cli._not_implemented("later", "P9", "x")(argparse.Namespace())
    assert (
        exc.value.exit_code == 3 and "NOT IMPLEMENTED" in str(exc.value) and "P9" in str(exc.value)
    )
    assert main(["later"]) == 3
    assert snapshot_files(isolated_paths.root) == before


def test_python_dash_m_entry_point():
    proc = run_module("--help")
    assert proc.returncode == 0, proc.stderr
    assert "python -m fewshotlab" in proc.stdout


def test_version_flag():
    proc = run_module("--version")
    assert proc.returncode == 0, proc.stderr
    assert "fewshotlab 0.1.0" in proc.stdout
    assert "fewshotlab-v1" in proc.stdout
