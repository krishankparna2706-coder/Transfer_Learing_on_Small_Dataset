"""P0: the CLI surface exists, documents its flags, and unfinished commands fail clearly."""

from __future__ import annotations

import pytest

from fewshotlab.cli import main
from helpers import run_module, snapshot_files

DOCUMENTED_FLAGS = {
    ("doctor",): ["--json", "--require-device"],
    ("data", "prepare"): ["--dataset", "--source", "--allow-download", "--mirror", "--cross-check"],
    ("data", "audit"): ["--dataset", "--near-dup-threshold"],
    ("support", "build"): ["--config", "--dataset"],
    ("train",): ["--config", "--device", "--allow-download", "--resume"],
    ("evaluate",): ["--run", "--split", "--device"],
    ("benchmark",): ["--config", "--dry-run", "--device", "--allow-download"],
    ("lock-test",): ["--plan"],
    ("report",): ["--ledger", "--out", "--study"],
    ("demo",): ["--port"],
}

# Commands that are still stubs in this phase, with the phase that implements them.
STUBS = {
    "lock-test": (["lock-test", "--plan", "configs/not-used.yaml"], "P5"),
    "demo": (["demo"], "P6"),
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


@pytest.mark.parametrize("name", sorted(STUBS))
def test_stub_commands_exit_3_and_write_nothing(name, isolated_paths, capsys):
    argv, phase = STUBS[name]
    before = snapshot_files(isolated_paths.root)
    assert main(argv) == 3
    err = capsys.readouterr().err
    assert "NOT IMPLEMENTED" in err
    assert phase in err
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
