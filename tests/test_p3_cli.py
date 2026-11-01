"""P3: train and evaluate through the CLI on a software fixture (scratch needs no weights)."""

from __future__ import annotations

import yaml

from data_fixtures import fixture_config, training_fixture
from fewshotlab.cli import main
from fewshotlab.config import resolved_dict
from helpers import snapshot_files


def test_train_and_evaluate_from_the_command_line(isolated_paths, tmp_path, capsys):
    paths = training_fixture(isolated_paths.data_root)
    config = tmp_path / "fixture-scratch.yaml"
    config.write_text(yaml.safe_dump(resolved_dict(fixture_config(paths, "scratch", max_epochs=2))))
    assert main(["train", "--config", str(config), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "VALIDATION-ONLY" in out and "software fixture" in out
    run_id = out.split(" / ")[0].strip()
    assert main(["evaluate", "--run", run_id, "--split", "val", "--device", "cpu"]) == 0
    before = snapshot_files(isolated_paths.root)
    assert main(["evaluate", "--run", run_id, "--split", "test"]) == 5
    assert "locked" in capsys.readouterr().err
    assert snapshot_files(isolated_paths.root) == before
