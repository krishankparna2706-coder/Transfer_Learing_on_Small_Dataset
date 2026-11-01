"""P3: end-to-end SOFTWARE-FIXTURE runs of the training engine.

The fixture is three noisy colour classes. Every artifact it produces carries is_fixture=true,
and report aggregation rejects it. Frozen, partial and prototype fixtures use a random encoder
(``init_override``), which only software fixtures may use.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect

import numpy as np
import pytest

from data_fixtures import fixture_config, training_fixture
from fewshotlab.engine import artifacts
from fewshotlab.engine.ledger import latest_status
from fewshotlab.engine.runner import evaluate_run, run_training
from fewshotlab.engine.trainer import (
    NonFiniteError,
    SupportSet,
    ValidationSet,
    fit_learned,
    fit_prototype_method,
)
from fewshotlab.errors import ProtocolRefusal
from fewshotlab.io_utils import read_json
from fewshotlab.reporting.aggregate import collect_runs, verify_evidence
from helpers import snapshot_files

METHODS = ["scratch", "frozen", "partial", "prototypes"]


@pytest.fixture
def paths(isolated_paths):
    return training_fixture(isolated_paths.data_root)


def _history(attempt_dir):
    with open(attempt_dir / "train_history.csv", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("method", METHODS)
def test_fixture_run_completes_with_full_evidence(paths, method):
    result = run_training(fixture_config(paths, method), paths=paths, init_override="random")
    adir = result.attempt_dir
    expected = {
        "config.resolved.yaml",
        "environment.json",
        "provenance.json",
        "model_spec.json",
        "predictions_val.csv",
        "val_metrics.json",
        "status.json",
    }
    expected |= {"prototypes.npz"} if method == "prototypes" else {"best.ckpt", "train_history.csv"}
    present = {p.name for p in adir.iterdir()}
    assert expected <= present, expected - present
    assert not {"test_metrics.json", "predictions_test.csv"} & present
    assert "last.ckpt" not in present  # resume state is removed once a run completes
    doc = read_json(adir / "val_metrics.json")
    assert (doc["status"], doc["split"], doc["is_fixture"]) == ("COMPLETED", "val", True)
    assert doc["metrics"]["macro_f1"] >= 0.5  # learnable colours: well above chance (1/3)
    assert verify_evidence(adir) is None  # metrics regenerate from the stored predictions
    provenance = read_json(adir / "provenance.json")
    assert len(provenance["code"]["source_sha256"]) == 64
    assert provenance["data"]["support"]["content_sha256"]
    assert provenance["seeds"]["streams"]["augmentation"][0] == "augment"
    if method != "prototypes":
        assert len(_history(adir)) == doc["epochs_run"]
        assert doc["reload_check"]["identical_predictions"]
    assert (adir / "plots" / "confusion_val.png").is_file()


def test_selection_rule_and_patience(paths):
    result = run_training(
        fixture_config(paths, "frozen", max_epochs=40, patience=2),
        paths=paths,
        init_override="random",
    )
    rows = _history(result.attempt_dir)
    assert result.stop_reason == "patience"
    assert int(rows[-1]["epochs_without_improvement"]) == 2
    best = max(
        rows, key=lambda r: (float(r["val_macro_f1"]), -float(r["val_loss"]), -int(r["epoch"]))
    )
    assert result.selected_epoch == int(best["epoch"])


def test_scratch_training_loss_decreases(paths):
    result = run_training(fixture_config(paths, "scratch", max_epochs=10), paths=paths)
    rows = _history(result.attempt_dir)
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])


def test_fixture_results_are_rejected_by_aggregation(paths):
    run_training(fixture_config(paths, "prototypes"), paths=paths, init_override="random")
    report = collect_runs(paths, categories=("core", "fixture"))
    assert report["accepted"] == []
    assert report["rejected"] and all("fixture" in r["reason"] for r in report["rejected"])


def test_interruption_fails_the_attempt_and_resume_continues_exactly(paths, tmp_path):
    cfg = fixture_config(paths, "partial", max_epochs=6)

    def interrupt(epoch):
        if epoch == 3:
            raise KeyboardInterrupt("simulated interruption")

    with pytest.raises(KeyboardInterrupt):
        run_training(cfg, paths=paths, init_override="random", on_epoch_end=interrupt)
    first = artifacts.attempt_dirs(
        paths,
        artifacts.make_run_id(
            cfg, __import__("fewshotlab.config", fromlist=["config_hash"]).config_hash(cfg)
        ),
    )[0]
    assert read_json(first / "status.json")["status"] == "FAILED"
    assert not (first / "val_metrics.json").exists()
    rejected = collect_runs(paths, categories=("fixture",))["rejected"]
    assert any(r["status"] == "FAILED" for r in rejected)
    resumed = run_training(cfg, paths=paths, init_override="random", resume=True)
    assert read_json(resumed.attempt_dir / "provenance.json")["resumed_from"] == {
        "attempt_id": "a001",
        "epoch": 3,
    }
    reference = run_training(
        cfg, paths=dataclasses.replace(paths, runs=tmp_path / "ref-runs"), init_override="random"
    )
    keep = ("epoch", "train_loss", "val_loss", "val_macro_f1", "is_best", "optimizer_steps")
    assert [{k: r[k] for k in keep} for r in _history(resumed.attempt_dir)] == [
        {k: r[k] for k in keep} for r in _history(reference.attempt_dir)
    ]
    a = artifacts.read_predictions(resumed.attempt_dir / "predictions_val.csv")
    b = artifacts.read_predictions(reference.attempt_dir / "predictions_val.csv")
    assert np.array_equal(a["pred"], b["pred"]) and np.array_equal(a["scores"], b["scores"])


def test_two_repetitions_on_the_same_device_match(paths, tmp_path):
    cfg = fixture_config(paths, "scratch", max_epochs=3)
    first = run_training(cfg, paths=paths)
    second = run_training(cfg, paths=dataclasses.replace(paths, runs=tmp_path / "repeat"))
    a = artifacts.read_predictions(first.attempt_dir / "predictions_val.csv")
    b = artifacts.read_predictions(second.attempt_dir / "predictions_val.csv")
    assert np.array_equal(a["pred"], b["pred"])
    assert np.allclose(a["scores"], b["scores"], atol=1e-6, rtol=0)  # declared CPU tolerance
    assert first.metrics == second.metrics


def test_completed_runs_are_immutable(paths):
    cfg = fixture_config(paths, "prototypes")
    run_training(cfg, paths=paths, init_override="random")
    with pytest.raises(ProtocolRefusal, match="immutable"):
        run_training(cfg, paths=paths, init_override="random")


def test_non_finite_loss_marks_the_run_failed(paths, monkeypatch):
    import fewshotlab.engine.trainer as trainer

    real = trainer.F.cross_entropy
    calls = {"n": 0}

    def poisoned(*args, **kwargs):
        calls["n"] += 1
        out = real(*args, **kwargs)
        return out * float("nan") if calls["n"] == 3 else out  # epoch 2's training step

    monkeypatch.setattr(trainer.F, "cross_entropy", poisoned)
    with pytest.raises(NonFiniteError):
        run_training(fixture_config(paths, "scratch", max_epochs=4), paths=paths)
    (event,) = latest_status(paths).values()
    assert event["status"] == "FAILED" and "NonFiniteError" in event["reason"]
    adir = paths.runs / event["run_id"] / event["attempt_id"]
    assert read_json(adir / "status.json")["failure"]["type"] == "NonFiniteError"
    assert not (adir / "val_metrics.json").exists()


def test_training_never_reads_test_rows(paths, monkeypatch):
    import fewshotlab.engine.runner as runner_module

    requested = []
    real = runner_module.read_manifest

    def spy(path, *, splits):
        requested.append(tuple(splits))
        return real(path, splits=splits)

    monkeypatch.setattr(runner_module, "read_manifest", spy)
    run_training(fixture_config(paths, "frozen", max_epochs=2), paths=paths, init_override="random")
    assert requested and all("test" not in splits for splits in requested)


def test_fit_interfaces_accept_only_support_and_validation():
    for function in (fit_learned, fit_prototype_method):
        names = list(inspect.signature(function).parameters)
        assert not [n for n in names if "test" in n.lower()]
    from fewshotlab.data.manifest import ImageRecord

    held_out = ImageRecord("x", "x.png", 0, "c", "test", "0" * 64)
    with pytest.raises(ProtocolRefusal):
        SupportSet((held_out,))
    with pytest.raises(ProtocolRefusal):
        ValidationSet((held_out,))


def test_validation_reevaluation_and_test_refusal(paths):
    result = run_training(fixture_config(paths, "scratch", max_epochs=2), paths=paths)
    verification = evaluate_run(result.run_id, "val", paths=paths)
    assert verification["identical_predictions"]
    assert verification["recomputed_macro_f1"] == verification["stored_macro_f1"]
    before = snapshot_files(paths.runs)
    with pytest.raises(ProtocolRefusal, match="locked"):
        evaluate_run(result.run_id, "test", paths=paths)
    assert snapshot_files(paths.runs) == before


def test_init_override_is_reserved_for_fixtures(paths):
    from fewshotlab.errors import ConfigError

    cfg = dataclasses.replace(
        fixture_config(paths, "frozen"), category="diagnostic", is_fixture=False
    )
    with pytest.raises(ConfigError, match="software fixtures"):
        run_training(cfg, paths=paths, init_override="random")
