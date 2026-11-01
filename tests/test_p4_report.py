"""P4: study plans, benchmark execution and validation reports on a SOFTWARE FIXTURE."""

from __future__ import annotations

import csv

import pytest

from data_fixtures import training_fixture
from fewshotlab.engine import artifacts
from fewshotlab.engine.study import load_study, plan_study, run_study, verify_plan
from fewshotlab.errors import ConfigError
from fewshotlab.io_utils import read_json
from fewshotlab.reporting.errors import select_errors
from fewshotlab.reporting.study_report import build_study_report

STUDY = {
    "study_id": "fixture-study",
    "category": "fixture",
    "datasets": ["fixture-color"],
    "shots": [4],
    "methods": ["scratch", "frozen", "partial", "prototypes"],
    "support_seeds": [11],
    "device": "cpu",
}


@pytest.fixture
def paths(isolated_paths):
    return training_fixture(isolated_paths.data_root)


def test_plan_checks_and_prototype_deduplication(isolated_paths):
    from data_fixtures import install_dataset, make_color_dataset
    from fewshotlab.data.audit import run_audit
    from fewshotlab.data.support import build_supports

    records, cmap = make_color_dataset(
        isolated_paths.data_root, per_split={"train": 4, "val": 2, "test": 1}
    )
    paths = install_dataset(records, cmap, dataset="fixture-color")
    run_audit("fixture-color", paths=paths)
    build_supports(paths, "fixture-color", shots=(2, 4), seeds=(11, 22))
    study = {**STUDY, "shots": [2, 4], "support_seeds": [11, 22]}
    slots = plan_study(study, paths)
    assert len(slots) == 16
    assert all(c["status"] == "PASS" for c in verify_plan(slots))
    duplicates = [s for s in slots if s.duplicate_of]
    # K=4 uses every training image for both seeds: the deterministic prototype slot repeats
    assert [(s.method, s.k, s.seed) for s in duplicates] == [("prototypes", 4, 22)]
    assert not any(s.duplicate_of for s in slots if s.method != "prototypes")


def test_study_runs_reports_and_regenerates(paths):
    outcomes = run_study(STUDY, paths=paths, init_override="random")
    assert [o["status"] for o in outcomes] == ["COMPLETED"] * 4
    assert [o["outcome"] for o in run_study(STUDY, paths=paths, init_override="random")] == [
        "skipped"
    ] * 4  # finished slots are never re-run
    report = build_study_report(STUDY, paths, paths.reports)
    out = report["out"]
    for name in (
        "comparison.md",
        "comparison.csv",
        "comparison.json",
        "per_class.csv",
        "plots/val_metrics_by_k.png",
        "sanity/sanity.json",
        "errors/errors.json",
    ):
        assert (out / name).is_file(), name
    text = (out / "comparison.md").read_text()
    assert "VALIDATION ONLY" in text and "SOFTWARE FIXTURE" in text
    for row in report["rows_full"]:
        doc = read_json(row["_attempt_dir"] / "val_metrics.json")
        assert row["macro_f1"] == doc["metrics"]["macro_f1"]  # straight from stored records
    sanity = {item["method"]: item for item in report["sanity"]["runs"]}
    assert all(item["status"] == "PASS" for item in sanity.values()), sanity
    assert sanity["frozen"]["checks"]["encoder_unchanged"]
    assert sanity["partial"]["checks"]["only_layer4_changed"]
    assert sanity["partial"]["checks"]["no_batchnorm_drift"]
    assert report["sanity"]["augmentation"]["status"] == "PASS"


def test_unfinished_and_tampered_slots_have_null_metrics(paths):
    run_study({**STUDY, "methods": ["scratch"]}, paths=paths, init_override="random")
    slot = plan_study({**STUDY, "methods": ["scratch"]}, paths)[0]
    attempt = artifacts.attempt_dirs(paths, slot.run_id)[0]
    report = build_study_report(STUDY, paths, paths.reports, sanity=False, errors=False)
    rows = {r["method"]: r for r in report["rows"]}
    assert rows["scratch"]["status"] == "COMPLETED"
    assert rows["frozen"]["status"] == "NOT_RUN" and rows["frozen"]["macro_f1"] is None
    (attempt / "predictions_val.csv").write_text(
        (attempt / "predictions_val.csv").read_text().replace(",0,", ",1,", 1)
    )
    report = build_study_report(STUDY, paths, paths.reports, sanity=False, errors=False)
    rows = {r["method"]: r for r in report["rows"]}
    assert rows["scratch"]["status"] == "EVIDENCE_FAILED" and rows["scratch"]["macro_f1"] is None
    with open(report["out"] / "comparison.csv", newline="") as handle:
        assert {r["method"]: r["status"] for r in csv.DictReader(handle)}["scratch"] == (
            "EVIDENCE_FAILED"
        )


def test_error_selection_rule_is_fixed():
    ids = ["c", "a", "b", "d"]
    assert select_errors(ids, [0, 0, 1, 1], [1, 0, 0, 1], limit=12) == [2, 0]  # b then c


def test_study_config_is_strict(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(
        "study_id: x\ncategory: mvp\ndatasets: [flowers102]\nshots: [1]\n"
        "methods: [scratch]\nsupport_seeds: [11]\nextra: 1\n"
    )
    with pytest.raises(ConfigError, match="unknown keys"):
        load_study(path)
