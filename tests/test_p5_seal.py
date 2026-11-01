"""P5: lock-test sealing, fail-closed verification and the one-time sealed test evaluation.

Everything here runs on a SOFTWARE FIXTURE (noisy red/green/blue images, CPU, random init).
"""

from __future__ import annotations

import json

import pytest
import yaml

from data_fixtures import training_fixture
from fewshotlab.cli import main
from fewshotlab.data.manifest import read_manifest
from fewshotlab.data.registry import manifest_path
from fewshotlab.data.support import build_supports, support_path
from fewshotlab.data.transforms import build_transform
from fewshotlab.engine import artifacts
from fewshotlab.engine import seal as S
from fewshotlab.engine.study import run_study
from fewshotlab.engine.test_eval import GatedTestDataset, run_test_evaluation
from fewshotlab.errors import ConfigError, ProtocolRefusal
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import read_json
from fewshotlab.reporting.benchmark_report import write_aggregate_report, write_robustness_report
from fewshotlab.simulation import corruptions, review
from fewshotlab.simulation.replay import REPLAY_FILE

STUDY = {
    "study_id": "fixture-p5",
    "category": "fixture",
    "datasets": ["fixture-color"],
    "shots": [4],
    "methods": ["scratch", "frozen", "partial", "prototypes"],
    "support_seeds": [11],
    "device": "cpu",
    "demo": {"dataset": "fixture-color", "k": 4, "seed": 11},
}


@pytest.fixture
def trained(isolated_paths):
    paths = training_fixture(isolated_paths.data_root)
    outcomes = run_study(STUDY, paths=paths, init_override="random")
    assert [o["status"] for o in outcomes] == ["COMPLETED"] * 4
    return paths


def _digests(root):
    return {
        str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_preview_records_everything_and_seals_nothing(trained):
    doc = S.preview(STUDY, trained)
    assert doc["is_seal"] is False and doc["ready_to_seal"], doc["blocking"]
    content = doc["content"]
    assert len(content["runs"]) == 4 and content["missing"] == [] and content["duplicates"] == []
    assert content["demo"]["selected"]["method"] in STUDY["methods"]
    assert content["demo"]["threshold"]["selected"]["coverage"] >= review.TARGET_COVERAGE
    assert content["calibration"]["method"] == "none"
    assert content["data"]["fixture-color"]["expected_test_ids"]["count"] == 6
    assert content["replay"]["rows"] == 6 * len(corruptions.CONDITIONS)
    assert doc["validation_reproduction"]["all_identical"]
    assert not (trained.reports / STUDY["study_id"] / S.SEALED_PLAN).exists()
    assert S.find_test_outputs(trained) == []
    unsealed = write_aggregate_report(STUDY, trained, trained.reports, "test")
    assert unsealed["written"] is False and not unsealed["out"].exists()
    assert {r["status"] for r in unsealed["rows"]} == {"NOT_EVALUATED"}
    with pytest.raises(ProtocolRefusal, match="no sealed evaluation plan"):
        S.verify_seal(STUDY, trained)
    with pytest.raises(ProtocolRefusal, match="no sealed evaluation plan"):
        run_test_evaluation(STUDY, trained)


def test_seal_then_evaluate_each_run_exactly_once(trained):
    plan = S.seal(STUDY, trained)
    directory = trained.reports / STUDY["study_id"]
    assert plan["audit_token"].startswith("seal-") and not plan["partial"]
    events = S.read_evaluation_ledger(directory)
    assert events[0]["event"] == "sealed" and events[0]["audit_token"] == plan["audit_token"]
    assert S.verify_seal(STUDY, trained).plan_sha256 == plan["plan_sha256"]
    with pytest.raises(ProtocolRefusal, match="already sealed"):
        S.seal(STUDY, trained)

    result = run_test_evaluation(STUDY, trained)
    assert [r["outcome"] for r in result["runs"]] == ["evaluated"] * 4
    assert result["robustness"]["outcome"] == "evaluated"
    test_ids = sorted(
        r.image_id for r in read_manifest(manifest_path(trained, "fixture-color"), splits=("test",))
    )
    for run in plan["content"]["runs"]:
        adir = artifacts.run_root(trained, run["run_id"]) / run["attempt_id"]
        doc = read_json(adir / "test_metrics.json")
        assert doc["split"] == "test" and doc["audit_token"] == plan["audit_token"]
        assert artifacts.read_predictions(adir / "predictions_test.csv")["image_ids"] == test_ids
    assert S.audit_test_outputs(trained) == []

    before = _digests(trained.runs)
    again = run_test_evaluation(STUDY, trained)
    assert {r["outcome"] for r in again["runs"]} == {"already evaluated"}
    assert again["robustness"]["outcome"] == "already evaluated"
    assert _digests(trained.runs) == before  # a recorded evaluation is never re-run

    demo = plan["content"]["demo"]["selected"]
    stress = read_json(
        artifacts.run_root(trained, demo["run_id"])
        / demo["attempt_id"]
        / S.ROBUSTNESS_DIR
        / "robustness_test.json"
    )
    assert [c["condition_id"] for c in stress["conditions"]] == [
        corruptions.condition_id(c, s) for c, s in corruptions.CONDITIONS
    ]
    assert len(stress["paired_vs_clean"]) == len(corruptions.CONDITIONS) - 1
    assert stress["clean_path_consistency"]["identical_predictions"] is True
    assert stress["threshold"] == plan["content"]["demo"]["threshold"]["selected"]["threshold"]

    test_report = write_aggregate_report(STUDY, trained, trained.reports, "test")
    assert {r["status"] for r in test_report["rows"]} == {"EVALUATED"}
    by_run = {r["run_id"]: r for r in test_report["rows"]}
    for run in plan["content"]["runs"]:
        adir = artifacts.run_root(trained, run["run_id"]) / run["attempt_id"]
        stored = read_json(adir / "test_metrics.json")["metrics"]["macro_f1"]
        assert by_run[run["run_id"]]["macro_f1"] == stored
    assert "HELD-OUT TEST" in (test_report["out"] / "aggregate_test.md").read_text()
    assert write_robustness_report(STUDY, trained, trained.reports)["status"] == "VERIFIED"
    assert S.audit_test_outputs(trained) == []

    predictions = artifacts.run_root(trained, demo["run_id"]) / demo["attempt_id"]
    target = predictions / "predictions_test.csv"
    target.write_text(target.read_text().replace(",test,", ",test, ", 1))
    assert any("not recorded" in p for p in S.audit_test_outputs(trained))


def test_changed_inputs_fail_closed(trained, monkeypatch):
    plan = S.seal(STUDY, trained)
    directory = trained.reports / STUDY["study_id"]
    run = plan["content"]["runs"][0]
    adir = artifacts.run_root(trained, run["run_id"]) / run["attempt_id"]

    def refused(match):
        with pytest.raises(ProtocolRefusal, match=match):
            S.verify_seal(STUDY, trained)

    checkpoint = adir / run["artifact"]["file"]
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"\0")
    refused("no longer matches")
    with pytest.raises(ProtocolRefusal):
        run_test_evaluation(STUDY, trained)
    assert S.find_test_outputs(trained) == []
    checkpoint.write_bytes(original)

    support = support_path(trained, "fixture-color", 4, 11)
    text = support.read_text()
    doc = json.loads(text)
    doc["selected"]["0"][0] = doc["selected"]["1"][0]
    support.write_text(json.dumps(doc))
    refused("can no longer be re-derived")
    support.write_text(text)

    with monkeypatch.context() as patch:
        patch.setattr(review, "TARGET_COVERAGE", 0.5)
        refused("no longer matches")
    with monkeypatch.context() as patch:
        patch.setitem(corruptions.PARAMETERS, ("brightness", 1), {"factor": 0.5})
        refused("no longer matches")

    plan_path = directory / S.SEALED_PLAN
    text = plan_path.read_text()
    altered = json.loads(text)
    altered["content"]["demo"]["threshold"]["selected"]["threshold"] = 0.0
    plan_path.write_text(json.dumps(altered))
    refused("altered")
    plan_path.write_text(text)

    ledger = directory / S.EVALUATION_LEDGER
    text = ledger.read_text()
    ledger.write_text("")
    refused("audit token")
    ledger.write_text(text)

    replay = directory / REPLAY_FILE
    data = replay.read_bytes()
    replay.write_bytes(data.replace(b"clean", b"CLEAN", 1))
    refused(REPLAY_FILE)
    replay.write_bytes(data)

    assert S.verify_seal(STUDY, trained).audit_token == plan["audit_token"]


def test_test_images_need_a_verified_seal(trained):
    records = read_manifest(manifest_path(trained, "fixture-color"), splits=("test",))
    with pytest.raises(ProtocolRefusal):
        GatedTestDataset(
            records, trained.data_root, build_transform("fixture-color", "eval"), seal=object()
        )


def test_unrecorded_test_outputs_block_sealing(trained):
    run = S.build_seal_plan(STUDY, trained)["content"]["runs"][0]
    (
        artifacts.run_root(trained, run["run_id"]) / run["attempt_id"] / "test_metrics.json"
    ).write_text("{}")
    assert any("not recorded" in p for p in S.audit_test_outputs(trained))
    with pytest.raises(ProtocolRefusal, match="test-split outputs already exist"):
        S.seal(STUDY, trained)


def test_partial_benchmark_needs_an_explicit_opt_in(trained):
    build_supports(trained, "fixture-color", shots=(4,), seeds=(11, 22))
    study = {**STUDY, "study_id": "fixture-p5-partial", "support_seeds": [11, 22]}
    content = S.build_seal_plan(study, trained)["content"]
    assert len(content["missing"]) == 3  # learned seed-22 runs; prototypes s22 duplicates s11
    assert [d["method"] for d in content["duplicates"]] == ["prototypes"]
    with pytest.raises(ProtocolRefusal, match="allow-partial"):
        S.seal(study, trained)
    plan = S.seal(study, trained, allow_partial=True)
    assert plan["partial"] and len(plan["content"]["runs"]) == 4


def test_core_studies_cannot_move_the_demo_cell(trained):
    from fewshotlab.simulation.demo import demo_target_for

    with pytest.raises(Exception, match="fixes the demo cell"):
        demo_target_for({**STUDY, "category": "core", "demo": {"dataset": "x", "k": 1, "seed": 1}})
    assert demo_target_for({**STUDY, "category": "core", "demo": None}) == {
        "dataset": "neu_cls",
        "k": 10,
        "seed": 11,
    }


def test_study_demo_cell_must_belong_to_the_study(tmp_path):
    from fewshotlab.engine.study import load_study

    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({**STUDY, "demo": {"dataset": "fixture-color", "k": 9, "seed": 11}})
    )
    with pytest.raises(ConfigError, match="not part of the study"):
        load_study(path)


def test_lock_test_and_sealed_evaluation_from_the_command_line(trained, tmp_path, capsys):
    config = tmp_path / "fixture-p5.yaml"
    config.write_text(yaml.safe_dump(STUDY))
    assert main(["lock-test", "--plan", str(config), "--dry-run", "--skip-reproduction"]) == 0
    assert "PREVIEW" in capsys.readouterr().out
    assert main(["evaluate", "--split", "test", "--study", str(config)]) == 5
    assert "locked" in capsys.readouterr().err
    assert main(["evaluate", "--split", "test", "--run", "anything"]) == 5
    capsys.readouterr()
    assert main(["lock-test", "--plan", str(config)]) == 0
    out = capsys.readouterr().out
    assert "SEALED fixture-p5" in out and "seal-" in out
    assert main(["lock-test", "--plan", str(config)]) == 5
    capsys.readouterr()
    assert main(["evaluate", "--split", "test", "--study", str(config), "--device", "cpu"]) == 0
    assert "4 run(s) evaluated now, 0 already recorded" in capsys.readouterr().out
    assert main(["evaluate", "--split", "test", "--study", str(config)]) == 0
    assert "0 run(s) evaluated now, 4 already recorded" in capsys.readouterr().out
    assert main(["evaluate", "--split", "val"]) == 2
    report_args = ["--ledger", str(trained.ledger), "--out", str(trained.reports)]
    assert main(["report", "--study", str(config), *report_args]) == 0
    out = capsys.readouterr().out
    assert "held-out test aggregates (sealed)" in out and "robustness study" in out
