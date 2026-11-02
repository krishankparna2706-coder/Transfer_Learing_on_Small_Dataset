"""P7: live replay inference, the timing protocol, the offline rehearsal and separation checks.

Uses the P6 SOFTWARE FIXTURE setup (noisy red/green/blue images, CPU): four runs are trained,
sealed and evaluated once. The real-project check is skipped where its files are absent.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import socket
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fewshotlab.cli import main
from fewshotlab.data.manifest import read_manifest
from fewshotlab.data.registry import manifest_path
from fewshotlab.data.support import load_support
from fewshotlab.engine import artifacts, predict
from fewshotlab.engine import seal as S
from fewshotlab.engine.ledger import read_events
from fewshotlab.engine.study import load_study, run_study
from fewshotlab.engine.test_eval import run_test_evaluation
from fewshotlab.hashing import sha256_file
from fewshotlab.paths import get_paths
from fewshotlab.simulation import rehearsal, replay_live
from fewshotlab.simulation.sealed_replay import load_sealed_replay
from test_p6_app import APP, STUDY, _project

P7_STUDY = {**STUDY, "study_id": "fixture-p7"}
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        project = _project(tmp_path_factory.mktemp("p7") / "project", mp, P7_STUDY)
        outcomes = run_study(P7_STUDY, paths=project.paths, init_override="random")
        assert {o["status"] for o in outcomes} == {"COMPLETED"}
        S.seal(P7_STUDY, project.paths)
        run_test_evaluation(P7_STUDY, project.paths)
        yield project


def test_live_replay_matches_the_sealed_predictions_and_times_each_stage(sealed):
    view = load_sealed_replay(P7_STUDY, sealed.paths)
    live = replay_live.LiveReplay.load(view, sealed.paths, "cpu")
    assert live.cold_start_ms > 0
    for stream in view.streams:
        for item in view.items(stream.stream_id):
            result = live.infer(item)
            assert result.matches_stored, (item.derivative_id, result.score_difference)
            assert result.preprocess_ms > 0 and 0 < result.forward_ms <= result.end_to_end_ms
    timing = live.time_forward(view.items("clean")[0], warmup=3, timed=7)
    assert timing["n"] == 7 and len(timing["samples_ms"]) == 7
    assert timing["batch_size"] == 1 and timing["warmup_requests"] == 3
    assert timing["input_shape"][0] == 1 and timing["median_ms"] <= timing["p95_ms"]


def test_the_live_model_must_be_the_sealed_checkpoint(sealed):
    view = load_sealed_replay(P7_STUDY, sealed.paths)
    forged = dataclasses.replace(view, identity={**view.identity, "checkpoint_sha256": "0" * 64})
    with pytest.raises(predict.ModelUnavailable, match="sealed plan"):
        replay_live.LiveReplay.load(forged, sealed.paths, "cpu")


def test_the_rehearsal_runs_offline_and_records_everything(sealed, capsys):
    assert main(["demo", "--rehearse", "--study", str(sealed.study_file)]) == 0
    assert "REPLAY REHEARSAL" in capsys.readouterr().out
    out = rehearsal.output_dir(sealed.paths, "fixture-p7")
    summary = json.loads((out / rehearsal.SUMMARY_FILE).read_text())
    assert summary["all_live_match_stored"] and summary["per_image"]["images"] == 7 * 6
    fwd = summary["forward_latency"]
    assert (fwd["warmup_requests"], fwd["timed_requests"], len(fwd["samples_ms"])) == (20, 200, 200)
    with (out / rehearsal.EVENTS_FILE).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 42 and {r["matches_stored"] for r in rows} == {"True"}
    assert {r["run_id"] for r in rows} == {summary["model"]["run_id"]}
    event = [e for e in read_events(sealed.paths) if e.get("event") == "replay_rehearsal"][-1]
    for name, digest in event["files"].items():
        assert sha256_file(out / name) == digest
    assert "not a benchmark result" in (out / rehearsal.REPORT_FILE).read_text(encoding="utf-8")
    assert S.audit_test_outputs(sealed.paths) == []


def test_no_network_refuses_every_connection_and_restores_it():
    with rehearsal.no_network(), pytest.raises(rehearsal.NetworkAttempt):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    with pytest.raises(OSError):  # a closed local port, refused by the system as usual
        socket.create_connection(("127.0.0.1", 9), timeout=1)


def _separation_problems(paths, study: dict) -> list[str]:
    """Corruption results stay apart from the benchmark; training sees only train images."""
    problems = []
    plan = json.loads((paths.reports / study["study_id"] / S.SEALED_PLAN).read_text())
    for run in plan["content"]["runs"]:
        folder = artifacts.run_root(paths, run["run_id"]) / run["attempt_id"]
        with (folder / "predictions_test.csv").open(encoding="utf-8") as handle:
            if {row["condition"] for row in csv.DictReader(handle)} != {"clean"}:
                problems.append(f"{run['run_id']}: non-clean rows in the benchmark predictions")
    demo = plan["content"]["demo"]["selected"]
    robust = artifacts.run_root(paths, demo["run_id"]) / demo["attempt_id"] / "robustness"
    clean = artifacts.read_predictions(robust / "predictions_test__clean.csv")["image_ids"]
    for path in sorted(robust.glob("predictions_test__*.csv")):
        if artifacts.read_predictions(path)["image_ids"] != clean:
            problems.append(f"{path.name}: parent IDs differ from the clean stream")
    for dataset in study["datasets"]:
        train = {
            r.image_id for r in read_manifest(manifest_path(paths, dataset), splits=("train",))
        }
        for k in study["shots"]:
            for seed in study["support_seeds"]:
                ids = {
                    i for v in load_support(paths, dataset, k, seed)["selected"].values() for i in v
                }
                if not ids <= train or any("#" in i for i in ids):
                    problems.append(f"{dataset} K={k} seed {seed}: support outside the train split")
    return problems


def test_corruption_results_stay_apart_and_training_sees_only_train_images(sealed):
    assert _separation_problems(sealed.paths, P7_STUDY) == []


@pytest.mark.skipif(
    not (REPO / "reports" / "fewshotlab-v1-full" / S.SEALED_PLAN).is_file(),
    reason="the real study's seal and runs are not on this machine",
)
def test_real_study_keeps_corruption_results_apart(monkeypatch):
    for name in ("DATA_ROOT", "MANIFESTS_DIR", "RUNS_DIR", "REPORTS_DIR", "CACHE_DIR"):
        monkeypatch.delenv(f"FEWSHOTLAB_{name}", raising=False)
    study = load_study(REPO / "configs" / "full-study.yaml")
    assert _separation_problems(get_paths(), study) == []


def test_the_replay_page_runs_live_and_falls_back_visibly(sealed, monkeypatch):
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.switch_page("views/replay.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    captions = [c.value for c in at.caption]
    assert any("this request" in c for c in captions), "no measured latency shown"
    assert any("matches the stored sealed prediction" in c for c in captions)
    assert at.get("download_button"), "no event-log download"

    def unavailable(*_args):
        raise predict.ModelUnavailable("the checkpoint is missing")

    monkeypatch.setattr(sys.modules["views.common"], "live_replay", unavailable)
    again = AppTest.from_file(str(APP), default_timeout=180)
    again.run()
    again.switch_page("views/replay.py")
    again.run()
    assert not again.exception
    assert any("Live inference is unavailable" in w.value for w in again.warning)
    assert any("Source: the stored sealed prediction" in c.value for c in again.caption)
