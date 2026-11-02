"""P6: the demonstration app, live prediction, the run registry and the sealed replay view.

Most tests share one SOFTWARE FIXTURE project (noisy red/green/blue images, CPU, random init):
four runs are trained, sealed and evaluated once on test, exactly as in P5. The last tests
smoke-test the real project when its data and runs exist on this machine.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image
from streamlit.testing.v1 import AppTest

from data_fixtures import training_fixture
from fewshotlab.app_launch import startup_check, streamlit_command
from fewshotlab.cli import main
from fewshotlab.data.manifest import ClassMap, read_manifest
from fewshotlab.data.registry import manifest_path
from fewshotlab.engine import predict
from fewshotlab.engine import seal as S
from fewshotlab.engine.study import run_study
from fewshotlab.engine.test_eval import run_test_evaluation
from fewshotlab.hashing import sha256_file
from fewshotlab.reporting import run_registry
from fewshotlab.simulation import corruptions, sealed_replay

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app" / "streamlit_app.py"
PAGES = (
    "views/overview.py",
    "views/data_budget.py",
    "views/explorer.py",
    "views/predict.py",
    "views/replay.py",
    "views/reproducibility.py",
)
ENV = {
    "data_root": "FEWSHOTLAB_DATA_ROOT",
    "manifests": "FEWSHOTLAB_MANIFESTS_DIR",
    "runs": "FEWSHOTLAB_RUNS_DIR",
    "reports": "FEWSHOTLAB_REPORTS_DIR",
    "cache": "FEWSHOTLAB_CACHE_DIR",
    "demo_assets": "FEWSHOTLAB_DEMO_ASSETS_DIR",
}
METHODS = ("scratch", "frozen", "partial", "prototypes")
STUDY = {
    "study_id": "fixture-p6",
    "category": "fixture",
    "datasets": ["fixture-color"],
    "shots": [4],
    "methods": list(METHODS),
    "support_seeds": [11],
    "device": "cpu",
    "demo": {"dataset": "fixture-color", "k": 4, "seed": 11},
}


def _project(root: Path, mp: pytest.MonkeyPatch, study: dict) -> SimpleNamespace:
    locations = {
        "data_root": root / "data" / "raw",
        "manifests": root / "data" / "manifests",
        "runs": root / "runs",
        "reports": root / "reports",
        "cache": root / ".cache",
        "demo_assets": root / "demo_assets",
    }
    for key in ("manifests", "runs", "reports", "demo_assets"):
        locations[key].mkdir(parents=True)
    for key, name in ENV.items():
        mp.setenv(name, str(locations[key]))
    study_file = root / f"{study['study_id']}.yaml"
    study_file.write_text(yaml.safe_dump(study), encoding="utf-8")
    mp.setenv("FEWSHOTLAB_APP_STUDY", str(study_file))
    paths = training_fixture(locations["data_root"])
    return SimpleNamespace(root=root, paths=paths, study_file=study_file)


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        project = _project(tmp_path_factory.mktemp("p6") / "project", mp, STUDY)
        outcomes = run_study(STUDY, paths=project.paths, init_override="random")
        assert [o["status"] for o in outcomes] == ["COMPLETED"] * 4
        S.seal(STUDY, project.paths)
        run_test_evaluation(STUDY, project.paths)
        yield project


def _digests(*roots: Path) -> dict[str, str]:
    return {
        str(p): sha256_file(p) for root in roots for p in sorted(root.rglob("*")) if p.is_file()
    }


def _png(size=(32, 32), color=(200, 40, 40), fmt="PNG", **save) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt, **save)
    return buffer.getvalue()


def _app_page(page: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


# --- uploads -----------------------------------------------------------------------------------


def test_uploads_are_accepted_by_content_and_rejected_with_a_reason():
    for fmt in ("PNG", "JPEG", "BMP", "TIFF", "WEBP"):
        assert predict.decode_upload(_png(fmt=fmt)).size == (32, 32)
    cases = [
        (b"", "empty"),
        (b"just some text, not an image" * 4, "not a readable image"),
        (_png(fmt="GIF"), "GIF images are not accepted"),
        (
            _png(fmt="PNG", save_all=True, append_images=[Image.new("RGB", (32, 32))]),
            "animated or multi-page",
        ),
    ]
    for data, reason in cases:
        with pytest.raises(predict.UploadRejected, match=reason):
            predict.decode_upload(data)
    with pytest.raises(predict.UploadRejected, match="the limit is"):
        predict.decode_upload(_png(), max_bytes=10)
    with pytest.raises(predict.UploadRejected, match="pixels"):
        predict.decode_upload(_png(size=(200, 200)), max_pixels=1000)
    # EXIF orientation 6 (rotate 90 degrees) is applied, as for dataset images.
    exif = Image.Exif()
    exif[0x0112] = 6
    rotated = predict.decode_upload(_png(size=(40, 20), fmt="JPEG", exif=exif.tobytes()))
    assert rotated.size == (20, 40)


# --- model identity and live predictions -------------------------------------------------------


def test_models_load_only_after_their_identity_checks(sealed, monkeypatch):
    registry = run_registry.load_registry(STUDY, sealed.paths)
    with pytest.raises(predict.ModelUnavailable, match="NOT RUN"):
        predict.model_ref(sealed.paths, "no-such-run")
    with pytest.raises(predict.ModelUnavailable, match="not part of"):
        run_registry.cell_model(sealed.paths, registry, "fixture-color", "partial", 7, 11)
    ref = run_registry.cell_model(sealed.paths, registry, "fixture-color", "partial", 4, 11)
    assert ref.artifact_sha256 == registry["seal"]["artifacts"][ref.run_id]
    with pytest.raises(predict.ModelUnavailable, match="differs from the sealed plan"):
        predict.model_ref(sealed.paths, ref.run_id, sealed_artifact_sha256="0" * 64)
    with monkeypatch.context() as m:
        m.setattr(predict, "sha256_file", lambda _path: "0" * 64)
        with pytest.raises(predict.ModelUnavailable, match="differs from its run record"):
            predict.model_ref(sealed.paths, ref.run_id)
    with monkeypatch.context() as m:
        other = ClassMap("fixture-color", ("cyan", "magenta", "yellow"))
        m.setattr(predict, "load_class_map", lambda _path: other)
        with pytest.raises(predict.ModelUnavailable, match="different fixture-color class map"):
            predict.model_ref(sealed.paths, ref.run_id)


def test_live_predictions_match_stored_validation_and_the_cli(sealed, tmp_path, capsys):
    paths = sealed.paths
    registry = run_registry.load_registry(STUDY, paths)
    records = read_manifest(manifest_path(paths, "fixture-color"), splits=("val",))
    for method in METHODS:
        ref = run_registry.cell_model(paths, registry, "fixture-color", method, 4, 11)
        model = predict.load_model(paths, ref, "cpu")
        through_dataset = predict.predict_records(model, records, paths.data_root)
        uploads = [
            predict.decode_upload((paths.data_root / r.relative_path).read_bytes()) for r in records
        ]
        through_upload = predict.predict_images(model, uploads)
        for record, a, b in zip(records, through_dataset, through_upload, strict=True):
            assert a.scores == b.scores, "the upload path must equal the evaluation path"
            same = predict.compare_scores(
                a, predict.stored_validation_prediction(paths, ref, record.image_id)
            )
            assert same["same_class"] and same["max_abs_score_difference"] == 0.0
        # The CLI scores a file through the same code; its name and extension are irrelevant.
        # The app also scores one image at a time, so the comparison is batch for batch: the
        # same image alone is bit-identical, and within the declared tolerance of a larger batch
        # (CPU kernels may round differently for another batch size).
        renamed = tmp_path / f"{method}.jpg"
        renamed.write_bytes((paths.data_root / records[0].relative_path).read_bytes())
        assert main(["predict", "--run", ref.run_id, "--image", str(renamed), "--json"]) == 0
        printed = json.loads(capsys.readouterr().out)[0]
        alone = predict.predict_images(model, [uploads[0]])[0]
        assert printed["scores"] == list(alone.scores)
        batch = predict.compare_scores(
            alone,
            {"pred_class_id": through_upload[0].pred_class_id, "scores": through_upload[0].scores},
        )
        assert batch["same_class"] and batch["within_tolerance"]
        assert printed["score_label"] == "model score" and printed["run_id"] == ref.run_id
    assert main(["predict", "--run", ref.run_id, "--image", str(tmp_path / "missing.png")]) == 1
    assert "cannot read the file" in capsys.readouterr().err


# --- registry and sealed replay ----------------------------------------------------------------


def test_registry_reports_measured_values_and_the_seal(sealed):
    registry = run_registry.load_registry(STUDY, sealed.paths)
    val, test = registry["rows"]["val"], registry["rows"]["test"]
    assert {r["status"] for r in val} == {"COMPLETED"} and all(
        r["macro_f1"] is not None for r in val
    )
    assert {r["status"] for r in test} == {"EVALUATED"}
    plan = json.loads((sealed.paths.reports / "fixture-p6" / S.SEALED_PLAN).read_text())
    seal = registry["seal"]
    assert seal["audit_token"] == plan["audit_token"] and seal["test_evaluations"] == 4
    assert registry["datasets"]["fixture-color"]["classes"] == 3
    assert run_registry.default_cell(registry, "fixture-color")[1:] == (
        plan["content"]["demo"]["selected"]["method"],
        4,
        11,
    )


def test_the_sealed_replay_verifies_and_fails_closed(sealed, monkeypatch):
    paths = sealed.paths
    view = sealed_replay.load_sealed_replay(STUDY, paths)
    assert [s.stream_id for s in view.streams][0] == "clean" and len(view.streams) == 7
    clean = view.items("clean")
    assert len(clean) == 6 and all(len(view.items(s.stream_id)) == 6 for s in view.streams)
    # Every stream keeps the parents' sealed order, so a derivative is compared with its parent.
    for stream in view.streams:
        assert [i.parent_image_id for i in view.items(stream.stream_id)] == [
            i.parent_image_id for i in clean
        ]
        assert view.pixels(view.items(stream.stream_id)[0]).dtype.name == "uint8"
    tally = view.tally("clean")
    assert tally["shown"] == 6 and tally["reviewed"] + tally["accepted"] == 6
    assert abs(tally["coverage"] - view.summaries["clean"]["coverage"]) < 1e-12

    with pytest.raises(sealed_replay.ReplayUnavailable, match="no intact sealed plan"):
        sealed_replay.load_sealed_replay({**STUDY, "study_id": "not-sealed"}, paths)
    with monkeypatch.context() as m:
        changed = {**corruptions.protocol(), "seed_base": 704}
        m.setattr(sealed_replay.corruptions, "protocol", lambda: changed)
        with pytest.raises(sealed_replay.ReplayUnavailable, match="corruption protocol"):
            sealed_replay.load_sealed_replay(STUDY, paths)
    real_hash = sealed_replay.sha256_file
    with monkeypatch.context() as m:
        m.setattr(
            sealed_replay,
            "sha256_file",
            lambda p: "0" * 64 if Path(p).name == "corruptions.py" else real_hash(p),
        )
        with pytest.raises(sealed_replay.ReplayUnavailable, match="changed after sealing"):
            sealed_replay.load_sealed_replay(STUDY, paths)
    for target in (
        paths.reports / "fixture-p6" / "replay_manifest.csv",
        paths.runs
        / "artifacts"
        / view.identity["run_id"]
        / view.identity["attempt_id"]
        / "robustness"
        / "predictions_test__clean.csv",
    ):
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n")
            with pytest.raises(sealed_replay.ReplayUnavailable, match="fail closed"):
                sealed_replay.load_sealed_replay(STUDY, paths)
        finally:
            target.write_bytes(original)
    first = view.items("gaussian_noise-2")[0]
    key = (first.parent_image_id, first.stream_id)
    saved = view._checksums[key]
    try:
        view._checksums[key] = "0" * 64
        with pytest.raises(sealed_replay.ReplayUnavailable, match="sealed checksum"):
            view.pixels(first)
    finally:
        view._checksums[key] = saved


def test_demo_check_passes_offline_and_the_launcher_stays_local(sealed, capsys):
    result = startup_check(sealed.paths, sealed.study_file)
    statuses = {c["check"]: c["status"] for c in result["checks"]}
    assert result["ok"], result["checks"]
    assert statuses["sealed replay"] == "PASS" and statuses["fixture-color model on cpu"] == "PASS"
    assert main(["demo", "--check", "--study", str(sealed.study_file)]) == 0
    assert "ready" in capsys.readouterr().out
    command = streamlit_command(sealed.paths, host="127.0.0.1", port=8599)
    joined = " ".join(command)
    assert "--browser.gatherUsageStats false" in joined and "--server.address 127.0.0.1" in joined
    assert "--server.maxUploadSize 10" in joined and command[-1] == "false"


# --- the app -----------------------------------------------------------------------------------


def test_every_page_renders_on_the_sealed_fixture_and_writes_nothing(sealed):
    before = _digests(sealed.paths.runs, sealed.paths.reports, sealed.root / "data")
    for page in PAGES:
        at = _app_page(page)
        assert any("Synthetic images" in e.value for e in at.error), "fixture badge missing"
    explorer = _app_page("views/explorer.py")
    explorer.radio(key="ex_split").set_value("test").run()
    assert any("each run was evaluated" in s.value for s in explorer.success)
    explorer.multiselect(key="ex_methods").set_value([]).run()
    assert any("Choose at least one method" in i.value for i in explorer.info)
    for method in METHODS:
        explorer.multiselect(key="ex_methods").set_value([method]).run()
        assert not explorer.exception
    repro = _app_page("views/reproducibility.py")
    token = run_registry.seal_summary(STUDY, sealed.paths)["audit_token"]
    assert any(token in m.value for m in repro.markdown)
    assert _digests(sealed.paths.runs, sealed.paths.reports, sealed.root / "data") == before


def test_the_prediction_page_loads_each_method_and_matches_stored_predictions(sealed):
    registry = run_registry.load_registry(STUDY, sealed.paths)
    at = _app_page("views/predict.py")
    assert any("Upload an image" in i.value for i in at.info)
    at.radio(key="pred_source").set_value("Pick a validation image").run()
    for method in METHODS:
        at.selectbox(key="pred_method_fixture-color").set_value(method).run()
        assert not at.exception, [e.value for e in at.exception]
        ref = run_registry.cell_model(sealed.paths, registry, "fixture-color", method, 4, 11)
        assert any(ref.run_id in m.value for m in at.markdown), f"{method}: wrong model shown"
        assert any("Matches the prediction this run stored" in s.value for s in at.success)


def test_the_replay_page_steps_through_the_sealed_stream(sealed):
    at = _app_page("views/replay.py")
    token = run_registry.seal_summary(STUDY, sealed.paths)["audit_token"]
    assert any(token in m.value for m in at.markdown)
    assert any("Image 1 of 6" in p.proto.text for p in at.get("progress"))
    next(b for b in at.button if b.label == "Next").click().run()
    assert any("Image 2 of 6" in p.proto.text for p in at.get("progress"))
    assert next(m for m in at.metric if m.label == "Errors among accepted").value == "hidden"
    at.toggle(key="replay_reveal").set_value(True).run()
    assert next(m for m in at.metric if m.label == "Errors among accepted").value != "hidden"
    at.selectbox(key="replay_stream").set_value("gaussian_blur-2").run()
    assert not at.exception
    captions = [image.caption for element in at.get("image") for image in element.proto.imgs]
    assert any("gaussian_blur-2" in c for c in captions), captions
    assert any("its clean parent" in c for c in captions), captions


def test_an_empty_project_shows_not_run_and_never_crashes(tmp_path, monkeypatch):
    project = _project(tmp_path / "project", monkeypatch, {**STUDY, "study_id": "fixture-p6-empty"})
    before = _digests(project.paths.runs, project.paths.reports)
    overview = _app_page("views/overview.py")
    assert any(m.value == "NOT RUN" for m in overview.metric)
    explorer = _app_page("views/explorer.py")
    assert any("NOT RUN" in i.value for i in explorer.info)
    assert any("No usable model" in e.value for e in _app_page("views/predict.py").error)
    replay = _app_page("views/replay.py")
    assert any("not available" in e.value for e in replay.error)
    assert any("Fallback" in i.value for i in replay.info)
    assert any("NOT SEALED" in i.value for i in _app_page("views/reproducibility.py").info)
    assert _digests(project.paths.runs, project.paths.reports) == before


def test_the_app_never_reaches_training_test_evaluation_or_sealing():
    """Changing a filter or uploading an image can only select or score; it cannot train."""
    forbidden_modules = {
        "fewshotlab.engine.trainer",
        "fewshotlab.engine.runner",
        "fewshotlab.engine.test_eval",
        "fewshotlab.engine.gate",
    }
    forbidden_names = {"run_study", "run_training", "run_test_evaluation", "seal", "build_supports"}
    for path in sorted((REPO / "app").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_modules, f"{path.name} imports {node.module}"
                assert not forbidden_names & {a.name for a in node.names}, path.name
            elif isinstance(node, ast.Import):
                assert not forbidden_modules & {a.name for a in node.names}, path.name


# --- the real project (skipped where its data and runs are absent) -------------------------------

REAL_SEAL = REPO / "reports" / "fewshotlab-v1-full" / "sealed_plan.json"
real_project = pytest.mark.skipif(
    not (
        REAL_SEAL.is_file()
        and (REPO / "data" / "raw").is_dir()
        and (REPO / "runs" / "artifacts").is_dir()
    ),
    reason="the real study's data, runs and seal are not on this machine",
)


def _real_environment(monkeypatch) -> None:
    for name in (*ENV.values(), "FEWSHOTLAB_APP_STUDY"):
        monkeypatch.delenv(name, raising=False)


@real_project
def test_real_project_pages_render_and_write_nothing(monkeypatch):
    _real_environment(monkeypatch)
    ledger = REPO / "runs" / "ledger.jsonl"
    before = (sha256_file(ledger), _digests(REPO / "reports" / "fewshotlab-v1-full"))
    for page in PAGES:
        at = _app_page(page)
        assert not any("Synthetic images" in e.value for e in at.error)
    assert (sha256_file(ledger), _digests(REPO / "reports" / "fewshotlab-v1-full")) == before


@real_project
def test_real_project_demo_check_is_ready_offline(monkeypatch, capsys):
    _real_environment(monkeypatch)
    assert main(["demo", "--check"]) == 0
    out = capsys.readouterr().out
    assert "sealed replay" in out and "ready" in out
