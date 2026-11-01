"""P1: nested, shared, reproducible K-shot supports that never touch validation or test."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys

import pytest

from data_fixtures import install_dataset, make_dataset
from fewshotlab.data.audit import run_audit
from fewshotlab.data.manifest import read_manifest
from fewshotlab.data.registry import manifest_path, supports_dir
from fewshotlab.data.support import build_supports, load_support, support_path
from fewshotlab.errors import DataValidationError
from fewshotlab.io_utils import read_json

SHOTS = (1, 3, 5, 10)
SEEDS = (11, 22)


@pytest.fixture
def prepared(isolated_paths):
    records, cmap = make_dataset(isolated_paths.data_root)
    paths = install_dataset(records, cmap)
    run_audit("fixture", paths=paths)
    return paths, records


def test_exact_k_nested_and_inside_train(prepared):
    paths, records = prepared
    build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS)
    train = {r.image_id: r.class_id for r in records if r.split == "train"}
    heldout = {r.image_id for r in records if r.split != "train"}
    for seed in SEEDS:
        previous = None
        for k in SHOTS:
            doc = load_support(paths, "fixture", k, seed)
            assert doc["n_images"] == 3 * k
            for key, ids in doc["selected"].items():
                assert len(ids) == k
                assert all(train[i] == int(key) for i in ids)
                assert not set(ids) & heldout
                if previous is not None:
                    assert set(previous[key]) <= set(ids)
            previous = doc["selected"]
    one = load_support(paths, "fixture", 10, 11)["selected"]
    other = load_support(paths, "fixture", 10, 22)["selected"]
    assert one != other or all(len(v) == 10 for v in one.values())


def test_rebuild_is_idempotent_and_hash_ignores_timestamps(prepared):
    paths, _ = prepared
    first = build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS)
    second = build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS)
    assert [r["content_sha256"] for r in first] == [r["content_sha256"] for r in second]
    assert {r["state"] for r in second} == {"verified (unchanged)"}


def test_membership_is_reproducible_across_processes(prepared, tmp_path):
    paths, _ = prepared
    in_process = build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS)
    fresh = tmp_path / "fresh-manifests"
    shutil.copytree(paths.manifests, fresh, ignore=shutil.ignore_patterns("supports"))
    code = (
        "import json; from fewshotlab.paths import get_paths; "
        "from fewshotlab.data.support import build_supports; "
        "r = build_supports(get_paths(), 'fixture', shots=(1, 3, 5, 10), seeds=(11, 22), "
        "require_audit=False); print(json.dumps([x['content_sha256'] for x in r]))"
    )
    env = dict(os.environ, FEWSHOTLAB_MANIFESTS_DIR=str(fresh))
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert json.loads(out.stdout) == [r["content_sha256"] for r in in_process]


def test_k_larger_than_available_is_an_error(isolated_paths):
    records, cmap = make_dataset(
        isolated_paths.data_root, per_split={"train": 2, "val": 1, "test": 1}
    )
    paths = install_dataset(records, cmap)
    run_audit("fixture", paths=paths)
    with pytest.raises(DataValidationError, match="with replacement"):
        build_supports(paths, "fixture", shots=(1, 3), seeds=(11,))


def test_supports_are_immutable_when_the_split_changes(prepared):
    paths, records = prepared
    build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS)
    changed = [
        dataclasses.replace(r, image_id=r.image_id + "x")
        if r.split == "train" and r.class_id == 0
        else r
        for r in records
    ]
    from fewshotlab.data.manifest import ClassMap

    install_dataset(changed, ClassMap("fixture", ("class_0", "class_1", "class_2")))
    with pytest.raises(DataValidationError, match="immutable"):
        build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS, require_audit=False)


def test_tampering_and_leakage_are_detected(prepared):
    paths, records = prepared
    build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS)
    path = support_path(paths, "fixture", 3, 11)
    doc = read_json(path)
    val_id = next(r.image_id for r in records if r.split == "val" and r.class_id == 0)
    doc["selected"]["0"][0] = val_id
    path.write_text(json.dumps(doc))
    with pytest.raises(DataValidationError, match="validation/test image"):
        load_support(paths, "fixture", 3, 11)


def test_support_build_requires_a_passing_audit(isolated_paths):
    records, cmap = make_dataset(isolated_paths.data_root)
    paths = install_dataset(records, cmap)
    with pytest.raises(DataValidationError, match="data audit"):
        build_supports(paths, "fixture", shots=SHOTS, seeds=SEEDS)
    assert not supports_dir(paths).exists()


def test_reader_never_returns_test_rows_for_train_requests(prepared):
    paths, _ = prepared
    rows = read_manifest(manifest_path(paths, "fixture"), splits=("train",))
    assert {r.split for r in rows} == {"train"}


def test_more_than_ten_classes_regression(isolated_paths):
    """Class keys are strings; '10' sorts before '2', so keys must be compared as a set."""
    records, cmap = make_dataset(
        isolated_paths.data_root,
        num_classes=12,
        per_split={"train": 10, "val": 1, "test": 1},
        size=(16, 16),
    )
    paths = install_dataset(records, cmap)
    run_audit("fixture", paths=paths)
    built = build_supports(paths, "fixture", shots=SHOTS, seeds=(11,))
    assert len(built) == 4
    assert load_support(paths, "fixture", 10, 11)["n_images"] == 120
