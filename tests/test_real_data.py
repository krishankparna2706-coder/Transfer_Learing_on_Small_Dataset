"""Real-data checks: run against prepared datasets; skipped when a dataset is not prepared.

These read the real manifests, files and support sets. They never train or touch test labels
beyond structural counts.
"""

from __future__ import annotations

import pytest

from fewshotlab.config import EXPECTED_COUNTS, NUM_CLASSES, SHOTS, SUPPORT_SEEDS
from fewshotlab.data.audit import latest_audit
from fewshotlab.data.manifest import (
    SPLITS,
    load_class_map,
    read_manifest,
    validate_records,
    verify_files,
)
from fewshotlab.data.registry import class_map_path, manifest_path, read_prepare_record
from fewshotlab.data.support import load_support, support_path
from fewshotlab.hashing import sha256_file
from fewshotlab.paths import get_paths

pytestmark = pytest.mark.real_data
DATASETS = ("flowers102", "neu_cls")
PER_CLASS = {
    "flowers102": {"train": 10, "val": 10},
    "neu_cls": {"train": 180, "val": 60, "test": 60},
}


def _prepared(dataset):
    paths = get_paths()
    record = read_prepare_record(paths, dataset)
    if not record or record.get("status") != "PREPARED":
        pytest.skip(f"{dataset} is not prepared")
    return paths, record


@pytest.mark.parametrize("dataset", DATASETS)
def test_manifest_counts_and_file_hashes(dataset):
    paths, record = _prepared(dataset)
    mpath = manifest_path(paths, dataset)
    assert sha256_file(mpath) == record["manifest"]["sha256"]
    cmap = load_class_map(class_map_path(paths, dataset))
    assert cmap.num_classes == NUM_CLASSES[dataset]
    rows = read_manifest(mpath, splits=SPLITS)
    validate_records(
        rows,
        cmap,
        expected_counts=EXPECTED_COUNTS[dataset],
        expected_per_class=PER_CLASS[dataset],
        require_all_classes_in=SPLITS,
    )
    verify_files(rows, paths.data_root)


@pytest.mark.parametrize("dataset", DATASETS)
def test_latest_audit_passed_for_the_current_manifest(dataset):
    paths, record = _prepared(dataset)
    audit = latest_audit(paths, dataset)
    if audit is None:
        pytest.skip(f"{dataset} has not been audited yet")
    assert audit["status"] == "PASS"
    assert audit["manifest"]["sha256"] == record["manifest"]["sha256"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_supports_are_exact_nested_and_verified(dataset):
    paths, _ = _prepared(dataset)
    if not support_path(paths, dataset, 1, SUPPORT_SEEDS[0]).is_file():
        pytest.skip(f"{dataset} supports have not been built yet")
    for seed in SUPPORT_SEEDS:
        previous = None
        for k in SHOTS:
            doc = load_support(paths, dataset, k, seed)
            assert doc["n_images"] == k * NUM_CLASSES[dataset]
            if previous is not None:
                assert all(set(previous[c]) <= set(doc["selected"][c]) for c in previous)
            previous = doc["selected"]
    if dataset == "flowers102":  # every seed uses all ten training images per class at K=10
        sets = [
            {c: sorted(v) for c, v in load_support(paths, dataset, 10, s)["selected"].items()}
            for s in SUPPORT_SEEDS
        ]
        assert all(s == sets[0] for s in sets)
