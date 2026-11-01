"""P1: the audit checks integrity and flags duplicates without changing any manifest."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from data_fixtures import install_dataset, make_dataset
from fewshotlab.data.audit import hamming_scan, require_passing_audit, run_audit
from fewshotlab.data.imageio import open_image, phash64
from fewshotlab.data.registry import audit_dir, manifest_path
from fewshotlab.errors import DataValidationError
from fewshotlab.hashing import sha256_file


@pytest.fixture
def dataset(isolated_paths):
    records, cmap = make_dataset(isolated_paths.data_root)
    paths = install_dataset(records, cmap)
    return paths, records, cmap


def test_clean_dataset_passes_and_writes_reports(dataset):
    paths, records, _ = dataset
    before = sha256_file(manifest_path(paths, "fixture"))
    report = run_audit("fixture", paths=paths)
    assert report["status"] == "PASS"
    assert report["counts"] == {"train": 30, "val": 6, "test": 6}
    assert report["exact_duplicates"]["pixel_identical_groups"] == 0
    out = audit_dir(paths, "fixture")
    for name in (
        "data_audit.json",
        "data_audit.md",
        "near_duplicate_candidates.csv",
        "image_audit.csv",
        "audit_history.jsonl",
    ):
        assert (out / name).is_file(), name
    assert sha256_file(manifest_path(paths, "fixture")) == before  # never edited
    require_passing_audit(paths, "fixture", before)
    run_audit("fixture", paths=paths)
    assert len((out / "audit_history.jsonl").read_text().splitlines()) == 2  # append-only


def _replace_file(paths, records, cmap, index, data: bytes):
    path = paths.data_root / records[index].relative_path
    path.write_bytes(data)
    records = list(records)
    records[index] = dataclasses.replace(records[index], sha256=sha256_file(path))
    install_dataset(records, cmap)
    return records


def test_cross_split_exact_duplicate_is_flagged(dataset):
    paths, records, cmap = dataset
    train_index = next(i for i, r in enumerate(records) if r.split == "train")
    test_index = next(i for i, r in enumerate(records) if r.split == "test" and r.class_id == 0)
    source = (paths.data_root / records[train_index].relative_path).read_bytes()
    _replace_file(paths, records, cmap, test_index, source)
    report = run_audit("fixture", paths=paths)
    assert report["status"] == "PASS"  # integrity is fine; this is a finding, not a repair
    assert len(report["exact_duplicates"]["cross_split_groups"]) == 1
    assert any("span splits" in f for f in report["findings"])


def test_near_duplicate_is_listed_for_review(dataset):
    paths, records, cmap = dataset
    train_index = next(i for i, r in enumerate(records) if r.split == "train")
    val_index = next(i for i, r in enumerate(records) if r.split == "val")
    image = np.asarray(open_image(paths.data_root / records[train_index].relative_path))
    shifted = np.clip(image.astype(int) + 3, 0, 255).astype(np.uint8)
    from PIL import Image

    target = paths.data_root / records[val_index].relative_path
    Image.fromarray(shifted).save(target)
    _replace_file(paths, records, cmap, val_index, target.read_bytes())
    report = run_audit("fixture", paths=paths)
    assert report["near_duplicates"]["cross_split_pairs"] >= 1
    rows = (audit_dir(paths, "fixture") / "near_duplicate_candidates.csv").read_text()
    assert records[val_index].image_id in rows


def test_corrupt_or_changed_file_fails_the_audit(dataset):
    paths, records, _ = dataset
    path = paths.data_root / records[0].relative_path
    path.write_bytes(path.read_bytes()[:40])  # truncated, and the hash no longer matches
    report = run_audit("fixture", paths=paths)
    assert report["status"] == "FAIL"
    assert report["integrity_failures"] and report["decode_failures"]
    with pytest.raises(DataValidationError, match="FAIL"):
        require_passing_audit(paths, "fixture", sha256_file(manifest_path(paths, "fixture")))


def test_phash_distances(tmp_path):
    from data_fixtures import noise_image

    a = noise_image(1, (64, 64))
    b = noise_image(2, (64, 64))
    assert phash64(a) == phash64(a.copy())
    distance = bin(phash64(a) ^ phash64(b)).count("1")
    assert distance > 12


def test_hamming_scan_matches_brute_force():
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2**63, size=40, dtype=np.int64).astype(np.uint64)
    hashes[7] = hashes[3] ^ np.uint64(0b101)  # distance 2
    pairs, histogram = hamming_scan(hashes, max_distance=4, chunk=7)
    brute = [
        (i, j, bin(int(hashes[i]) ^ int(hashes[j])).count("1"))
        for i in range(40)
        for j in range(i + 1, 40)
    ]
    assert sorted(pairs) == sorted(p for p in brute if p[2] <= 4)
    assert histogram.sum() == len(brute)
    assert (3, 7, 2) in pairs


def test_audit_requires_a_prepared_dataset(isolated_paths):
    from fewshotlab.errors import DatasetUnavailableError

    with pytest.raises(DatasetUnavailableError, match="NOT_PREPARED"):
        run_audit("fixture")


def test_report_is_json_serializable(dataset):
    paths, _, _ = dataset
    report = run_audit("fixture", paths=paths)
    json.dumps(report)
