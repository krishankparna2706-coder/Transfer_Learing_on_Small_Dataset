"""P1: manifests are canonical, and validation rejects every structural problem."""

from __future__ import annotations

import dataclasses
import os

import pytest

from data_fixtures import make_dataset
from fewshotlab.data.manifest import (
    ClassMap,
    load_class_map,
    manifest_bytes,
    read_manifest,
    validate_records,
    verify_files,
    write_class_map,
    write_manifest,
)
from fewshotlab.errors import DataValidationError


@pytest.fixture
def fixture(tmp_path):
    records, cmap = make_dataset(tmp_path / "raw")
    return tmp_path, records, cmap


def test_round_trip_and_canonical_bytes(fixture):
    tmp, records, _ = fixture
    digest = write_manifest(records, tmp / "m.csv")
    assert write_manifest(list(reversed(records)), tmp / "m2.csv") == digest  # order-independent
    loaded = read_manifest(tmp / "m.csv", splits=("train", "val", "test"))
    assert sorted(loaded, key=lambda r: r.image_id) == sorted(records, key=lambda r: r.image_id)
    assert manifest_bytes(loaded) == (tmp / "m.csv").read_bytes()


def test_reader_requires_explicit_splits(fixture):
    tmp, records, _ = fixture
    write_manifest(records, tmp / "m.csv")
    train_val = read_manifest(tmp / "m.csv", splits=("train", "val"))
    assert {r.split for r in train_val} == {"train", "val"}
    with pytest.raises(DataValidationError):
        read_manifest(tmp / "m.csv", splits=())
    with pytest.raises(DataValidationError):
        read_manifest(tmp / "m.csv", splits=("holdout",))


def test_valid_manifest_passes(fixture):
    _, records, cmap = fixture
    validate_records(
        records,
        cmap,
        expected_counts={"train": 30, "val": 6, "test": 6},
        expected_per_class={"train": 10, "val": 2},
        require_all_classes_in=("train", "val", "test"),
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rs: rs + [rs[0]], "duplicate image_id"),
        (
            lambda rs: rs + [dataclasses.replace(rs[1], image_id="new-id")],
            "duplicate relative_path",
        ),
        (lambda rs: [dataclasses.replace(rs[0], class_id=99)] + rs[1:], "unknown class_id"),
        (lambda rs: [dataclasses.replace(rs[0], class_name="other")] + rs[1:], "does not match"),
        (lambda rs: [dataclasses.replace(rs[0], sha256="xyz")] + rs[1:], "sha256"),
        (lambda rs: [dataclasses.replace(rs[0], split="holdout")] + rs[1:], "unknown split"),
    ],
)
def test_structural_problems_are_rejected(fixture, mutate, message):
    _, records, cmap = fixture
    with pytest.raises(DataValidationError, match=message):
        validate_records(mutate(records), cmap)


@pytest.mark.parametrize(
    "bad_path", ["../x.png", "/abs/x.png", "a/../b.png", "a\\b.png", "./a.png", "a//b.png", ""]
)
def test_unsafe_paths_are_rejected(fixture, bad_path):
    _, records, cmap = fixture
    with pytest.raises(DataValidationError, match="unsafe relative_path"):
        validate_records(
            [dataclasses.replace(records[0], relative_path=bad_path)] + records[1:], cmap
        )


def test_overlapping_and_merged_splits_are_caught(fixture):
    _, records, cmap = fixture
    val = [r for r in records if r.split == "val"]
    # the same image listed again under test is a cross-split overlap
    with pytest.raises(DataValidationError, match="duplicate image_id"):
        validate_records(records + [dataclasses.replace(val[0], split="test")], cmap)
    # silently merging validation into training changes the per-class counts
    merged = [dataclasses.replace(r, split="train") if r.split == "val" else r for r in records]
    with pytest.raises(DataValidationError, match="do not have exactly 10"):
        validate_records(merged, cmap, expected_per_class={"train": 10, "val": 2})


def test_wrong_counts_and_missing_classes(fixture):
    _, records, cmap = fixture
    without_class_2 = [r for r in records if not (r.split == "test" and r.class_id == 2)]
    with pytest.raises(DataValidationError, match="missing class"):
        validate_records(without_class_2, cmap, require_all_classes_in=("test",))
    with pytest.raises(DataValidationError, match="expected 7"):
        validate_records(records, cmap, expected_counts={"test": 7})


def test_group_spanning_splits_is_rejected(fixture):
    _, records, cmap = fixture
    grouped = [
        dataclasses.replace(r, group_id="coil-1") if i in (0, len(records) - 1) else r
        for i, r in enumerate(records)
    ]
    with pytest.raises(DataValidationError, match="spans splits"):
        validate_records(grouped, cmap)


def test_file_verification(fixture):
    tmp, records, _ = fixture
    root = tmp / "raw"
    verify_files(records, root)
    (root / records[0].relative_path).write_bytes(b"changed")
    with pytest.raises(DataValidationError, match="SHA-256"):
        verify_files(records, root)
    (root / records[1].relative_path).unlink()
    with pytest.raises(DataValidationError, match="missing file"):
        verify_files(records[1:2], root)


def test_symlink_escape_is_rejected(fixture):
    tmp, records, _ = fixture
    root = tmp / "raw"
    outside = tmp / "outside.png"
    outside.write_bytes((root / records[2].relative_path).read_bytes())
    link = root / records[2].relative_path
    link.unlink()
    os.symlink(outside, link)
    with pytest.raises(DataValidationError, match="outside the data root"):
        verify_files(records[2:3], root)


def test_class_map_round_trip_and_tamper_detection(tmp_path):
    cmap = ClassMap("demo", ("a", "b", "c"), {"a": {"code": "A"}})
    write_class_map(cmap, tmp_path / "cm.json")
    loaded = load_class_map(tmp_path / "cm.json")
    assert loaded.names == cmap.names and loaded.sha256() == cmap.sha256()
    text = (tmp_path / "cm.json").read_text().replace('"class_name": "b"', '"class_name": "z"')
    (tmp_path / "cm.json").write_text(text)
    with pytest.raises(DataValidationError, match="class_map_sha256"):
        load_class_map(tmp_path / "cm.json")
