"""P1: the Flowers102 manifest comes from official-format files with one label shift."""

from __future__ import annotations

import json
import shutil

import pytest
from torchvision.datasets import Flowers102

from data_fixtures import make_flowers_files, md5
from fewshotlab.data import flowers
from fewshotlab.data.manifest import SPLITS, load_class_map, read_manifest
from fewshotlab.data.registry import class_map_path, manifest_path, read_prepare_record
from fewshotlab.errors import DatasetUnavailableError, DataValidationError
from fewshotlab.paths import get_paths


@pytest.fixture
def small_flowers(monkeypatch):
    """A 3-class, 21-image official-format fixture (2/2/3 images per class and split)."""
    monkeypatch.setattr(flowers, "NUM_CLASSES", 3)
    monkeypatch.setattr(flowers, "EXPECTED_COUNTS", {"train": 6, "val": 6, "test": 9})
    monkeypatch.setattr(flowers, "EXPECTED_PER_CLASS", {"train": 2, "val": 2})

    def install(base):
        table = make_flowers_files(base)
        monkeypatch.setattr(Flowers102, "_file_dict", table)
        return table

    return install


def _base(isolated_paths):
    return isolated_paths.data_root / "flowers102" / "flowers-102"


def test_manifest_applies_the_label_shift_exactly_once(isolated_paths, small_flowers):
    base = _base(isolated_paths)
    small_flowers(base)
    record = flowers.prepare(get_paths())
    assert record["status"] == "PREPARED"
    assert record["source"]["mode"] == "already present"
    from scipy.io import loadmat

    raw = loadmat(base / "imagelabels.mat", squeeze_me=True)["labels"]
    rows = read_manifest(manifest_path(get_paths(), "flowers102"), splits=SPLITS)
    assert len(rows) == 21
    for r in rows:
        number = int(r.image_id.rsplit("_", 1)[1])
        assert r.class_id == int(raw[number - 1]) - 1
        assert r.class_name == f"label_{r.class_id + 1:03d}"
    assert load_class_map(class_map_path(get_paths(), "flowers102")).names == (
        "label_001",
        "label_002",
        "label_003",
    )


def test_user_supplied_official_files(isolated_paths, small_flowers, tmp_path):
    source = tmp_path / "official"
    small_flowers(source)
    shutil.rmtree(source / "jpg")  # only the three official files; images come from the tgz
    record = flowers.prepare(get_paths(), source=source)
    assert record["status"] == "PREPARED"
    assert (_base(isolated_paths) / "jpg" / "image_00001.jpg").is_file()


def test_zero_based_mat_labels_are_rejected(isolated_paths, small_flowers, monkeypatch):
    from scipy.io import loadmat, savemat

    base = _base(isolated_paths)
    table = small_flowers(base)
    savemat(base / "imagelabels.mat", {"labels": loadmat(base / "imagelabels.mat")["labels"] - 1})
    monkeypatch.setattr(
        Flowers102,
        "_file_dict",
        {**table, "label": ("imagelabels.mat", md5(base / "imagelabels.mat"))},
    )
    with pytest.raises(DataValidationError, match="must span 1..3"):
        flowers.prepare(get_paths())


def test_tampered_official_file_is_rejected(isolated_paths, small_flowers):
    base = _base(isolated_paths)
    small_flowers(base)
    (base / "setid.mat").write_bytes((base / "setid.mat").read_bytes() + b"\0")
    with pytest.raises(DataValidationError, match="MD5"):
        flowers.prepare(get_paths())


def test_no_download_without_permission(isolated_paths):
    with pytest.raises(DatasetUnavailableError, match="--allow-download"):
        flowers.prepare(get_paths())
    assert read_prepare_record(get_paths(), "flowers102") is None


def test_unreachable_official_source_is_recorded_as_blocked(isolated_paths, monkeypatch):
    monkeypatch.setattr(
        flowers,
        "_download_with_retries",
        lambda root, **kw: [{"attempt": 1, "error": "TimeoutError: timed out"}],
    )
    with pytest.raises(DatasetUnavailableError, match="BLOCKED"):
        flowers.prepare(get_paths(), allow_download=True)
    record = read_prepare_record(get_paths(), "flowers102")
    assert record["status"] == "BLOCKED"
    assert "resume_command" in record


def test_mirror_bytes_must_match_the_official_md5(
    isolated_paths, small_flowers, monkeypatch, tmp_path
):
    small_flowers(tmp_path / "unused")

    def wrong_bytes(url, target, *, timeout_s):
        target.write_bytes(b"not the official file")
        return {"url": url, "bytes": 21, "md5": md5(target), "sha256": "x"}

    monkeypatch.setattr(flowers, "_stream_download", wrong_bytes)
    with pytest.raises(DataValidationError, match="not the official"):
        flowers.prepare(get_paths(), allow_download=True, mirror="internet-archive")


def test_mirror_download_with_matching_md5_is_accepted(
    isolated_paths, small_flowers, monkeypatch, tmp_path
):
    official = tmp_path / "archive-copy"
    small_flowers(official)
    by_name = {
        "setid.mat": "setid.mat",
        "imagelabels.mat": "imagelabels.mat",
        "102flowers.tgz": "102flowers.tgz",
    }

    def archived(url, target, *, timeout_s):
        shutil.copy(official / by_name[target.name], target)
        return {"url": url, "bytes": target.stat().st_size, "md5": md5(target), "sha256": "x"}

    monkeypatch.setattr(flowers, "_stream_download", archived)
    with pytest.raises(DatasetUnavailableError, match="allow-download"):
        flowers.prepare(get_paths(), mirror="internet-archive")
    record = flowers.prepare(get_paths(), allow_download=True, mirror="internet-archive")
    assert record["status"] == "PREPARED"
    assert record["acquisition"]["mirror"] == "internet-archive"
    again = flowers.prepare(get_paths())  # files already present: provenance is carried over
    assert again["acquisition"]["mirror"] == "internet-archive"


def test_cross_check_of_a_repackaged_copy(isolated_paths, small_flowers, tmp_path):
    base = _base(isolated_paths)
    small_flowers(base)
    flowers.prepare(get_paths())
    rows = read_manifest(manifest_path(get_paths(), "flowers102"), splits=SPLITS)
    copy = tmp_path / "kaggle"
    for r in rows:  # Udacity-style layout: labelled train/ folders, flat unlabelled test/
        folder = copy / "test" if r.split == "test" else copy / "train" / str(r.class_id + 1)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(isolated_paths.data_root / r.relative_path, folder)
    (copy / "cat_to_name.json").write_text(json.dumps({"1": "rose", "2": "tulip", "3": "daisy"}))
    check = flowers.cross_check_copy(copy, rows)
    assert check["byte_identical_to_official"] == 21
    assert (check["folder_labels_agree"], check["folder_labels_disagree"]) == (12, 0)
    assert check["unlabeled_files"] == 9
    assert flowers.display_names_from_copy(copy, check) == {
        "label_001": "rose",
        "label_002": "tulip",
        "label_003": "daisy",
    }
    record = flowers.prepare(get_paths(), cross_check=copy)
    assert record["cross_check"]["byte_identical_to_official"] == 21
    cmap_json = json.loads(class_map_path(get_paths(), "flowers102").read_text())
    assert cmap_json["classes"][0]["display_name"] == "rose"
    # a mislabelled folder is reported and switches the display names off
    moved = next((copy / "train" / "1").iterdir())
    shutil.move(str(moved), copy / "train" / "2" / moved.name)
    check = flowers.cross_check_copy(copy, rows)
    assert check["folder_labels_disagree"] == 1
    assert flowers.display_names_from_copy(copy, check) is None
