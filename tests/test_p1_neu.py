"""P1: the NEU-CLS adapter accepts only the original layout and splits deterministically."""

from __future__ import annotations

import shutil
import urllib.error
import zipfile
from collections import Counter

import pytest

from data_fixtures import neu_tree, noise_image, zip_dir
from fewshotlab.data import neu
from fewshotlab.data.manifest import SPLITS, read_manifest
from fewshotlab.data.registry import manifest_path, read_prepare_record
from fewshotlab.errors import DatasetUnavailableError, DataValidationError
from fewshotlab.paths import get_paths

SMALL = {"images_per_class": 5, "split_per_class": (("train", 3), ("val", 1), ("test", 1))}


def _prepare(source, **overrides):
    return neu.prepare(get_paths(), source=source, **{**SMALL, **overrides})


def test_valid_archive_is_prepared(isolated_paths, tmp_path):
    archive = zip_dir(neu_tree(tmp_path / "src"), tmp_path / "NEU-CLS.zip")
    record = _prepare(archive)
    assert record["status"] == "PREPARED"
    assert record["archive"]["sha256"] and record["extraction"]["kind"] == "zip"
    rows = read_manifest(manifest_path(get_paths(), "neu_cls"), splits=SPLITS)
    assert len(rows) == 30
    per = Counter((r.class_id, r.split) for r in rows)
    for class_id in range(6):
        assert (per[(class_id, "train")], per[(class_id, "val")], per[(class_id, "test")]) == (
            3,
            1,
            1,
        )
    names = {r.class_id: r.class_name for r in rows}
    assert names == {
        0: "rolled-in_scale",
        1: "patches",
        2: "crazing",
        3: "pitted_surface",
        4: "inclusion",
        5: "scratches",
    }
    assert {r.image_id for r in rows if r.class_id == 2} == {f"neu_cls/Cr_{n}" for n in range(1, 6)}


def test_split_is_deterministic_across_installations(tmp_path, monkeypatch):
    digests = []
    for name in ("first", "second"):
        root = tmp_path / name
        monkeypatch.setenv("FEWSHOTLAB_DATA_ROOT", str(root / "raw"))
        monkeypatch.setenv("FEWSHOTLAB_MANIFESTS_DIR", str(root / "manifests"))
        archive = zip_dir(neu_tree(tmp_path / f"src-{name}"), tmp_path / f"{name}.zip")
        digests.append(_prepare(archive)["manifest"]["sha256"])
    assert digests[0] == digests[1]


def test_os_metadata_files_are_ignored_and_listed(isolated_paths, tmp_path):
    tree = neu_tree(tmp_path / "with-metadata")
    (tree / "Thumbs.db").write_bytes(b"windows thumbnail cache")
    (tree / ".DS_Store").write_bytes(b"finder")
    record = _prepare(zip_dir(tree, tmp_path / "meta.zip"))
    assert record["status"] == "PREPARED"
    assert sorted(record["ignored_os_metadata_files"]) == [
        "NEU-CLS/.DS_Store",
        "NEU-CLS/Thumbs.db",
    ]


def test_download_provenance_is_kept_for_revalidation(isolated_paths, monkeypatch, tmp_path):
    import hashlib

    archive = zip_dir(neu_tree(tmp_path / "drive"), tmp_path / "NEU-CLS.zip")
    payload = archive.read_bytes()

    def drive_file(url, dest_dir, timeout_s):
        part = dest_dir / "NEU-CLS.zip.part"
        part.write_bytes(payload)
        return part, {
            "url": url,
            "filename": "NEU-CLS.zip",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "content_type": "application/zip",
        }

    monkeypatch.setattr(neu, "_fetch", drive_file)
    first = neu.prepare(get_paths(), allow_download=True, **SMALL)
    assert first["archive"]["content_type"] == "application/zip"
    again = neu.prepare(get_paths(), **SMALL)  # already present: provenance is carried over
    assert again["source"]["mode"].startswith("already present")
    assert again["archive"]["content_type"] == "application/zip"
    assert again["manifest"]["sha256"] == first["manifest"]["sha256"]


def test_folder_source_is_accepted(isolated_paths, tmp_path):
    record = _prepare(neu_tree(tmp_path / "folder"))
    assert record["status"] == "PREPARED"
    assert "folder" in record["source"]["mode"]


def _rejected(tmp_path, tree, message):
    with pytest.raises(DataValidationError, match=message):
        _prepare(zip_dir(tree, tmp_path / "candidate.zip"))
    record = read_prepare_record(get_paths(), "neu_cls")
    assert record["status"] == "FAILED"  # the failure is recorded, never marked PREPARED
    assert record["summary"]


def test_neu_det_layout_is_rejected(isolated_paths, tmp_path):
    tree = tmp_path / "det"
    (tree / "IMAGES").mkdir(parents=True)
    (tree / "ANNOTATIONS").mkdir()
    noise_image(1, (200, 200), "L").save(tree / "IMAGES" / "crazing_1.jpg")
    (tree / "ANNOTATIONS" / "crazing_1.xml").write_text("<annotation/>")
    _rejected(tmp_path, tree, "NEU-DET")


def test_neu_det_naming_is_rejected(isolated_paths, tmp_path):
    tree = tmp_path / "det-names"
    tree.mkdir()
    noise_image(1, (200, 200), "L").save(tree / "crazing_1.bmp")
    _rejected(tmp_path, tree, "NEU-DET naming")


def test_neu_cls_64_sizes_are_rejected(isolated_paths, tmp_path):
    _rejected(tmp_path, neu_tree(tmp_path / "n64", size=(64, 64)), "NEU-CLS-64")


def test_neu_cls_64_classes_are_rejected(isolated_paths, tmp_path):
    tree = neu_tree(tmp_path / "n64-classes")
    noise_image(5, (64, 64), "L").save(tree / "GG_1.bmp")
    _rejected(tmp_path, tree, "NEU-CLS-64")


def test_colour_images_are_rejected(isolated_paths, tmp_path):
    _rejected(tmp_path, neu_tree(tmp_path / "rgb", mode="RGB"), "not grayscale")


def test_wrong_image_size_is_rejected(isolated_paths, tmp_path):
    _rejected(tmp_path, neu_tree(tmp_path / "small", size=(100, 100)), "100x100")


def test_missing_file_is_rejected(isolated_paths, tmp_path):
    tree = neu_tree(tmp_path / "missing")
    (tree / "Sc_5.bmp").unlink()
    _rejected(tmp_path, tree, "structure check failed")


def test_html_instead_of_archive_is_rejected(isolated_paths, tmp_path):
    fake = tmp_path / "NEU-CLS.zip"
    fake.write_text("<!DOCTYPE html><html><body>Google Drive virus scan warning</body></html>")
    with pytest.raises(DataValidationError, match="HTML page"):
        _prepare(fake)


def test_archive_path_traversal_is_rejected(isolated_paths, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.bmp", b"x")
        zf.writestr("NEU-CLS/RS_1.bmp", b"x")
    with pytest.raises(DataValidationError, match="unsafe archive member"):
        _prepare(archive)
    assert not (isolated_paths.data_root / "neu_cls" / "evil.bmp").exists()
    assert not (isolated_paths.data_root / "evil.bmp").exists()


def test_identical_images_under_two_labels_block_preparation(isolated_paths, tmp_path):
    tree = neu_tree(tmp_path / "conflict")
    shutil.copy(tree / "RS_1.bmp", tree / "Pa_1.bmp")
    with pytest.raises(DataValidationError, match="label conflict"):
        _prepare(tree)


def test_duplicate_clusters_stay_in_one_split():
    ids = {0: [f"x{i}" for i in range(10)]}
    clusters = {"x3": "x3", "x7": "x3"}
    per_class = (("train", 6), ("val", 2), ("test", 2))
    first = neu.assign_split(ids, clusters, seed=2026, per_class=per_class)
    assert first["x3"] == first["x7"]
    assert len(first) == 10
    assert first == neu.assign_split(ids, clusters, seed=2026, per_class=per_class)


def test_no_download_without_permission(isolated_paths):
    with pytest.raises(DatasetUnavailableError, match="--allow-download"):
        neu.prepare(get_paths())
    assert read_prepare_record(get_paths(), "neu_cls") is None


def test_occupied_raw_folder_is_never_overwritten(isolated_paths, tmp_path):
    archive = zip_dir(neu_tree(tmp_path / "one"), tmp_path / "one.zip")
    _prepare(archive)
    with pytest.raises(DataValidationError, match="never deletes"):
        _prepare(archive)


def test_drive_confirm_form_is_followed_only_on_the_drive_host():
    html = (
        '<html><form id="download-form" action="https://drive.usercontent.google.com/download" '
        'method="get"><input type="hidden" name="id" value="abc">'
        '<input type="hidden" name="export" value="download">'
        '<input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="u-1"></form></html>'
    )
    url = neu.parse_drive_confirm_form(html)
    assert url is not None and url.startswith("https://drive.usercontent.google.com/download?")
    assert "confirm=t" in url and "id=abc" in url
    other = html.replace(
        "https://drive.usercontent.google.com/download", "https://example.com/download"
    )
    assert neu.parse_drive_confirm_form(other) is None


def test_unreachable_drive_is_recorded_as_blocked(isolated_paths, monkeypatch):
    def unreachable(url, dest_dir, timeout_s):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(neu, "_fetch", unreachable)
    with pytest.raises(DatasetUnavailableError):
        neu.prepare(get_paths(), allow_download=True)
    record = read_prepare_record(get_paths(), "neu_cls")
    assert record["status"] == "BLOCKED"
    assert "resume_command" in record


def test_html_page_without_a_form_is_blocked(isolated_paths, monkeypatch):
    def html_page(url, dest_dir, timeout_s):
        part = dest_dir / "page.html.part"
        part.write_text("<!DOCTYPE html><html>quota exceeded</html>")
        return part, {
            "url": url,
            "filename": "page.html",
            "bytes": 40,
            "sha256": "x",
            "content_type": "text/html",
        }

    monkeypatch.setattr(neu, "_fetch", html_page)
    with pytest.raises(DatasetUnavailableError, match="manually"):
        neu.prepare(get_paths(), allow_download=True)
