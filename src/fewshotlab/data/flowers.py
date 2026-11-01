"""Oxford Flowers102 adapter.

Files come only from the official source, fetched through torchvision's loader when
``--allow-download`` is given, or from a user-supplied folder holding the three official files.
Every file is checked against the MD5 table recorded in the installed torchvision. The manifest
is built from the official ``setid.mat`` / ``imagelabels.mat``. Raw MAT labels are 1..102 and are
shifted to 0..101 exactly once, and the result is cross-checked against torchvision's own
(already zero-based) labels. No substitute source is ever used.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from fewshotlab import __version__
from fewshotlab.data.manifest import (
    ClassMap,
    ImageRecord,
    split_counts,
    validate_records,
    write_class_map,
    write_manifest,
)
from fewshotlab.data.registry import (
    class_map_path,
    manifest_path,
    prepare_record_path,
    read_prepare_record,
    write_prepare_record,
)
from fewshotlab.errors import DatasetUnavailableError, DataValidationError
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import utc_now
from fewshotlab.paths import ProjectPaths, portable_path
from fewshotlab.status import DataStatus

DATASET = "flowers102"
NUM_CLASSES = 102
EXPECTED_COUNTS = {"train": 1020, "val": 1020, "test": 6149}
EXPECTED_PER_CLASS = {"train": 10, "val": 10}
SPLIT_KEYS = {"train": "trnid", "val": "valid", "test": "tstid"}
BASE_FOLDER = "flowers-102"  # torchvision's folder under its root
OFFICIAL_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
TERMS_NOTE = (
    "Usage terms not verified: the official page was unreachable from this network on "
    "2026-11-01. FewShotLab never redistributes the images; check the official page before "
    "any redistribution."
)
RESUME_COMMAND = (
    "python -m fewshotlab data prepare --dataset flowers102 --allow-download "
    "[--mirror internet-archive] (or --source DIR containing 102flowers.tgz, imagelabels.mat "
    "and setid.mat)"
)
#: Internet Archive captures of the three official Oxford URLs (pinned timestamps). Their bytes
#: are accepted only when the MD5 equals the value recorded in torchvision.
INTERNET_ARCHIVE_SNAPSHOTS = {
    "setid": "https://web.archive.org/web/20141102073126id_/"
    "http://www.robots.ox.ac.uk:80/~vgg/data/flowers/102/setid.mat",
    "label": "https://web.archive.org/web/20141028182747id_/"
    "http://www.robots.ox.ac.uk:80/~vgg/data/flowers/102/imagelabels.mat",
    "image": "https://web.archive.org/web/20201122103345id_/"
    "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
}
MIRRORS = ("internet-archive",)
DISPLAY_NAME_SOURCE = (
    "cat_to_name.json in the user-supplied Udacity/Kaggle copy (secondary source; display only)"
)
MAX_DOWNLOAD_BYTES = 600 * 1024**2


def official_files() -> dict[str, Any]:
    """URL prefix and MD5 table exactly as recorded in the installed torchvision."""
    import torchvision
    from torchvision.datasets import Flowers102

    return {
        "loader": f"torchvision.datasets.Flowers102 (torchvision {torchvision.__version__})",
        "url_prefix": Flowers102._download_url_prefix,
        "files": {
            key: {"filename": name, "md5": md5}
            for key, (name, md5) in Flowers102._file_dict.items()
        },
    }


def class_map() -> ClassMap:
    names = tuple(f"label_{i + 1:03d}" for i in range(NUM_CLASSES))
    return ClassMap(
        DATASET,
        names,
        notes=(
            "class_id = official MAT label - 1. Names label_001..label_102 keep the official "
            "1-based label number. Human-readable flower names are optional display metadata "
            "from a secondary source and are never used as identity."
        ),
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_complete(base: Path, official: dict[str, Any]) -> bool:
    return (
        all((base / spec["filename"]).is_file() for spec in official["files"].values())
        and (base / "jpg").is_dir()
    )


def _import_from_source(
    source: Path, base: Path, official: dict[str, Any], *, move: bool = False
) -> None:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise DataValidationError(f"--source {source} must be a folder with the official files")
    for spec in official["files"].values():
        candidate = source / spec["filename"]
        if not candidate.is_file():
            raise DataValidationError(f"--source is missing {spec['filename']}")
        if _md5(candidate) != spec["md5"]:
            raise DataValidationError(
                f"{candidate} does not match torchvision's recorded MD5 {spec['md5']}; "
                "it is not the original official file"
            )
    base.mkdir(parents=True, exist_ok=True)
    for spec in official["files"].values():
        target = base / spec["filename"]
        if not target.exists():
            if move:
                shutil.move(source / spec["filename"], target)
            else:
                shutil.copy2(source / spec["filename"], target)
    if not (base / "jpg").is_dir():
        archive = base / official["files"]["image"]["filename"]
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(base, filter="data")


def _stream_download(url: str, target: Path, *, timeout_s: float) -> dict[str, Any]:
    """Stream ``url`` to ``target`` (via a .part file); return size, MD5 and SHA-256."""
    request = urllib.request.Request(
        url, headers={"User-Agent": f"fewshotlab/{__version__} (dataset acquisition)"}
    )
    part = target.with_name(target.name + ".part")
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=timeout_s) as response, open(part, "wb") as out:
        while chunk := response.read(1 << 20):
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise DataValidationError(f"{url} exceeds {MAX_DOWNLOAD_BYTES} bytes; aborted")
            out.write(chunk)
            md5.update(chunk)
            sha.update(chunk)
    part.rename(target)
    return {
        "url": url,
        "bytes": size,
        "md5": md5.hexdigest(),
        "sha256": sha.hexdigest(),
        "downloaded_utc": utc_now(),
    }


def _fetch_from_internet_archive(
    root: Path, official: dict[str, Any], *, timeout_s: float
) -> tuple[Path, list[dict[str, Any]]]:
    """Download the archived official files into a staging folder; keep only MD5 matches."""
    staging = root / "download-internet-archive"
    staging.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = []
    for key in ("setid", "label", "image"):
        spec = official["files"][key]
        target = staging / spec["filename"]
        if target.is_file() and _md5(target) == spec["md5"]:
            log.append({"file": spec["filename"], "result": "already staged (MD5 verified)"})
            continue
        url = INTERNET_ARCHIVE_SNAPSHOTS[key]
        try:
            meta = _stream_download(url, target, timeout_s=timeout_s)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DatasetUnavailableError(
                f"Internet Archive download of {spec['filename']} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if meta["md5"] != spec["md5"]:
            rejected = target.with_name(target.name + ".rejected")
            target.rename(rejected)
            raise DataValidationError(
                f"{spec['filename']} from {url} has MD5 {meta['md5']}, not the official "
                f"{spec['md5']}; rejected (kept as {rejected.name} for inspection)"
            )
        log.append(
            {
                "file": spec["filename"],
                **meta,
                "result": "downloaded; MD5 matches torchvision's record",
            }
        )
    return staging, log


def cross_check_copy(copy_dir: Path, records: list[ImageRecord]) -> dict[str, Any]:
    """Compare a user-supplied re-packaged copy (e.g. Kaggle) with the verified official files.

    The copy is only read. Folder names that are integers are treated as official 1-based
    labels, and that assumption is itself checked against the official labels.
    """
    copy_dir = Path(copy_dir).expanduser().resolve()
    if not copy_dir.is_dir():
        raise DataValidationError(f"--cross-check {copy_dir} is not a folder")
    official = {Path(r.relative_path).name: r for r in records}
    files = sorted(
        p
        for p in copy_dir.rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in (".jpg", ".jpeg")
    )
    identical, different, foreign = [], [], []
    agree = 0
    disagree: list[str] = []
    unlabeled = 0
    for path in files:
        record = official.get(path.name)
        if record is None:
            foreign.append(path.name)
            continue
        (identical if sha256_file(path) == record.sha256 else different).append(path.name)
        if path.parent.name.isdigit():
            if int(path.parent.name) - 1 == record.class_id:
                agree += 1
            else:
                disagree.append(f"{path.relative_to(copy_dir)} vs official {record.class_name}")
        else:
            unlabeled += 1
    names = {p.name for p in files}
    layout = {
        sub.name: sum(1 for p in sub.rglob("*") if p.is_file())
        for sub in sorted(copy_dir.iterdir())
        if sub.is_dir()
    }
    return {
        "path": portable_path(copy_dir),
        "layout_file_counts": layout,
        "image_files": len(files),
        "byte_identical_to_official": len(identical),
        "byte_different_from_official": len(different),
        "not_in_official_set": len(foreign),
        "official_images_missing_from_copy": sum(1 for n in official if n not in names),
        "duplicate_names_in_copy": len(files) - len(names),
        "folder_labels_agree": agree,
        "folder_labels_disagree": len(disagree),
        "unlabeled_files": unlabeled,
        "examples": {
            "different": different[:10],
            "foreign": foreign[:10],
            "label_disagreements": disagree[:10],
        },
        "use": "verification only; the manifest and split come from the official files",
    }


def display_names_from_copy(copy_dir: Path, check: dict[str, Any]) -> dict[str, str] | None:
    """Optional display names from a copy's cat_to_name.json, used only if every folder label
    agreed with the official labels. Names are metadata, never identity."""
    path = Path(copy_dir) / "cat_to_name.json"
    if not path.is_file() or check["folder_labels_disagree"] or not check["folder_labels_agree"]:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or sorted(data, key=lambda k: int(k) if k.isdigit() else -1) != [
        str(i) for i in range(1, NUM_CLASSES + 1)
    ]:
        return None
    names = {}
    for key, value in data.items():
        if not isinstance(value, str) or not 0 < len(value) <= 60 or not value.isprintable():
            return None
        names[f"label_{int(key):03d}"] = value
    return names


def _download_with_retries(
    root: Path, *, attempts: int, timeout_s: float, backoff_s: float
) -> list[dict[str, Any]]:
    from torchvision.datasets import Flowers102

    log: list[dict[str, Any]] = []
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        for attempt in range(1, attempts + 1):
            started = utc_now()
            try:
                Flowers102(str(root), split="train", download=True)
            except Exception as exc:  # recorded in the prepare record, never hidden
                log.append(
                    {
                        "attempt": attempt,
                        "started_utc": started,
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
                )
                if attempt < attempts:
                    time.sleep(backoff_s * attempt)
                continue
            log.append({"attempt": attempt, "started_utc": started, "result": "downloaded"})
            break
    finally:
        socket.setdefaulttimeout(previous)
    return log


def build_records(base: Path, data_root: Path, cmap: ClassMap) -> list[ImageRecord]:
    """Manifest rows in official ``setid.mat`` order, with the single label shift."""
    from scipy.io import loadmat

    setid = loadmat(base / "setid.mat", squeeze_me=True)
    raw_labels = np.asarray(loadmat(base / "imagelabels.mat", squeeze_me=True)["labels"])
    raw_labels = raw_labels.astype(np.int64).ravel()
    if raw_labels.min() != 1 or raw_labels.max() != cmap.num_classes:
        raise DataValidationError(
            f"imagelabels.mat labels must span 1..{cmap.num_classes}; "
            f"found {raw_labels.min()}..{raw_labels.max()}"
        )
    zero_based = raw_labels - 1  # the one and only label shift
    label_of = {index + 1: int(label) for index, label in enumerate(zero_based)}
    seen: dict[int, str] = {}
    records: list[ImageRecord] = []
    for split, key in SPLIT_KEYS.items():
        for image_number in np.asarray(setid[key]).astype(np.int64).ravel().tolist():
            if image_number in seen:
                raise DataValidationError(
                    f"image {image_number} is listed in both {seen[image_number]} and {split}"
                )
            seen[image_number] = split
            if image_number not in label_of:
                raise DataValidationError(f"setid.mat lists image {image_number} without a label")
            relative = f"{DATASET}/{BASE_FOLDER}/jpg/image_{image_number:05d}.jpg"
            path = data_root / relative
            if not path.is_file():
                raise DataValidationError(f"missing image file {relative}")
            class_id = label_of[image_number]
            records.append(
                ImageRecord(
                    image_id=f"{DATASET}/image_{image_number:05d}",
                    relative_path=relative,
                    class_id=class_id,
                    class_name=cmap.names[class_id],
                    split=split,
                    sha256=sha256_file(path),
                )
            )
    return records


def cross_check_with_torchvision(root: Path, records: list[ImageRecord]) -> None:
    """Our labels must equal torchvision's zero-based labels, image by image, per split."""
    from torchvision.datasets import Flowers102

    for split in SPLIT_KEYS:
        loader = Flowers102(str(root), split=split, download=False)
        theirs = [
            (Path(p).name, int(label))
            for p, label in zip(loader._image_files, loader._labels, strict=True)
        ]
        ours = [(Path(r.relative_path).name, r.class_id) for r in records if r.split == split]
        if theirs != ours:
            raise DataValidationError(
                f"{split}: manifest labels disagree with torchvision's Flowers102 loader"
            )


def prepare(
    paths: ProjectPaths,
    *,
    source: Path | None = None,
    allow_download: bool = False,
    mirror: str | None = None,
    cross_check: Path | None = None,
    attempts: int = 3,
    timeout_s: float = 30.0,
    backoff_s: float = 5.0,
) -> dict[str, Any]:
    if mirror is not None and mirror not in MIRRORS:
        raise DataValidationError(f"unknown mirror {mirror!r}; choose from {list(MIRRORS)}")
    root = paths.data_root / DATASET
    base = root / BASE_FOLDER
    official = official_files()
    previous = read_prepare_record(paths, DATASET) or {}
    record: dict[str, Any] = {
        "dataset": DATASET,
        "attempted_utc": utc_now(),
        "source": {
            "official_page": OFFICIAL_PAGE,
            "download_url_prefix": official["url_prefix"],
            "loader": official["loader"],
            "expected_files": official["files"],
        },
        "terms": TERMS_NOTE,
    }
    if source is not None:
        _import_from_source(Path(source), base, official)
        record["source"]["mode"] = f"user-supplied folder {portable_path(source)}"
    elif _files_complete(base, official):
        record["source"]["mode"] = "already present"
        if previous.get("acquisition"):
            record["acquisition"] = previous["acquisition"]
    elif mirror == "internet-archive":
        if not allow_download:
            raise DatasetUnavailableError(
                "--mirror internet-archive downloads files; add --allow-download to permit it"
            )
        staging, log = _fetch_from_internet_archive(root, official, timeout_s=max(timeout_s, 60))
        _import_from_source(staging, base, official, move=True)
        record["source"]["mode"] = (
            "Internet Archive captures of the official Oxford URLs, MD5-verified against "
            "torchvision's recorded checksums"
        )
        record["acquisition"] = {
            "mirror": "internet-archive",
            "snapshots": INTERNET_ARCHIVE_SNAPSHOTS,
            "log": log,
        }
    else:
        if not allow_download:
            raise DatasetUnavailableError(
                f"Flowers102 files are not present under {base}. FewShotLab never downloads "
                f"implicitly: re-run with --allow-download to fetch them from "
                f"{official['url_prefix']}, or pass --source DIR with the three official files."
            )
        log = _download_with_retries(
            root, attempts=attempts, timeout_s=timeout_s, backoff_s=backoff_s
        )
        record["download_attempts"] = log
        if not _files_complete(base, official):
            last = log[-1].get("error", "unknown error") if log else "no attempt made"
            record.update(
                status=DataStatus.BLOCKED,
                summary=f"official source unreachable after {len(log)} attempt(s): {last}",
                resume_command=RESUME_COMMAND,
            )
            write_prepare_record(paths, DATASET, record)
            raise DatasetUnavailableError(
                f"{record['summary']}. Recorded as BLOCKED in "
                f"{prepare_record_path(paths, DATASET)}. To resume: {RESUME_COMMAND}"
            )
        record["source"]["mode"] = "downloaded through torchvision's official loader"
    files: dict[str, Any] = {}
    for key, spec in official["files"].items():
        path = base / spec["filename"]
        md5 = _md5(path)
        if md5 != spec["md5"]:
            raise DataValidationError(f"{path} MD5 {md5} does not match the official {spec['md5']}")
        files[key] = {
            "filename": spec["filename"],
            "bytes": path.stat().st_size,
            "md5": md5,
            "sha256": sha256_file(path),
        }
    cmap = class_map()
    records = build_records(base, paths.data_root, cmap)
    validate_records(
        records,
        cmap,
        expected_counts=EXPECTED_COUNTS,
        expected_per_class=EXPECTED_PER_CLASS,
        require_all_classes_in=("train", "val", "test"),
    )
    cross_check_with_torchvision(root, records)
    if cross_check is not None:
        check = cross_check_copy(Path(cross_check), records)
        record["cross_check"] = check
        names = display_names_from_copy(Path(cross_check), check)
        if names:
            cmap = ClassMap(
                DATASET,
                cmap.names,
                extra={
                    n: {
                        "display_name": names[n],
                        "display_name_source": DISPLAY_NAME_SOURCE,
                    }
                    for n in cmap.names
                },
                notes=cmap.notes,
            )
    elif previous.get("cross_check"):
        record["cross_check"] = previous["cross_check"]
    manifest_sha = write_manifest(records, manifest_path(paths, DATASET))
    write_class_map(cmap, class_map_path(paths, DATASET))
    counts = split_counts(records)
    record.update(
        status=DataStatus.PREPARED,
        summary=f"{len(records)} images; train/val/test = "
        f"{counts['train']}/{counts['val']}/{counts['test']}",
        prepared_utc=utc_now(),
        files=files,
        label_conversion=(
            "imagelabels.mat labels 1..102 minus 1, applied once; cross-checked against "
            "torchvision Flowers102 (zero-based) for every split"
        ),
        manifest={
            "path": manifest_path(paths, DATASET).name,
            "sha256": manifest_sha,
            "rows": len(records),
            "counts": counts,
        },
        class_map={"path": class_map_path(paths, DATASET).name, "sha256": cmap.sha256()},
    )
    record.pop("resume_command", None)
    write_prepare_record(paths, DATASET, record)
    return record
