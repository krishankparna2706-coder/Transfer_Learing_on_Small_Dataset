"""NEU-CLS adapter: the authors' image-only classification archive -> validated manifest.

Only the original NEU-CLS layout is accepted: six defect classes x 300 grayscale 200x200 BMP files
named ``<prefix>_<n>.bmp``. NEU-DET (detection annotations) and NEU-CLS-64 (64x64 crops) are
rejected explicitly, never reshaped into NEU-CLS. The project split (180/60/60 per class,
split_seed 2026) is drawn after an exact-duplicate check, and exact-duplicate clusters stay in one
split. A cluster that would change the counts, or identical images under two labels, is a
blocking discrepancy that needs a documented decision.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fewshotlab import __version__
from fewshotlab.data.archive import safe_extract, sniff_kind
from fewshotlab.data.imageio import ImageDecodeError, is_grayscale_content, open_image, pixel_sha256
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
    write_prepare_record,
)
from fewshotlab.errors import DatasetUnavailableError, DataValidationError
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import atomic_write_json, read_json, utc_now
from fewshotlab.paths import ProjectPaths, portable_path
from fewshotlab.seeds import NUMPY_BIT_GENERATOR, SEED_DERIVATION, numpy_rng
from fewshotlab.status import DataStatus

DATASET = "neu_cls"
SOURCE_PAGE = "https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm"
DRIVE_FILE_ID = "1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc"
DRIVE_DOWNLOAD_ENDPOINT = "https://drive.usercontent.google.com/download"
TERMS_NOTE = (
    "No explicit license is stated on the source page (checked 2026-11-01); the authors ask "
    "users to cite their works. FewShotLab never redistributes the images."
)
#: (prefix as printed on the source page, class name); list order defines class_id 0..5.
CLASSES: tuple[tuple[str, str], ...] = (
    ("RS", "rolled-in_scale"),
    ("Pa", "patches"),
    ("Cr", "crazing"),
    ("PS", "pitted_surface"),
    ("In", "inclusion"),
    ("Sc", "scratches"),
)
IMAGES_PER_CLASS = 300
IMAGE_SIZE = (200, 200)
SPLIT_PER_CLASS: tuple[tuple[str, int], ...] = (("train", 180), ("val", 60), ("test", 60))
SPLIT_SEED = 2026
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"}
NAME_PATTERN = re.compile(r"^([A-Za-z]{2})_(\d+)$")
NEU64_ONLY_PREFIXES = {"GG", "RP", "SP"}
DET_CLASS_WORDS = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
}
MAX_DOWNLOAD_BYTES = 1024**3
#: Operating-system metadata that archives often carry; ignored and listed in the record.
OS_METADATA_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
_PREFIX_TO_CLASS = {prefix.upper(): index for index, (prefix, _) in enumerate(CLASSES)}


def class_map() -> ClassMap:
    names = tuple(name for _, name in CLASSES)
    return ClassMap(
        DATASET,
        names,
        extra={name: {"code": prefix} for prefix, name in CLASSES},
        notes=(
            "Class order follows the source page listing (RS, Pa, Cr, PS, In, Sc); it is fixed "
            "explicitly, never inferred from folder order. There is no normal/good class."
        ),
    )


# --------------------------------------------------------------------------- download


class _FormParser(HTMLParser):
    """Collects the download form of a Google Drive "can't scan for viruses" page."""

    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.fields: dict[str, str] = {}
        self._in_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k: v or "" for k, v in attrs}
        if tag == "form" and "download" in values.get("action", ""):
            self.action = values["action"]
            self._in_form = True
        elif tag == "input" and self._in_form and values.get("type") == "hidden":
            self.fields[values.get("name", "")] = values.get("value", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._in_form = False


def parse_drive_confirm_form(html_text: str) -> str | None:
    """The confirmed download URL from a Drive interstitial, or None.

    Only forms that submit to ``drive.usercontent.google.com/download`` are followed.
    """
    parser = _FormParser()
    parser.feed(html_text)
    if not parser.action or parser.action.split("?")[0] != DRIVE_DOWNLOAD_ENDPOINT:
        return None
    if not parser.fields.get("id"):
        return None
    return f"{DRIVE_DOWNLOAD_ENDPOINT}?{urllib.parse.urlencode(parser.fields)}"


def _filename_from_headers(headers: Message, fallback: str) -> str:
    name = headers.get_filename() or fallback
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or fallback
    return name


def _fetch(url: str, dest_dir: Path, timeout_s: float) -> tuple[Path, dict[str, Any]]:
    request = urllib.request.Request(
        url, headers={"User-Agent": f"fewshotlab/{__version__} (dataset acquisition)"}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        name = _filename_from_headers(response.headers, f"neu_cls_{DRIVE_FILE_ID}.bin")
        part = dest_dir / f"{name}.part"
        digest = hashlib.sha256()
        size = 0
        with open(part, "wb") as out:
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise DataValidationError("download exceeds 1 GiB; refusing to continue")
                out.write(chunk)
                digest.update(chunk)
        meta = {
            "url": url,
            "filename": name,
            "bytes": size,
            "sha256": digest.hexdigest(),
            "content_type": response.headers.get("Content-Type", ""),
        }
    return part, meta


def download_from_drive(dest_dir: Path, *, timeout_s: float = 120.0) -> tuple[Path, dict[str, Any]]:
    """Fetch the authors' NEU-CLS file into the new, empty ``dest_dir``."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise DataValidationError(f"download folder {dest_dir} must be new and empty")
    dest_dir.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"id": DRIVE_FILE_ID, "export": "download"})
    url = f"{DRIVE_DOWNLOAD_ENDPOINT}?{query}"
    attempts: list[dict[str, Any]] = []
    for _hop in range(2):  # the file itself, or one confirmation page then the file
        try:
            part, meta = _fetch(url, dest_dir, timeout_s)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            attempts.append({"url": url, "error": f"{type(exc).__name__}: {exc}"[:500]})
            break
        if sniff_kind(part) == "html":
            confirm = parse_drive_confirm_form(part.read_text(encoding="utf-8", errors="replace"))
            part.unlink()
            attempts.append({"url": url, "result": "HTML page instead of the file"})
            if confirm is None:
                break
            url = confirm
            continue
        final = dest_dir / meta["filename"]
        part.rename(final)
        meta["attempts"] = attempts
        meta["downloaded_utc"] = utc_now()
        return final, meta
    raise DatasetUnavailableError(
        "could not download NEU-CLS from the authors' Google Drive link "
        f"(attempts: {attempts}). Download it manually from {SOURCE_PAGE} (the 'NEU-CLS' link) "
        "and re-run with --source PATH"
    )


# --------------------------------------------------------------------------- validation


def is_os_metadata(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        "__MACOSX" in relative.parts or path.name in OS_METADATA_NAMES or path.name.startswith("._")
    )


def scan_tree(
    root: Path,
    *,
    images_per_class: int = IMAGES_PER_CLASS,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[dict[int, list[tuple[int, Path]]], list[str]]:
    """Locate and validate the NEU-CLS images under ``root``; reject other variants.

    Returns the images per class and the OS metadata files that were ignored.
    """
    every = [p for p in sorted(root.rglob("*")) if p.is_file()]
    ignored = [p.relative_to(root).as_posix() for p in every if is_os_metadata(p, root)]
    files = [p for p in every if not is_os_metadata(p, root)]
    xml = [p for p in files if p.suffix.lower() == ".xml"]
    if xml:
        raise DataValidationError(
            f"found {len(xml)} XML annotation file(s): this looks like NEU-DET (detection), not "
            f"the image-only NEU-CLS archive; download 'NEU-CLS' from {SOURCE_PAGE}"
        )
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    if any(p.stem.lower().rstrip("_0123456789") in DET_CLASS_WORDS for p in images):
        raise DataValidationError(
            "images are named like 'crazing_1' (NEU-DET naming); FewShotLab needs NEU-CLS"
        )
    by_class: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    unexpected: list[str] = []
    for path in images:
        match = NAME_PATTERN.match(path.stem)
        prefix = match.group(1).upper() if match else ""
        if prefix in NEU64_ONLY_PREFIXES:
            raise DataValidationError(
                f"found {path.name}: classes such as GG/RP/SP exist only in NEU-CLS-64, which is "
                "not a substitute for NEU-CLS"
            )
        if not match or prefix not in _PREFIX_TO_CLASS:
            unexpected.append(path.name)
            continue
        by_class[_PREFIX_TO_CLASS[prefix]].append((int(match.group(2)), path))
    others = [p.name for p in files if p.suffix.lower() not in IMAGE_SUFFIXES]
    if unexpected or others:
        raise DataValidationError(
            f"unexpected file(s) for NEU-CLS: {(unexpected + others)[:8]} (NEU-CLS contains only "
            "<prefix>_<n>.bmp images)"
        )
    non_bmp = [
        p.name for entries in by_class.values() for _, p in entries if p.suffix.lower() != ".bmp"
    ]
    if non_bmp:
        raise DataValidationError(
            f"NEU-CLS images are BMP files; found {non_bmp[:5]} (a re-encoded copy is not the "
            "original archive)"
        )
    problems: list[str] = []
    for class_id, (prefix, name) in enumerate(CLASSES):
        entries = sorted(by_class.get(class_id, []))
        numbers = [n for n, _ in entries]
        if len(entries) != images_per_class or sorted(set(numbers)) != list(
            range(1, images_per_class + 1)
        ):
            problems.append(
                f"class {prefix} ({name}): {len(entries)} files with numbers "
                f"{numbers[:3]}...; expected exactly {prefix}_1..{prefix}_{images_per_class}"
            )
        by_class[class_id] = entries
    if problems:
        raise DataValidationError("NEU-CLS structure check failed:\n  - " + "\n  - ".join(problems))
    for entries in by_class.values():
        for _, path in entries:
            try:
                image = open_image(path)
            except ImageDecodeError as exc:
                raise DataValidationError(str(exc)) from exc
            if image.size == (64, 64):
                raise DataValidationError(f"{path.name} is 64x64: this is NEU-CLS-64, not NEU-CLS")
            if image.size != image_size:
                raise DataValidationError(
                    f"{path.name} is {image.size[0]}x{image.size[1]}; NEU-CLS images are "
                    f"{image_size[0]}x{image_size[1]}"
                )
            if not is_grayscale_content(image):
                raise DataValidationError(f"{path.name} is not grayscale (channels differ)")
    return dict(by_class), ignored


def assign_split(
    ids_by_class: dict[int, list[str]],
    clusters: dict[str, str],
    *,
    seed: int = SPLIT_SEED,
    per_class: tuple[tuple[str, int], ...] = SPLIT_PER_CLASS,
) -> dict[str, str]:
    """Per class: sorted IDs -> cluster units -> seeded permutation -> cumulative split bounds.

    A cluster (exact-duplicate group) always lands in a single split.
    """
    assignment: dict[str, str] = {}
    bounds: list[tuple[str, int]] = []
    running = 0
    for split, count in per_class:
        running += count
        bounds.append((split, running))
    for class_id in sorted(ids_by_class):
        units: dict[str, list[str]] = defaultdict(list)
        for image_id in sorted(ids_by_class[class_id]):
            units[clusters.get(image_id, image_id)].append(image_id)
        keys = sorted(units)
        order = numpy_rng(DATASET, "split", seed, class_id).permutation(len(keys))
        position = 0
        for index in order:
            members = units[keys[int(index)]]
            split = next(s for s, bound in bounds if position < bound)
            for member in members:
                assignment[member] = split
            position += len(members)
    return assignment


def prepare(
    paths: ProjectPaths,
    *,
    source: Path | None = None,
    allow_download: bool = False,
    images_per_class: int = IMAGES_PER_CLASS,
    image_size: tuple[int, int] = IMAGE_SIZE,
    split_per_class: tuple[tuple[str, int], ...] = SPLIT_PER_CLASS,
) -> dict[str, Any]:
    root = paths.data_root / DATASET
    download_dir, extract_dir = root / "download", root / "extracted"
    acquisition_path = root / "acquisition.json"
    record: dict[str, Any] = {
        "dataset": DATASET,
        "attempted_utc": utc_now(),
        "source": {
            "page": SOURCE_PAGE,
            "variant": "NEU-CLS (image-only classification archive)",
            "google_drive_file_id": DRIVE_FILE_ID,
        },
        "terms": TERMS_NOTE,
    }
    occupied = any(d.is_dir() and any(d.iterdir()) for d in (download_dir, extract_dir))
    if source is not None:
        if occupied:
            raise DataValidationError(
                f"{root} already holds NEU files; move or delete that folder yourself before "
                "importing a different source (FewShotLab never deletes raw data)"
            )
        src = Path(source).expanduser().resolve()
        if src.is_dir():
            if any(p.is_symlink() for p in src.rglob("*")):
                raise DataValidationError(f"{src} contains symbolic links; refusing to import")
            shutil.copytree(src, extract_dir)
            record["source"]["mode"] = f"user-supplied folder {portable_path(src)}"
        elif src.is_file():
            download_dir.mkdir(parents=True, exist_ok=True)
            archive = download_dir / src.name
            shutil.copy2(src, archive)
            record["archive"] = {
                "filename": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            }
            record["extraction"] = safe_extract(archive, extract_dir)
            record["source"]["mode"] = f"user-supplied archive {portable_path(src)}"
        else:
            raise DataValidationError(f"--source {src} does not exist")
    elif extract_dir.is_dir() and any(extract_dir.iterdir()):
        record["source"]["mode"] = "already present (re-validated)"
        archives = (
            sorted(p for p in download_dir.glob("*") if p.is_file())
            if download_dir.is_dir()
            else []
        )
        if archives:
            digest = sha256_file(archives[0])
            if acquisition_path.is_file():
                acquired = read_json(acquisition_path)
                if acquired.get("sha256") != digest:
                    raise DataValidationError(
                        f"{archives[0].name} no longer matches its acquisition record"
                    )
                record["archive"] = acquired
            else:
                record["archive"] = {
                    "filename": archives[0].name,
                    "bytes": archives[0].stat().st_size,
                    "sha256": digest,
                    "note": "acquisition metadata was not retained",
                }
    else:
        if not allow_download:
            raise DatasetUnavailableError(
                f"NEU-CLS is not present under {root}. FewShotLab never downloads implicitly: "
                f"re-run with --allow-download to fetch it from the authors' Google Drive link "
                f"listed on {SOURCE_PAGE}, or download NEU-CLS yourself and pass --source PATH."
            )
        try:
            archive, meta = download_from_drive(download_dir)
        except DatasetUnavailableError as exc:
            record.update(
                status=DataStatus.BLOCKED,
                summary=str(exc)[:500],
                resume_command="python -m fewshotlab data prepare --dataset neu_cls "
                "--source PATH  (after downloading NEU-CLS manually)",
            )
            write_prepare_record(paths, DATASET, record)
            raise
        meta.update(page=SOURCE_PAGE, google_drive_file_id=DRIVE_FILE_ID)
        record["archive"] = meta
        record["extraction"] = safe_extract(archive, extract_dir)
        meta["extraction"] = record["extraction"]
        atomic_write_json(acquisition_path, meta)
        record["source"]["mode"] = "downloaded from the authors' Google Drive link"
    try:
        return _validate_and_write(
            paths,
            record,
            extract_dir,
            images_per_class=images_per_class,
            image_size=image_size,
            split_per_class=split_per_class,
        )
    except DataValidationError as exc:
        record.update(status=DataStatus.FAILED, summary=str(exc)[:1000])
        write_prepare_record(paths, DATASET, record)
        raise


def _validate_and_write(
    paths: ProjectPaths,
    record: dict[str, Any],
    extract_dir: Path,
    *,
    images_per_class: int,
    image_size: tuple[int, int],
    split_per_class: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    by_class, ignored = scan_tree(
        extract_dir, images_per_class=images_per_class, image_size=image_size
    )
    record["ignored_os_metadata_files"] = ignored
    cmap = class_map()
    canonical = {index: prefix for index, (prefix, _) in enumerate(CLASSES)}
    pixel_hash: dict[str, str] = {}
    entries: list[tuple[str, int, Path]] = []
    for class_id, items in by_class.items():
        for number, path in items:
            image_id = f"{DATASET}/{canonical[class_id]}_{number}"
            entries.append((image_id, class_id, path))
            pixel_hash[image_id] = pixel_sha256(open_image(path))
    groups: dict[str, list[str]] = defaultdict(list)
    for image_id, digest in pixel_hash.items():
        groups[digest].append(image_id)
    class_of = {image_id: class_id for image_id, class_id, _ in entries}
    clusters: dict[str, str] = {}
    duplicate_groups = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        labels = sorted({class_of[m] for m in members})
        if len(labels) > 1:
            raise DataValidationError(
                f"label conflict: identical pixels under different classes {members}; this needs "
                "a documented dataset decision before any split is drawn"
            )
        duplicate_groups.append(members)
        for member in members:
            clusters[member] = members[0]
    ids_by_class: dict[int, list[str]] = defaultdict(list)
    for image_id, class_id, _ in entries:
        ids_by_class[class_id].append(image_id)
    assignment = assign_split(ids_by_class, clusters, seed=SPLIT_SEED, per_class=split_per_class)
    records = [
        ImageRecord(
            image_id=image_id,
            relative_path=path.relative_to(paths.data_root).as_posix(),
            class_id=class_id,
            class_name=cmap.names[class_id],
            split=assignment[image_id],
            sha256=sha256_file(path),
        )
        for image_id, class_id, path in sorted(entries)
    ]
    expected_per_class = dict(split_per_class)
    try:
        validate_records(
            records,
            cmap,
            expected_per_class=expected_per_class,
            require_all_classes_in=("train", "val", "test"),
        )
    except DataValidationError as exc:
        raise DataValidationError(
            f"{exc}\nKeeping exact-duplicate clusters together changed the protocol counts; this "
            "is a blocking discrepancy that needs a documented protocol revision."
        ) from exc
    manifest_sha = write_manifest(records, manifest_path(paths, DATASET))
    write_class_map(cmap, class_map_path(paths, DATASET))
    counts = split_counts(records)
    record.update(
        status=DataStatus.PREPARED,
        summary=f"{len(records)} images; train/val/test = "
        f"{counts['train']}/{counts['val']}/{counts['test']}",
        prepared_utc=utc_now(),
        split={
            "seed": SPLIT_SEED,
            "per_class": expected_per_class,
            "procedure": "per class: image_id ascending -> exact-duplicate clusters as units -> "
            "permutation -> cumulative bounds (train, val, test)",
            "rng": {
                "bit_generator": NUMPY_BIT_GENERATOR,
                "seed_labels": [DATASET, "split", SPLIT_SEED, "<class_id>"],
                "derivation": SEED_DERIVATION,
            },
            "exact_duplicate_clusters": duplicate_groups,
            "group_metadata": "none available from the source; image-level split",
        },
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


def record_path(paths: ProjectPaths) -> Path:
    return prepare_record_path(paths, DATASET)
