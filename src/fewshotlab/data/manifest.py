"""Immutable image manifests (one CSV row per image), class maps and their validation.

A manifest is written canonically (rows sorted by ``image_id``, ``\\n`` line endings), so the
same content always has the same SHA-256. Readers must name the splits they want; training code
asks only for ``train``/``val``.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from fewshotlab.errors import DataValidationError
from fewshotlab.hashing import sha256_bytes, sha256_file, sha256_json
from fewshotlab.io_utils import atomic_write_bytes, atomic_write_json, read_json

MANIFEST_COLUMNS = (
    "image_id",
    "relative_path",
    "class_id",
    "class_name",
    "split",
    "sha256",
    "group_id",
)
SPLITS = ("train", "val", "test")
_HEX = set("0123456789abcdef")


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str  # POSIX path relative to the data root
    class_id: int
    class_name: str
    split: str
    sha256: str
    group_id: str = ""  # empty when no acquisition group is known


@dataclass(frozen=True)
class ClassMap:
    """Integer class IDs 0..C-1 and their stable names (the identity used by every model)."""

    dataset: str
    names: tuple[str, ...]
    extra: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # per-name metadata
    notes: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def sha256(self) -> str:
        return sha256_json(
            {"dataset": self.dataset, "classes": [[i, n] for i, n in enumerate(self.names)]}
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "class_map_sha256": self.sha256(),
            "classes": [
                {"class_id": i, "class_name": n, **dict(self.extra.get(n, {}))}
                for i, n in enumerate(self.names)
            ],
            "notes": self.notes,
        }


def write_class_map(class_map: ClassMap, path: str | Path) -> Path:
    return atomic_write_json(path, class_map.to_json())


def load_class_map(path: str | Path) -> ClassMap:
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"class map not found: {path}")
    data = read_json(path)
    classes = data.get("classes", [])
    ids = [c.get("class_id") for c in classes]
    if ids != list(range(len(classes))):
        raise DataValidationError(f"{path}: class IDs must be 0..C-1 in order")
    names = tuple(c["class_name"] for c in classes)
    if len(set(names)) != len(names):
        raise DataValidationError(f"{path}: class names must be unique")
    extra = {
        c["class_name"]: {k: v for k, v in c.items() if k not in ("class_id", "class_name")}
        for c in classes
    }
    cmap = ClassMap(data["dataset"], names, extra, data.get("notes", ""))
    if data.get("class_map_sha256") != cmap.sha256():
        raise DataValidationError(f"{path}: stored class_map_sha256 does not match its contents")
    return cmap


def manifest_bytes(records: Iterable[ImageRecord]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for r in sorted(records, key=lambda rec: rec.image_id):
        writer.writerow(
            [r.image_id, r.relative_path, r.class_id, r.class_name, r.split, r.sha256, r.group_id]
        )
    return buffer.getvalue().encode("utf-8")


def write_manifest(records: Iterable[ImageRecord], path: str | Path) -> str:
    """Write the canonical manifest atomically and return its SHA-256."""
    data = manifest_bytes(records)
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def read_manifest(path: str | Path, *, splits: Iterable[str]) -> list[ImageRecord]:
    """Rows of the named splits only. Callers must name the splits explicitly."""
    wanted = tuple(splits)
    unknown = sorted(set(wanted) - set(SPLITS))
    if not wanted or unknown:
        raise DataValidationError(f"choose splits from {list(SPLITS)}; got {list(wanted)}")
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"manifest not found: {path}")
    out: list[ImageRecord] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise DataValidationError(f"{path}: columns must be {list(MANIFEST_COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            try:
                class_id = int(row["class_id"])
            except ValueError as exc:
                raise DataValidationError(f"{path}:{line}: class_id is not an integer") from exc
            if row["split"] not in SPLITS:
                raise DataValidationError(f"{path}:{line}: unknown split {row['split']!r}")
            if row["split"] in wanted:
                out.append(
                    ImageRecord(
                        row["image_id"],
                        row["relative_path"],
                        class_id,
                        row["class_name"],
                        row["split"],
                        row["sha256"],
                        row["group_id"] or "",
                    )
                )
    return out


def is_safe_relative_path(value: str) -> bool:
    """POSIX, relative, canonical, and without ``..`` components."""
    if not value or "\\" in value or "\x00" in value:
        return False
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        return False
    return str(pure) == value


def split_counts(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts = Counter(r.split for r in records)
    return {s: counts.get(s, 0) for s in SPLITS}


def per_class_counts(records: Sequence[ImageRecord], split: str) -> dict[int, int]:
    counts = Counter(r.class_id for r in records if r.split == split)
    return dict(sorted(counts.items()))


def validate_records(
    records: Sequence[ImageRecord],
    class_map: ClassMap,
    *,
    expected_counts: Mapping[str, int] | None = None,
    expected_per_class: Mapping[str, int] | None = None,
    require_all_classes_in: Iterable[str] = (),
) -> None:
    """Raise :class:`DataValidationError` listing every structural problem found."""
    problems: list[str] = []
    for value, count in Counter(r.image_id for r in records).items():
        if count > 1:
            problems.append(f"duplicate image_id {value!r} ({count} rows)")
    for value, count in Counter(r.relative_path for r in records).items():
        if count > 1:
            problems.append(f"duplicate relative_path {value!r} ({count} rows)")
    groups: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if not r.image_id or any(ch in r.image_id for ch in ",\n\r"):
            problems.append(f"invalid image_id {r.image_id!r}")
        if not is_safe_relative_path(r.relative_path):
            problems.append(f"{r.image_id}: unsafe relative_path {r.relative_path!r}")
        if not 0 <= r.class_id < class_map.num_classes:
            problems.append(f"{r.image_id}: unknown class_id {r.class_id}")
        elif class_map.names[r.class_id] != r.class_name:
            problems.append(
                f"{r.image_id}: class_name {r.class_name!r} does not match the class map "
                f"({class_map.names[r.class_id]!r})"
            )
        if r.split not in SPLITS:
            problems.append(f"{r.image_id}: unknown split {r.split!r}")
        if len(r.sha256) != 64 or not set(r.sha256) <= _HEX:
            problems.append(f"{r.image_id}: sha256 is not a 64-character lowercase hex digest")
        if r.group_id:
            groups[r.group_id].add(r.split)
    for group, splits in sorted(groups.items()):
        if len(splits) > 1:
            problems.append(f"group {group!r} spans splits {sorted(splits)}")
    counts = split_counts(records)
    for split, expected in (expected_counts or {}).items():
        if counts.get(split, 0) != expected:
            problems.append(
                f"split {split!r} has {counts.get(split, 0)} images; expected {expected}"
            )
    for split, expected in (expected_per_class or {}).items():
        per_class = per_class_counts(records, split)
        wrong = {
            c: per_class.get(c, 0)
            for c in range(class_map.num_classes)
            if per_class.get(c, 0) != expected
        }
        if wrong:
            sample = dict(list(wrong.items())[:5])
            problems.append(
                f"split {split!r}: {len(wrong)} class(es) do not have exactly {expected} images "
                f"(e.g. {sample})"
            )
    for split in require_all_classes_in:
        present = set(per_class_counts(records, split))
        missing = sorted(set(range(class_map.num_classes)) - present)
        if missing:
            problems.append(f"split {split!r} is missing class(es) {missing[:10]}")
    if problems:
        shown = "\n  - ".join(problems[:50])
        more = f"\n  ... and {len(problems) - 50} more" if len(problems) > 50 else ""
        raise DataValidationError(f"manifest validation failed:\n  - {shown}{more}")


def verify_files(
    records: Sequence[ImageRecord], data_root: str | Path, *, check_hashes: bool = True
) -> None:
    """Every file exists inside ``data_root`` (no link escapes) and matches its SHA-256."""
    root = Path(data_root).resolve()
    problems: list[str] = []
    for r in records:
        path = Path(data_root) / r.relative_path
        resolved = path.resolve()
        if root not in resolved.parents:
            problems.append(f"{r.image_id}: {r.relative_path} resolves outside the data root")
            continue
        if not path.is_file():
            problems.append(f"{r.image_id}: missing file {r.relative_path}")
            continue
        if check_hashes and sha256_file(path) != r.sha256:
            problems.append(f"{r.image_id}: SHA-256 of {r.relative_path} changed")
    if problems:
        shown = "\n  - ".join(problems[:50])
        more = f"\n  ... and {len(problems) - 50} more" if len(problems) > 50 else ""
        raise DataValidationError(f"file verification failed:\n  - {shown}{more}")
