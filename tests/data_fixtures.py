"""Synthetic image datasets for data-layer tests. Software fixtures only, never real data."""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from fewshotlab.data.manifest import (
    ClassMap,
    ImageRecord,
    split_counts,
    write_class_map,
    write_manifest,
)
from fewshotlab.data.registry import class_map_path, manifest_path, write_prepare_record
from fewshotlab.hashing import sha256_file
from fewshotlab.paths import get_paths

NEU_PREFIXES = ("RS", "Pa", "Cr", "PS", "In", "Sc")


def noise_image(seed: int, size: tuple[int, int] = (32, 32), mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    shape = (size[1], size[0], 3) if mode == "RGB" else (size[1], size[0])
    return Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8))


def make_dataset(
    data_root: Path,
    *,
    dataset: str = "fixture",
    num_classes: int = 3,
    per_split: dict[str, int] | None = None,
    size: tuple[int, int] = (32, 32),
) -> tuple[list[ImageRecord], ClassMap]:
    per_split = per_split or {"train": 10, "val": 2, "test": 2}
    cmap = ClassMap(dataset, tuple(f"class_{c}" for c in range(num_classes)))
    records: list[ImageRecord] = []
    seed = 0
    for split, count in per_split.items():
        for class_id in range(num_classes):
            for i in range(count):
                relative = f"{dataset}/{split}/c{class_id}_{i:02d}.png"
                path = Path(data_root) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                noise_image(seed, size).save(path)
                seed += 1
                records.append(
                    ImageRecord(
                        f"{dataset}/{split}_c{class_id}_{i:02d}",
                        relative,
                        class_id,
                        cmap.names[class_id],
                        split,
                        sha256_file(path),
                    )
                )
    return records, cmap


def install_dataset(records: list[ImageRecord], cmap: ClassMap, dataset: str = "fixture"):
    """Register a fixture dataset as PREPARED in the (isolated) manifests folder."""
    paths = get_paths()
    digest = write_manifest(records, manifest_path(paths, dataset))
    write_class_map(cmap, class_map_path(paths, dataset))
    write_prepare_record(
        paths,
        dataset,
        {
            "dataset": dataset,
            "status": "PREPARED",
            "summary": "software fixture",
            "is_fixture": True,
            "source": {"mode": "software fixture"},
            "manifest": {
                "path": manifest_path(paths, dataset).name,
                "sha256": digest,
                "rows": len(records),
                "counts": split_counts(records),
            },
            "class_map": {"sha256": cmap.sha256()},
        },
    )
    return paths


def neu_tree(
    root: Path,
    *,
    per_class: int = 5,
    size: tuple[int, int] = (200, 200),
    prefixes: tuple[str, ...] = NEU_PREFIXES,
    mode: str = "L",
    suffix: str = ".bmp",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    seed = 1000
    for prefix in prefixes:
        for number in range(1, per_class + 1):
            noise_image(seed, size, mode).save(root / f"{prefix}_{number}{suffix}")
            seed += 1
    return root


def zip_dir(src: Path, zip_path: Path, *, arc_prefix: str = "NEU-CLS") -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in sorted(src.rglob("*")):
            if file.is_file():
                zf.write(file, f"{arc_prefix}/{file.relative_to(src).as_posix()}")
    return zip_path


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def make_flowers_files(base: Path, *, classes: int = 3, per_split=(2, 2, 3)) -> dict:
    """Synthetic official-format files: setid.mat, imagelabels.mat (1-based), jpg/, tgz."""
    from scipy.io import savemat

    jpg = base / "jpg"
    jpg.mkdir(parents=True)
    numbers = iter(range(1, 100_000))
    split_ids: list[list[int]] = [[], [], []]
    label_of: dict[int, int] = {}
    for split_index, count in enumerate(per_split):
        for class_id in range(classes):
            for _ in range(count):
                number = next(numbers)
                split_ids[split_index].append(number)
                label_of[number] = class_id + 1  # official labels are 1-based
    total = max(label_of)
    savemat(
        base / "imagelabels.mat",
        {"labels": np.array([[label_of[i] for i in range(1, total + 1)]], dtype=np.uint8)},
    )
    savemat(
        base / "setid.mat",
        {
            key: np.array([ids], dtype=np.uint16)
            for key, ids in zip(("trnid", "valid", "tstid"), split_ids, strict=True)
        },
    )
    for number in range(1, total + 1):
        noise_image(number, (40, 30)).save(jpg / f"image_{number:05d}.jpg", quality=90)
    with tarfile.open(base / "102flowers.tgz", "w:gz") as tar:
        tar.add(jpg, arcname="jpg")
    return {
        "image": ("102flowers.tgz", md5(base / "102flowers.tgz")),
        "label": ("imagelabels.mat", md5(base / "imagelabels.mat")),
        "setid": ("setid.mat", md5(base / "setid.mat")),
    }


def make_color_dataset(
    data_root: Path,
    *,
    dataset: str = "fixture-color",
    per_split: dict[str, int] | None = None,
    size: tuple[int, int] = (48, 48),
) -> tuple[list[ImageRecord], ClassMap]:
    """A learnable fixture: noisy red / green / blue images (software test data only)."""
    per_split = per_split or {"train": 4, "val": 4, "test": 2}
    colors = ((200, 40, 40), (40, 200, 40), (40, 40, 200))
    cmap = ClassMap(dataset, ("red", "green", "blue"))
    rng = np.random.default_rng(7)
    records: list[ImageRecord] = []
    for split, count in per_split.items():
        for class_id, color in enumerate(colors):
            for i in range(count):
                noise = rng.normal(0, 20, size=(size[1], size[0], 3))
                pixels = np.clip(np.array(color)[None, None, :] + noise, 0, 255).astype(np.uint8)
                relative = f"{dataset}/{split}/c{class_id}_{i:02d}.png"
                path = Path(data_root) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
                records.append(
                    ImageRecord(
                        f"{dataset}/{split}_c{class_id}_{i:02d}",
                        relative,
                        class_id,
                        cmap.names[class_id],
                        split,
                        sha256_file(path),
                    )
                )
    return records, cmap


def training_fixture(
    data_root: Path, *, dataset: str = "fixture-color", k: int = 4, seed: int = 11
):
    """Prepared, audited fixture dataset with one support manifest; returns ProjectPaths."""
    from fewshotlab.data.audit import run_audit
    from fewshotlab.data.support import build_supports

    records, cmap = make_color_dataset(data_root, dataset=dataset)
    paths = install_dataset(records, cmap, dataset=dataset)
    run_audit(dataset, paths=paths)
    build_supports(paths, dataset, shots=(k,), seeds=(seed,))
    return paths


def fixture_config(
    paths,
    method: str,
    *,
    dataset: str = "fixture-color",
    k: int = 4,
    seed: int = 11,
    max_epochs: int = 8,
    patience: int = 15,
):
    from fewshotlab.config import parse_run_config
    from fewshotlab.data.support import support_path

    return parse_run_config(
        {
            "dataset": dataset,
            "method": method,
            "shots": k,
            "support_seed": seed,
            "split_manifest": str(manifest_path(paths, dataset)),
            "support_manifest": str(support_path(paths, dataset, k, seed)),
            "category": "fixture",
            "is_fixture": True,
            "device": "cpu",
            "train": {
                "max_epochs": max_epochs,
                "patience": patience,
                "batch_size_cap": 32,
                "num_workers": 0,
            },
        }
    )
