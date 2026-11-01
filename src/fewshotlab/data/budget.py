"""The full label budget per dataset and K, derived from stored manifests and supports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fewshotlab.config import DATASETS, SHOTS
from fewshotlab.data.manifest import SPLITS, load_class_map, read_manifest, split_counts
from fewshotlab.data.registry import (
    class_map_path,
    manifest_path,
    read_prepare_record,
    supports_dir,
)
from fewshotlab.io_utils import atomic_write_json, atomic_write_text, read_json, utc_now
from fewshotlab.paths import ProjectPaths
from fewshotlab.status import DataStatus


def label_budget(paths: ProjectPaths) -> dict[str, Any]:
    out: dict[str, Any] = {"generated_utc": utc_now(), "datasets": {}}
    for dataset in DATASETS:
        record = read_prepare_record(paths, dataset)
        status = (record or {}).get("status", DataStatus.NOT_PREPARED)
        if status != DataStatus.PREPARED:
            out["datasets"][dataset] = {"status": str(status)}
            continue
        cmap = load_class_map(class_map_path(paths, dataset))
        counts = split_counts(read_manifest(manifest_path(paths, dataset), splits=SPLITS))
        per_k = {}
        for k in SHOTS:
            files = sorted(supports_dir(paths).glob(f"{dataset}-k{k}-s*.json"))
            stored = sorted({read_json(f)["n_images"] for f in files})
            per_k[str(k)] = {
                "fit_labels": k * cmap.num_classes,
                "support_files": len(files),
                "support_n_images_stored": stored,
                "validation_labels": counts["val"],
                "held_out_test_images": counts["test"],
            }
        out["datasets"][dataset] = {
            "status": "PREPARED",
            "classes": cmap.num_classes,
            "train_pool": counts["train"],
            "per_k": per_k,
        }
    return out


def write_label_budget(paths: ProjectPaths) -> Path:
    budget = label_budget(paths)
    lines = [
        "# Label budget",
        "",
        f"Generated {budget['generated_utc']} from the stored manifests and support files. "
        "Fit labels are the K-shot support images; validation labels are also used for "
        "checkpoint selection; test images stay held out until the sealed P5 gate.",
        "",
    ]
    for dataset, info in budget["datasets"].items():
        lines.append(f"## {dataset}")
        if info["status"] != "PREPARED":
            lines += ["", f"Status: {info['status']} (no counts until the dataset is prepared)", ""]
            continue
        lines += [
            "",
            f"{info['classes']} classes; training pool {info['train_pool']} images.",
            "",
            "| K | fit labels | support files | validation labels | held-out test images |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, row in info["per_k"].items():
            stored = row["support_n_images_stored"]
            check = "" if stored in ([], [row["fit_labels"]]) else f" (stored {stored})"
            lines.append(
                f"| {k} | {row['fit_labels']}{check} | {row['support_files']} | "
                f"{row['validation_labels']} | {row['held_out_test_images']} |"
            )
        lines.append("")
    path = paths.reports / "label_budget.md"
    atomic_write_text(path, "\n".join(lines))
    atomic_write_json(paths.reports / "label_budget.json", budget)
    return path
