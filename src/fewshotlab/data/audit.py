"""Data audit: integrity, exact duplicates, near-duplicate candidates and split separation.

The audit is label-blind beyond structure: labels are read only to count classes per split and to
show which split and class a flagged pair belongs to. It never edits a manifest. Findings lead to
documented decisions, never to silent repairs.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from fewshotlab.data.imageio import ImageDecodeError, open_image, phash64, pixel_sha256
from fewshotlab.data.manifest import (
    SPLITS,
    load_class_map,
    per_class_counts,
    read_manifest,
    split_counts,
    validate_records,
)
from fewshotlab.data.registry import (
    audit_dir,
    class_map_path,
    manifest_path,
    read_prepare_record,
    require_prepared,
)
from fewshotlab.errors import DataValidationError
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import append_jsonl, atomic_write_json, atomic_write_text, utc_now
from fewshotlab.paths import ProjectPaths, get_paths

AUDIT_VERSION = 1
DEFAULT_NEAR_DUP_THRESHOLD = 4
HISTOGRAM_THRESHOLDS = (0, 2, 4, 6, 8, 10, 12)
PHASH_METHOD = (
    "64-bit DCT perceptual hash (8-bit grayscale, Lanczos resize to 32x32, orthonormal 2-D "
    "DCT-II, top-left 8x8 block vs its median); pairs compared by Hamming distance"
)


def hamming_scan(
    hashes: np.ndarray, max_distance: int, chunk: int = 512
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Pairs (i < j) within ``max_distance`` bits, plus a histogram over all pair distances."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = len(hashes)
    histogram = np.zeros(65, dtype=np.int64)
    pairs: list[tuple[int, int, int]] = []
    columns = np.arange(n)
    for start in range(0, n, chunk):
        block = hashes[start : start + chunk]
        distances = np.bitwise_count(block[:, None] ^ hashes[None, :]).astype(np.int64)
        rows = np.arange(start, start + len(block))[:, None]
        upper = columns[None, :] > rows
        histogram += np.bincount(distances[upper], minlength=65)
        hit_rows, hit_cols = np.nonzero(upper & (distances <= max_distance))
        for r, c in zip(hit_rows.tolist(), hit_cols.tolist(), strict=True):
            pairs.append((start + r, c, int(distances[r, c])))
    return pairs, histogram


def require_passing_audit(
    paths: ProjectPaths, dataset: str, manifest_sha256: str
) -> dict[str, Any]:
    """The audit is a blocking prerequisite for support selection and training."""
    report_path = audit_dir(paths, dataset) / "data_audit.json"
    if not report_path.is_file():
        raise DataValidationError(
            f"no data audit found for {dataset}; run `python -m fewshotlab data audit --dataset "
            f"{dataset}` first (the audit is a blocking prerequisite)"
        )
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest", {}).get("sha256") != manifest_sha256:
        raise DataValidationError(
            f"the {dataset} audit was made for a different manifest; re-run `data audit`"
        )
    if report.get("status") != "PASS":
        raise DataValidationError(f"the latest {dataset} audit status is {report.get('status')}")
    return report


def run_audit(
    dataset: str,
    *,
    near_dup_threshold: int = DEFAULT_NEAR_DUP_THRESHOLD,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    paths = paths or get_paths()
    prepare_record = require_prepared(paths, dataset)
    if not 0 <= near_dup_threshold <= 64:
        raise DataValidationError("--near-dup-threshold must be between 0 and 64")
    mpath = manifest_path(paths, dataset)
    manifest_sha = sha256_file(mpath)
    cmap = load_class_map(class_map_path(paths, dataset))
    records = read_manifest(mpath, splits=SPLITS)  # structural audit only
    validate_records(records, cmap)
    root = paths.data_root.resolve()
    integrity: list[str] = []
    decode_failures: list[str] = []
    image_rows: list[dict[str, Any]] = []
    dims: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    hashed: list[tuple[int, int]] = []  # (record index, phash)
    for index, r in enumerate(records):
        path = paths.data_root / r.relative_path
        if root not in path.resolve().parents or not path.is_file():
            integrity.append(f"{r.image_id}: file missing or outside the data root")
            continue
        if sha256_file(path) != r.sha256:
            integrity.append(f"{r.image_id}: SHA-256 differs from the manifest")
        try:
            image = open_image(path)
        except ImageDecodeError as exc:
            decode_failures.append(str(exc))
            continue
        width, height = image.size
        dims[f"{width}x{height}"] += 1
        modes[image.mode] += 1
        formats[str(image.info.get("fewshotlab_format"))] += 1
        ph = phash64(image)
        hashed.append((index, ph))
        image_rows.append(
            {
                "image_id": r.image_id,
                "split": r.split,
                "class_id": r.class_id,
                "width": width,
                "height": height,
                "mode": image.mode,
                "pixel_sha256": pixel_sha256(image),
                "phash": f"{ph:016x}",
            }
        )
    # exact duplicates, by file bytes and by decoded pixels
    byte_groups: dict[str, list[int]] = defaultdict(list)
    for index, r in enumerate(records):
        byte_groups[r.sha256].append(index)
    pixel_groups: dict[str, list[str]] = defaultdict(list)
    for row in image_rows:
        pixel_groups[row["pixel_sha256"]].append(row["image_id"])
    split_of = {r.image_id: r.split for r in records}
    class_of = {r.image_id: r.class_id for r in records}
    exact = [sorted(members) for members in pixel_groups.values() if len(members) > 1]
    exact_cross_split = [m for m in exact if len({split_of[i] for i in m}) > 1]
    exact_cross_class = [m for m in exact if len({class_of[i] for i in m}) > 1]
    # near duplicates
    order = [index for index, _ in hashed]
    values = np.array([ph for _, ph in hashed], dtype=np.uint64)
    pairs, histogram = hamming_scan(values, near_dup_threshold)
    cumulative = np.cumsum(histogram)
    near_rows = []
    for i, j, distance in pairs:
        a, b = records[order[i]], records[order[j]]
        near_rows.append(
            {
                "image_id_a": a.image_id,
                "split_a": a.split,
                "class_a": a.class_id,
                "image_id_b": b.image_id,
                "split_b": b.split,
                "class_b": b.class_id,
                "hamming": distance,
                "same_class": a.class_id == b.class_id,
                "cross_split": a.split != b.split,
            }
        )
    cross_split_near = [row for row in near_rows if row["cross_split"]]
    status = "FAIL" if integrity or decode_failures else "PASS"
    findings: list[str] = []
    if exact_cross_split:
        findings.append(
            f"{len(exact_cross_split)} exact-duplicate group(s) span splits; record a "
            "dataset decision (official splits are preserved, never rewritten)"
        )
    if exact_cross_class:
        findings.append(f"{len(exact_cross_class)} exact-duplicate group(s) carry different labels")
    if cross_split_near:
        findings.append(
            f"{len(cross_split_near)} near-duplicate candidate pair(s) span splits "
            f"(pHash distance <= {near_dup_threshold}); manual review required"
        )
    findings.append("no acquisition-group metadata is available; splits are image-level")
    counts = split_counts(records)
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "dataset": dataset,
        "generated_utc": utc_now(),
        "status": status,
        "manifest": {"path": mpath.name, "sha256": manifest_sha, "rows": len(records)},
        "class_map_sha256": cmap.sha256(),
        "source": {
            "mode": prepare_record.get("source", {}).get("mode"),
            "archive": prepare_record.get("archive"),
            "files": prepare_record.get("files"),
            "prepared_utc": prepare_record.get("prepared_utc"),
        },
        "counts": counts,
        "per_class_count_range": {
            split: {"min": min(pc.values()), "max": max(pc.values())}
            for split in SPLITS
            if (pc := per_class_counts(records, split))
        },
        "integrity_failures": integrity,
        "decode_failures": decode_failures,
        "dimensions_top": dict(dims.most_common(10)),
        "distinct_dimensions": len(dims),
        "modes": dict(modes),
        "formats": dict(formats),
        "exact_duplicates": {
            "byte_identical_groups": sum(1 for m in byte_groups.values() if len(m) > 1),
            "pixel_identical_groups": len(exact),
            "cross_split_groups": exact_cross_split,
            "cross_class_groups": exact_cross_class,
        },
        "near_duplicates": {
            "method": PHASH_METHOD,
            "threshold": near_dup_threshold,
            "pairs_within_threshold": len(near_rows),
            "cross_split_pairs": len(cross_split_near),
            "cross_class_pairs": sum(1 for row in near_rows if not row["same_class"]),
            "pairs_at_or_below": {str(t): int(cumulative[t]) for t in HISTOGRAM_THRESHOLDS},
            "decision_rule": (
                "candidates are for manual review only; nothing is removed automatically"
            ),
        },
        "groups": {
            "metadata": "none available from the source",
            "consequence": "image-level splits cannot establish generalization to new "
            "acquisition groups (lines, coils, cameras)",
        },
        "exclusions": [],
        "findings": findings,
    }
    out = audit_dir(paths, dataset)
    atomic_write_json(out / "data_audit.json", report)
    atomic_write_text(out / "data_audit.md", _markdown(report))
    atomic_write_text(
        out / "near_duplicate_candidates.csv",
        _csv(
            near_rows,
            [
                "image_id_a",
                "split_a",
                "class_a",
                "image_id_b",
                "split_b",
                "class_b",
                "hamming",
                "same_class",
                "cross_split",
            ],
        ),
    )
    atomic_write_text(
        out / "image_audit.csv",
        _csv(
            image_rows,
            ["image_id", "split", "class_id", "width", "height", "mode", "pixel_sha256", "phash"],
        ),
    )
    append_jsonl(
        out / "audit_history.jsonl",
        {
            "generated_utc": report["generated_utc"],
            "status": status,
            "manifest_sha256": manifest_sha,
            "near_dup_threshold": near_dup_threshold,
            "near_pairs": len(near_rows),
            "cross_split_near_pairs": len(cross_split_near),
            "exact_groups": len(exact),
            "integrity_failures": len(integrity),
            "decode_failures": len(decode_failures),
        },
    )
    return report


def _csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _markdown(report: dict[str, Any]) -> str:
    near = report["near_duplicates"]
    exact = report["exact_duplicates"]
    lines = [
        f"# Data audit: {report['dataset']}",
        "",
        f"- Generated: {report['generated_utc']} (audit v{report['audit_version']})",
        f"- **Status: {report['status']}**",
        f"- Manifest: `{report['manifest']['path']}` sha256 `{report['manifest']['sha256']}` "
        f"({report['manifest']['rows']} rows)",
        f"- Class map sha256: `{report['class_map_sha256']}`",
        f"- Source: {report['source']['mode']}",
        f"- Counts: train {report['counts']['train']}, val {report['counts']['val']}, "
        f"test {report['counts']['test']}",
        f"- Integrity failures: {len(report['integrity_failures'])}; decode failures: "
        f"{len(report['decode_failures'])}",
        f"- Dimensions (top): {report['dimensions_top']}; modes: {report['modes']}; "
        f"formats: {report['formats']}",
        "",
        "## Exact duplicates",
        f"- Byte-identical groups: {exact['byte_identical_groups']}; pixel-identical groups: "
        f"{exact['pixel_identical_groups']}",
        f"- Groups spanning splits: {len(exact['cross_split_groups'])}; groups spanning classes: "
        f"{len(exact['cross_class_groups'])}",
        "",
        "## Near-duplicate candidates (for manual review)",
        f"- Method: {near['method']}",
        f"- Threshold: Hamming distance <= {near['threshold']} of 64 bits",
        f"- Pairs within threshold: {near['pairs_within_threshold']} (cross-split: "
        f"{near['cross_split_pairs']}, cross-class: {near['cross_class_pairs']})",
        f"- Pairs at or below each distance: {near['pairs_at_or_below']}",
        f"- Rule: {near['decision_rule']}",
        "",
        "## Groups",
        f"- {report['groups']['metadata']}: {report['groups']['consequence']}",
        "",
        "## Findings",
        *[f"- {finding}" for finding in report["findings"]],
        "",
    ]
    return "\n".join(lines)


def latest_audit(paths: ProjectPaths, dataset: str) -> dict[str, Any] | None:
    path = audit_dir(paths, dataset) / "data_audit.json"
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def audit_source_summary(paths: ProjectPaths, dataset: str) -> str:
    record = read_prepare_record(paths, dataset) or {}
    return str(record.get("status", "NOT_PREPARED"))


__all__ = [
    "DEFAULT_NEAR_DUP_THRESHOLD",
    "audit_source_summary",
    "hamming_scan",
    "latest_audit",
    "require_passing_audit",
    "run_audit",
]
