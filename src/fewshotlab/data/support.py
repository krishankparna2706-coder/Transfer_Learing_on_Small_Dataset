"""Nested K-shot support sets.

For each dataset, support seed and class, the class's training IDs are sorted, permuted with a
class-local generator seeded from SHA-256(dataset, "support", seed, class_id), and the first
K = 1, 3, 5 and 10 entries are taken, so K=1 is contained in K=3, which is contained in K=5 and
K=10. Every method reads the same stored IDs. The stored manifest, not a re-run of the RNG, is the
authority: a support file is immutable, and a rebuild that would change it is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fewshotlab import PROTOCOL_ID
from fewshotlab.config import DATASETS, load_protocol
from fewshotlab.data.audit import require_passing_audit
from fewshotlab.data.manifest import load_class_map, read_manifest
from fewshotlab.data.registry import (
    class_map_path,
    manifest_path,
    read_prepare_record,
    require_prepared,
    supports_dir,
)
from fewshotlab.errors import DatasetUnavailableError, DataValidationError
from fewshotlab.hashing import sha256_file, sha256_json
from fewshotlab.io_utils import atomic_write_json, read_json, utc_now
from fewshotlab.paths import ProjectPaths, get_paths
from fewshotlab.seeds import NUMPY_BIT_GENERATOR, SEED_DERIVATION, numpy_rng
from fewshotlab.status import DataStatus

SCHEMA_VERSION = 1
_CONTENT_KEYS = (
    "schema_version",
    "protocol_id",
    "dataset",
    "split_manifest",
    "split_manifest_sha256",
    "class_map_sha256",
    "k",
    "seed",
    "rng",
    "selected",
    "n_images",
)


def support_path(paths: ProjectPaths, dataset: str, k: int, seed: int) -> Path:
    return supports_dir(paths) / f"{dataset}-k{k}-s{seed}.json"


def content_hash(doc: dict[str, Any]) -> str:
    """Canonical hash of a support's content (timestamps and tool versions excluded)."""
    return sha256_json({key: doc[key] for key in _CONTENT_KEYS})


def validate_support(
    doc: dict[str, Any], *, train_ids_by_class: dict[int, set[str]], heldout_ids: set[str]
) -> None:
    problems: list[str] = []
    k = doc["k"]
    classes = sorted(train_ids_by_class)
    if set(doc["selected"]) != {str(c) for c in classes}:  # keys compared as a set
        problems.append("selected classes do not match the class map")
    for key, ids in doc["selected"].items():
        class_id = int(key)
        if len(ids) != k:
            problems.append(f"class {class_id}: {len(ids)} images, expected exactly K={k}")
        if len(set(ids)) != len(ids):
            problems.append(f"class {class_id}: repeated image IDs")
        leaked = sorted(set(ids) & heldout_ids)
        if leaked:
            problems.append(f"class {class_id}: validation/test image(s) in a support set {leaked}")
        foreign = sorted(set(ids) - train_ids_by_class.get(class_id, set()) - set(leaked))
        if foreign:
            problems.append(
                f"class {class_id}: IDs that are not training images of this class {foreign[:5]}"
            )
    if doc["n_images"] != k * len(classes):
        problems.append(f"n_images {doc['n_images']} != K x classes = {k * len(classes)}")
    if doc.get("content_sha256") != content_hash(doc):
        problems.append("content_sha256 does not match the stored content (file was altered)")
    if problems:
        raise DataValidationError(
            f"support {doc['dataset']} K={k} seed={doc['seed']} is invalid:"
            "\n  - " + "\n  - ".join(problems)
        )


def _load_split_ids(mpath: Path) -> tuple[dict[int, set[str]], set[str]]:
    train = read_manifest(mpath, splits=("train",))
    heldout = read_manifest(mpath, splits=("val", "test"))  # IDs only, for disjointness checks
    by_class: dict[int, set[str]] = {}
    for r in train:
        by_class.setdefault(r.class_id, set()).add(r.image_id)
    return by_class, {r.image_id for r in heldout}


def build_supports(
    paths: ProjectPaths,
    dataset: str,
    *,
    shots: tuple[int, ...],
    seeds: tuple[int, ...],
    require_audit: bool = True,
) -> list[dict[str, Any]]:
    record = require_prepared(paths, dataset)
    mpath = manifest_path(paths, dataset)
    manifest_sha = sha256_file(mpath)
    if record.get("manifest", {}).get("sha256") != manifest_sha:
        raise DataValidationError(f"{mpath.name} changed after preparation; re-run data prepare")
    if require_audit:
        require_passing_audit(paths, dataset, manifest_sha)
    cmap = load_class_map(class_map_path(paths, dataset))
    train_by_class, heldout = _load_split_ids(mpath)
    for class_id in range(cmap.num_classes):
        train_by_class.setdefault(class_id, set())
    k_max = max(shots)
    short = {c: len(ids) for c, ids in train_by_class.items() if len(ids) < k_max}
    if short:
        raise DataValidationError(
            f"K={k_max} needs {k_max} training images per class, but class(es) {short} have "
            "fewer (sampling with replacement is never used)"
        )
    results: list[dict[str, Any]] = []
    for seed in seeds:
        orders = {}
        for class_id in sorted(train_by_class):
            ids = sorted(train_by_class[class_id])
            permutation = numpy_rng(dataset, "support", seed, class_id).permutation(len(ids))
            orders[class_id] = [ids[int(i)] for i in permutation]
        previous: dict[str, list[str]] | None = None
        for k in sorted(shots):
            doc: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "dataset": dataset,
                "split_manifest": mpath.name,
                "split_manifest_sha256": manifest_sha,
                "class_map_sha256": cmap.sha256(),
                "k": k,
                "seed": seed,
                "rng": {
                    "bit_generator": NUMPY_BIT_GENERATOR,
                    "derivation": SEED_DERIVATION,
                    "seed_labels": [dataset, "support", seed, "<class_id>"],
                    "ordering": "image_id ascending within each class, then permuted",
                },
                "selected": {str(c): orders[c][:k] for c in sorted(orders)},
                "n_images": k * cmap.num_classes,
            }
            doc["content_sha256"] = content_hash(doc)
            validate_support(doc, train_ids_by_class=train_by_class, heldout_ids=heldout)
            if previous is not None:
                for key, ids in previous.items():
                    if not set(ids) <= set(doc["selected"][key]):
                        raise DataValidationError(f"nesting violated for class {key} at K={k}")
            previous = doc["selected"]
            path = support_path(paths, dataset, k, seed)
            if path.exists():
                stored = read_json(path)
                if stored.get("content_sha256") != doc["content_sha256"]:
                    raise DataValidationError(
                        f"{path.name} already exists with different content (stored "
                        f"{stored.get('content_sha256')}, rebuilt {doc['content_sha256']}). "
                        "Support "
                        "manifests are immutable; a changed split needs a new protocol version."
                    )
                state = "verified (unchanged)"
            else:
                doc["created_utc"] = utc_now()
                doc["numpy_version"] = np.__version__
                atomic_write_json(path, doc)
                state = "written"
            results.append(
                {
                    "dataset": dataset,
                    "k": k,
                    "seed": seed,
                    "path": str(path),
                    "content_sha256": doc["content_sha256"],
                    "state": state,
                }
            )
    return results


def load_support(paths: ProjectPaths, dataset: str, k: int, seed: int) -> dict[str, Any]:
    """Read a stored support, verify its hash, its manifest and its membership rules."""
    path = support_path(paths, dataset, k, seed)
    if not path.is_file():
        raise DataValidationError(f"support manifest not found: {path}; run `support build`")
    doc = read_json(path)
    mpath = manifest_path(paths, dataset)
    if doc.get("split_manifest_sha256") != sha256_file(mpath):
        raise DataValidationError(f"{path.name} was built from a different {mpath.name}")
    train_by_class, heldout = _load_split_ids(mpath)
    validate_support(doc, train_ids_by_class=train_by_class, heldout_ids=heldout)
    return doc


def build_from_config(
    config_path: str | Path, *, dataset: str | None = None, paths: ProjectPaths | None = None
) -> dict[str, Any]:
    """``support build``: every requested dataset that is prepared and audited."""
    paths = paths or get_paths()
    protocol = load_protocol(config_path)
    shots = tuple(protocol["shots"])
    seeds = tuple(protocol["support_seeds"])
    wanted = [dataset] if dataset else [d for d in DATASETS if d in protocol["datasets"]]
    summary: dict[str, Any] = {"shots": shots, "seeds": seeds, "datasets": {}}
    for name in wanted:
        record = read_prepare_record(paths, name)
        status = (record or {}).get("status", DataStatus.NOT_PREPARED)
        if status != DataStatus.PREPARED:
            summary["datasets"][name] = {"status": str(status), "built": []}
            continue
        built = build_supports(paths, name, shots=shots, seeds=seeds)
        summary["datasets"][name] = {"status": "BUILT", "built": built}
    from fewshotlab.data.budget import write_label_budget

    summary["label_budget_report"] = str(write_label_budget(paths))
    missing = {n: v["status"] for n, v in summary["datasets"].items() if v["status"] != "BUILT"}
    summary["missing"] = missing
    return summary


__all__ = [
    "build_from_config",
    "build_supports",
    "content_hash",
    "load_support",
    "support_path",
    "validate_support",
    "DatasetUnavailableError",
]
