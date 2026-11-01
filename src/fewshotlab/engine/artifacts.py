"""Run identities, attempt directories, status files and prediction files."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fewshotlab.config import RunConfig
from fewshotlab.errors import DataValidationError
from fewshotlab.hashing import sha256_bytes
from fewshotlab.io_utils import atomic_write_bytes, atomic_write_json, read_json, utc_now
from fewshotlab.paths import ProjectPaths, portable_path
from fewshotlab.status import RunStatus, check_transition

PREDICTION_COLUMNS = (
    "image_id",
    "split",
    "condition",
    "true_class_id",
    "pred_class_id",
    "run_id",
    "attempt_id",
    "score_kind",
    "scores",
)


def make_run_id(cfg: RunConfig, config_sha256: str) -> str:
    """Protocol, dataset, method, K, seed and config hash; non-core categories are prefixed."""
    core = (
        f"{cfg.protocol_id}__{cfg.dataset}__{cfg.method}__k{cfg.shots}"
        f"__s{cfg.support_seed}__cfg{config_sha256[:8]}"
    )
    return core if cfg.category == "core" else f"{cfg.category}__{core}"


#: Run folders live under runs/artifacts/. On a machine where the project folder is cloud-synced,
#: runs/artifacts may be a symlink to local storage (decision D-039); elsewhere it is a folder.
ARTIFACTS_DIRNAME = "artifacts"


def run_root(paths: ProjectPaths, run_id: str) -> Path:
    return paths.runs / ARTIFACTS_DIRNAME / run_id


def logical_run_dir(paths: ProjectPaths, run_id: str, attempt_id: str) -> str:
    """The in-project path recorded in the ledger (symlinks are not resolved)."""
    return f"{portable_path(paths.runs)}/{ARTIFACTS_DIRNAME}/{run_id}/{attempt_id}"


def attempt_dirs(paths: ProjectPaths, run_id: str) -> list[Path]:
    root = run_root(paths, run_id)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("a"))


def new_attempt(paths: ProjectPaths, run_id: str) -> tuple[str, Path]:
    existing = attempt_dirs(paths, run_id)
    number = int(existing[-1].name[1:]) + 1 if existing else 1
    attempt_id = f"a{number:03d}"
    directory = run_root(paths, run_id) / attempt_id
    directory.mkdir(parents=True, exist_ok=False)
    return attempt_id, directory


def read_status(attempt_dir: Path) -> dict[str, Any] | None:
    path = attempt_dir / "status.json"
    return read_json(path) if path.is_file() else None


def write_status(attempt_dir: Path, status: RunStatus, **extra: Any) -> dict[str, Any]:
    previous = read_status(attempt_dir)
    old = RunStatus(previous["status"]) if previous else RunStatus.NOT_RUN
    if old != status:
        check_transition(old, status)
    record = {**(previous or {}), **extra, "status": status.value, "updated_utc": utc_now()}
    record.setdefault("history", [])
    if old != status:
        record["history"] = [*record["history"], {"status": status.value, "time_utc": utc_now()}]
    atomic_write_json(attempt_dir / "status.json", record)
    return record


def _format_scores(row: np.ndarray) -> str:
    return (
        "["
        + ",".join(np.format_float_positional(np.float32(v), unique=True, trim="-") for v in row)
        + "]"
    )


def write_predictions(
    path: Path,
    *,
    image_ids: Sequence[str],
    split: str,
    true: Sequence[int],
    pred: Sequence[int],
    scores: np.ndarray,
    run_id: str,
    attempt_id: str,
    score_kind: str,
    condition: str = "clean",
) -> str:
    """One row per image (sorted by image_id); exact float32 score vectors. Returns SHA-256."""
    if not (len(image_ids) == len(true) == len(pred) == len(scores)):
        raise DataValidationError("prediction columns have different lengths")
    order = np.argsort(np.asarray(image_ids, dtype=object), kind="stable")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PREDICTION_COLUMNS)
    scores32 = np.asarray(scores, dtype=np.float32)
    for i in order:
        writer.writerow(
            [
                image_ids[i],
                split,
                condition,
                int(true[i]),
                int(pred[i]),
                run_id,
                attempt_id,
                score_kind,
                _format_scores(scores32[i]),
            ]
        )
    data = buffer.getvalue().encode("utf-8")
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def read_predictions(path: Path) -> dict[str, Any]:
    ids, true, pred, scores = [], [], [], []
    meta: dict[str, set[str]] = {
        "split": set(),
        "run_id": set(),
        "attempt_id": set(),
        "condition": set(),
        "score_kind": set(),
    }
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise DataValidationError(f"{path}: unexpected prediction columns")
        for row in reader:
            ids.append(row["image_id"])
            true.append(int(row["true_class_id"]))
            pred.append(int(row["pred_class_id"]))
            scores.append(np.asarray(json.loads(row["scores"]), dtype=np.float32))
            for key in meta:
                meta[key].add(row[key])
    return {
        "image_ids": ids,
        "true": np.asarray(true),
        "pred": np.asarray(pred),
        "scores": np.stack(scores) if scores else np.zeros((0, 0), np.float32),
        **{key: sorted(values) for key, values in meta.items()},
    }
