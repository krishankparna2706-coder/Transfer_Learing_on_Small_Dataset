"""Where each dataset's manifest, class map, preparation record and supports live."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fewshotlab.errors import DatasetUnavailableError
from fewshotlab.io_utils import atomic_write_json, read_json
from fewshotlab.paths import ProjectPaths
from fewshotlab.status import DataStatus

MANIFEST_VERSION = "v1"


def manifest_path(paths: ProjectPaths, dataset: str) -> Path:
    return paths.manifests / f"{dataset}-{MANIFEST_VERSION}.csv"


def class_map_path(paths: ProjectPaths, dataset: str) -> Path:
    return paths.manifests / f"{dataset}.class_map.json"


def prepare_record_path(paths: ProjectPaths, dataset: str) -> Path:
    return paths.manifests / f"{dataset}.prepare.json"


def supports_dir(paths: ProjectPaths) -> Path:
    return paths.manifests / "supports"


def audit_dir(paths: ProjectPaths, dataset: str) -> Path:
    return paths.reports / "data_audit" / dataset


def read_prepare_record(paths: ProjectPaths, dataset: str) -> dict[str, Any] | None:
    path = prepare_record_path(paths, dataset)
    return read_json(path) if path.is_file() else None


def write_prepare_record(paths: ProjectPaths, dataset: str, record: dict[str, Any]) -> Path:
    return atomic_write_json(prepare_record_path(paths, dataset), record)


def require_prepared(paths: ProjectPaths, dataset: str) -> dict[str, Any]:
    record = read_prepare_record(paths, dataset)
    status = (record or {}).get("status", DataStatus.NOT_PREPARED)
    if status != DataStatus.PREPARED:
        detail = f" ({record.get('summary')})" if record and record.get("summary") else ""
        raise DatasetUnavailableError(
            f"{dataset} is {status}{detail}; run `python -m fewshotlab data prepare "
            f"--dataset {dataset}` first"
        )
    return record  # type: ignore[return-value]
