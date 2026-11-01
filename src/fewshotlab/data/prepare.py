"""Dispatch ``data prepare`` to the dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fewshotlab.errors import ConfigError
from fewshotlab.paths import ProjectPaths, get_paths


def prepare_dataset(
    dataset: str,
    *,
    source: str | Path | None = None,
    allow_download: bool = False,
    mirror: str | None = None,
    cross_check: str | Path | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    paths = paths or get_paths()
    src = Path(source) if source is not None else None
    if dataset == "flowers102":
        from fewshotlab.data import flowers

        return flowers.prepare(
            paths,
            source=src,
            allow_download=allow_download,
            mirror=mirror,
            cross_check=Path(cross_check) if cross_check is not None else None,
        )
    if mirror is not None or cross_check is not None:
        raise ConfigError("--mirror and --cross-check apply to flowers102 only")
    if dataset == "neu_cls":
        from fewshotlab.data import neu

        return neu.prepare(paths, source=src, allow_download=allow_download)
    raise ValueError(f"unknown dataset {dataset!r}")
