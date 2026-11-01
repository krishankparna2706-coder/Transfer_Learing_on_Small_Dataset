"""The ordered replay manifest, sealed before test access and consumed unchanged in P7.

The replay streams one dataset's held-out test images (NEU-CLS) in one fixed, seeded order:
the clean stream first, then each stress condition over the same parents in the same order.
The order is a permutation of the sorted test image IDs and does not depend on labels or pixels.
The true class is copied from the split manifest so that P7 can reveal it in demonstration mode.
Derivatives keep their parent's split; each row names its parent.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from fewshotlab.data.manifest import read_manifest
from fewshotlab.data.registry import manifest_path
from fewshotlab.hashing import sha256_bytes, sha256_json
from fewshotlab.paths import ProjectPaths
from fewshotlab.seeds import NUMPY_BIT_GENERATOR, SEED_DERIVATION, numpy_rng
from fewshotlab.simulation.corruptions import (
    CONDITIONS,
    CORRUPTION_SEED,
    PARAMETERS,
    condition_id,
    noise_seed,
)

REPLAY_FILE = "replay_manifest.csv"
COLUMNS = (
    "position",
    "stream",
    "parent_image_id",
    "derivative_id",
    "split",
    "true_class_id",
    "condition",
    "severity",
    "parameters",
    "seed",
)


def replay_manifest(paths: ProjectPaths, dataset: str = "neu_cls") -> dict[str, Any]:
    records = sorted(
        read_manifest(manifest_path(paths, dataset), splits=("test",)), key=lambda r: r.image_id
    )
    order = numpy_rng(dataset, "replay-order", CORRUPTION_SEED).permutation(len(records))
    parents = [records[int(i)] for i in order]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    position = 0
    for condition, severity in CONDITIONS:
        stream = condition_id(condition, severity)
        for record in parents:
            seed = noise_seed(record.image_id, condition, severity)
            writer.writerow(
                [
                    position,
                    stream,
                    record.image_id,
                    f"{record.image_id}#{stream}",
                    record.split,
                    record.class_id,
                    condition,
                    severity,
                    json.dumps(PARAMETERS[(condition, severity)], sort_keys=True),
                    "" if seed is None else seed,
                ]
            )
            position += 1
    data = buffer.getvalue().encode("utf-8")
    return {
        "file": REPLAY_FILE,
        "dataset": dataset,
        "split": "test",
        "parents": len(parents),
        "rows": position,
        "streams": [condition_id(c, s) for c, s in CONDITIONS],
        "order_rule": "test image IDs sorted ascending, then permuted once; same order per stream",
        "rng": {
            "bit_generator": NUMPY_BIT_GENERATOR,
            "derivation": SEED_DERIVATION,
            "seed_labels": [dataset, "replay-order", CORRUPTION_SEED],
        },
        "parent_order_sha256": sha256_json([r.image_id for r in parents]),
        "csv_sha256": sha256_bytes(data),
        "csv": data,
    }


def sealed_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest without its bytes (what the sealed plan records)."""
    return {key: value for key, value in manifest.items() if key != "csv"}
