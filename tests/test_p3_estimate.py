"""P3: the compute estimate is derived from recorded diagnostic timings (synthetic here)."""

from __future__ import annotations

import pytest

from fewshotlab.engine.ledger import append_event
from fewshotlab.io_utils import atomic_write_json
from fewshotlab.paths import get_paths
from fewshotlab.reporting.estimate import estimate, write_estimate


def _record(paths, dataset, method, support, val, timing):
    run_id = f"diagnostic__synthetic__{dataset}__{method}"
    attempt = paths.runs / run_id / "a001"
    attempt.mkdir(parents=True)
    atomic_write_json(
        attempt / "val_metrics.json",
        {
            "dataset": dataset,
            "method": method,
            "timing": timing,
            "device": "mps",
            "label_budget": {"fit_labels": support, "validation_labels": val},
        },
    )
    append_event(
        paths,
        event="status",
        run_id=run_id,
        attempt_id="a001",
        status="COMPLETED",
        category="diagnostic",
    )


def test_estimate_bounds_and_matrix_size(isolated_paths):
    paths = get_paths()
    learned = {"partial": 0.01, "scratch": 0.02, "frozen": 0.005}  # fit seconds per image
    for method, fit in learned.items():
        _record(
            paths,
            "flowers102",
            method,
            1020,
            1020,
            {"fit_seconds_per_epoch": [99.0, fit * 1020], "val_seconds_per_epoch": [99.0, 5.1]},
        )
    _record(
        paths,
        "flowers102",
        "prototypes",
        1020,
        1020,
        {"embedding_fit_seconds": 5.1, "val_seconds": 5.1},
    )
    _record(
        paths,
        "neu_cls",
        "partial",
        60,
        360,
        {"fit_seconds_per_epoch": [9.0, 0.6], "val_seconds_per_epoch": [9.0, 1.8]},
    )
    result = estimate(paths)
    assert result["missing_inputs"] == []
    assert result["full_matrix_unique_runs"] == 156

    def seconds(fit, val_pi, support, val, epochs):
        return epochs * (fit * support + val_pi * val) + val_pi * val

    for epochs, key in ((16, "lower_16_epochs"), (100, "upper_100_epochs")):
        expected = sum(
            seconds(fit, 0.005, k * 102, 1020, epochs) for fit in learned.values() for k in (1, 10)
        )
        expected += sum(0.005 * k * 102 + 0.005 * 1020 for k in (1, 10))
        assert result["mvp_8_jobs_hours"][key] == pytest.approx(round(expected / 3600, 2))
    assert any("ratio" in a for a in result["assumptions"])  # NEU scratch/frozen are scaled
    write_estimate(paths)
    assert (paths.reports / "compute_estimate.md").read_text().startswith("# Compute estimate")
