"""P5: seed aggregation counts deterministic duplicates once and pairs methods by seed."""

from __future__ import annotations

import math

import pytest

from fewshotlab.reporting.benchmark_report import aggregate, paired


def _row(method, seed, value, *, source=None, duplicate=False, k=10, dataset="d"):
    return {
        "dataset": dataset,
        "k": k,
        "method": method,
        "seed": seed,
        "run_id": f"{method}-{seed}",
        "source_run_id": source or f"{method}-{seed}",
        "duplicate": duplicate,
        "status": "COMPLETED" if value is not None else "NOT_RUN",
        "macro_f1": value,
        "top1": value,
        "balanced_accuracy": value,
        "top5": None,
    }


def test_duplicates_count_once_and_sd_is_the_sample_sd():
    rows = [_row("scratch", s, v) for s, v in zip((11, 22, 33), (0.2, 0.4, 0.6), strict=True)]
    rows += [_row("prototypes", 11, 0.7)]
    rows += [_row("prototypes", s, 0.7, source="prototypes-11", duplicate=True) for s in (22, 33)]
    cells = {a["method"]: a for a in aggregate(rows, "macro_f1")}
    assert cells["scratch"]["n_distinct"] == 3
    assert cells["scratch"]["mean"] == pytest.approx(0.4)
    assert cells["scratch"]["sd"] == pytest.approx(0.2)  # ddof = 1
    assert cells["prototypes"]["n_slots"] == 3 and cells["prototypes"]["n_distinct"] == 1
    assert cells["prototypes"]["duplicates"] == 2 and cells["prototypes"]["sd"] is None


def test_missing_values_stay_null_and_are_not_averaged():
    rows = [_row("scratch", 11, None), _row("scratch", 22, None)]
    cell = aggregate(rows, "macro_f1")[0]
    assert cell["mean"] is None and cell["n_with_value"] == 0 and cell["n_distinct"] == 0


def test_paired_differences_use_matched_seeds_only():
    rows = [_row("scratch", 11, 0.2), _row("scratch", 22, 0.4), _row("scratch", 33, None)]
    rows += [_row("frozen", 11, 0.5), _row("frozen", 22, 0.5), _row("frozen", 33, 0.9)]
    pair = next(p for p in paired(rows, "macro_f1") if (p["a"], p["b"]) == ("frozen", "scratch"))
    assert pair["n_pairs"] == 2 and pair["seeds"] == [11, 22]
    assert pair["mean_diff"] == pytest.approx(0.2)
    assert pair["sd_diff"] == pytest.approx(math.sqrt(0.02))
    assert pair["a_better"] == 2 and pair["ties"] == 0
