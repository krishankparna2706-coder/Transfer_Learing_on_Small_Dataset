"""P5: the review scorer and its validation-coverage threshold rule (hand-checked values)."""

from __future__ import annotations

import numpy as np
import pytest

from fewshotlab.simulation import review as R

SCORES = [0.03, 0.12, 0.22, 0.31, 0.47, 0.52, 0.66, 0.71, 0.88, 0.97]
TRUE = [0] * 10
PRED = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]  # wrong at the scores 0.03, 0.12 and 0.52


def test_threshold_grid_is_the_prescribed_one():
    assert len(R.THRESHOLD_GRID) == 21
    assert R.THRESHOLD_GRID[0] == 0.0 and R.THRESHOLD_GRID[-1] == 1.0
    assert all(
        round(b - a, 2) == 0.05
        for a, b in zip(R.THRESHOLD_GRID, R.THRESHOLD_GRID[1:], strict=False)
    )
    assert R.TARGET_COVERAGE == 0.80


def test_highest_threshold_keeping_80_percent_coverage():
    result = R.select_threshold(TRUE, PRED, SCORES)
    chosen = result["selected"]
    assert chosen["threshold"] == 0.2  # 8 of 10 scores are >= 0.20; only 7 are >= 0.25
    assert chosen["accepted"] == 8 and chosen["coverage"] == 0.8
    assert chosen["accepted_errors"] == 1 and chosen["accepted_error_rate"] == 0.125
    assert chosen["review_rate"] == pytest.approx(0.2)
    assert chosen["overall_error_rate"] == pytest.approx(0.3)
    assert next(r for r in result["curve"] if r["threshold"] == 0.25)["coverage"] == 0.7


def test_review_happens_strictly_below_the_threshold():
    assert R.review_flags([0.5, 0.49999, 0.6], 0.5).tolist() == [False, True, False]
    assert R.review_metrics([0, 0], [0, 0], [0.5, 0.5], 0.5)["coverage"] == 1.0


def test_accepted_error_is_null_when_nothing_is_accepted():
    m = R.review_metrics([0, 1], [0, 0], [0.3, 0.4], 1.0)
    assert m["accepted"] == 0 and m["coverage"] == 0.0
    assert m["accepted_error_rate"] is None


def test_coverage_never_increases_and_zero_always_qualifies():
    scores = np.random.default_rng(0).random(200)
    curve = R.coverage_curve(np.zeros(200, int), np.zeros(200, int), scores)
    coverage = [r["coverage"] for r in curve]
    assert coverage[0] == 1.0
    assert all(a >= b for a, b in zip(coverage, coverage[1:], strict=False))
    assert R.select_threshold([0] * 5, [0] * 5, [0.01] * 5)["selected"]["threshold"] == 0.0


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        R.review_metrics([0], [0, 1], [0.5], 0.5)
    with pytest.raises(ValueError):
        R.review_metrics([0], [0], [float("nan")], 0.5)
    with pytest.raises(ValueError):
        R.review_metrics([], [], [], 0.5)
