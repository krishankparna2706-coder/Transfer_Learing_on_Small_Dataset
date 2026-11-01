"""Review routing for the demonstration: accept a prediction or send it to human review.

The score is the model's maximum class score. That is the softmax of the logits for learned
methods, and softmax(-squared distance) for prototypes. It is an uncalibrated model score, not a
probability of being correct.
* Coverage is the fraction of images whose maximum score is at least the threshold.
* An image whose score is strictly below the threshold is flagged for review.
* The demo threshold is the highest value in {0.00, 0.05, ..., 1.00} that keeps at least 80 %
  validation coverage. The 80 % is a chosen workflow setting, not a safety target.
* When nothing is accepted, the accepted-error rate is null: it is never reported as 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))
TARGET_COVERAGE = 0.80


def _arrays(true, pred, max_scores):
    true_a = np.asarray(true, dtype=np.int64)
    pred_a = np.asarray(pred, dtype=np.int64)
    scores = np.asarray(max_scores, dtype=np.float64)
    if not (true_a.shape == pred_a.shape == scores.shape) or true_a.ndim != 1 or not len(true_a):
        raise ValueError("true, pred and max_scores must be equally long, non-empty 1-D arrays")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return true_a, pred_a, scores


def review_metrics(true, pred, max_scores, threshold: float) -> dict[str, Any]:
    true_a, pred_a, scores = _arrays(true, pred, max_scores)
    accepted = scores >= threshold
    n_accepted = int(accepted.sum())
    errors = int((true_a[accepted] != pred_a[accepted]).sum())
    return {
        "threshold": float(threshold),
        "n_images": int(len(true_a)),
        "accepted": n_accepted,
        "reviewed": int(len(true_a) - n_accepted),
        "coverage": n_accepted / len(true_a),
        "review_rate": 1.0 - n_accepted / len(true_a),
        "accepted_errors": errors,
        "accepted_error_rate": (errors / n_accepted) if n_accepted else None,
        "overall_error_rate": float((true_a != pred_a).mean()),
    }


def coverage_curve(true, pred, max_scores) -> list[dict[str, Any]]:
    return [review_metrics(true, pred, max_scores, t) for t in THRESHOLD_GRID]


def select_threshold(true, pred, max_scores) -> dict[str, Any]:
    """The highest grid threshold with coverage >= 80 % (threshold 0 always qualifies)."""
    curve = coverage_curve(true, pred, max_scores)
    eligible = [row for row in curve if row["coverage"] >= TARGET_COVERAGE]
    chosen = max(eligible, key=lambda row: row["threshold"])
    return {
        "rule": "highest threshold in {0.00, 0.05, ..., 1.00} with validation coverage "
        ">= 0.80; review when the maximum score is strictly below it",
        "selected": chosen,
        "curve": curve,
    }


def review_flags(max_scores, threshold: float) -> np.ndarray:
    return np.asarray(max_scores, dtype=np.float64) < threshold
