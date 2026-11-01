"""Metric definitions for protocol v1, computed only from stored predictions.

All metrics are computed over the C declared classes. An undefined class term is 0
(``zero_division = 0``):

    top1               = correct / N
    precision_c        = TP_c / (TP_c + FP_c)
    recall_c           = TP_c / (TP_c + FN_c)
    F1_c               = 2 TP_c / (2 TP_c + FP_c + FN_c)
    macro_F1           = mean over all classes of F1_c        (primary endpoint)
    balanced_accuracy  = mean over all classes of recall_c

Top-k ranks classes by score, descending; ties go to the lower class ID. Fractions stay in [0, 1].
Formatting as percentages happens only in presentation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from fewshotlab.errors import DataValidationError

ZERO_DIVISION = 0.0


def check_prediction_ids(ids: Sequence[str], expected: Sequence[str] | None = None) -> None:
    """Every image is scored exactly once, and (if given) exactly the expected set is scored."""
    duplicates = [i for i, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise DataValidationError(
            f"duplicate prediction IDs {duplicates[:5]}; an image must be scored exactly once"
        )
    if expected is not None:
        missing = sorted(set(expected) - set(ids))
        extra = sorted(set(ids) - set(expected))
        if missing or extra:
            raise DataValidationError(
                f"prediction IDs do not match the expected set (missing {len(missing)}: "
                f"{missing[:5]}; unexpected {len(extra)}: {extra[:5]}); a subset is never scored"
            )


def confusion_matrix(true: Sequence[int], pred: Sequence[int], num_classes: int) -> np.ndarray:
    """C x C counts with true classes on rows and predicted classes on columns."""
    true_a = np.asarray(true, dtype=np.int64)
    pred_a = np.asarray(pred, dtype=np.int64)
    if true_a.ndim != 1 or true_a.shape != pred_a.shape:
        raise DataValidationError("true and predicted labels must be 1-D and equally long")
    for name, values in (("true", true_a), ("predicted", pred_a)):
        if values.size and (values.min() < 0 or values.max() >= num_classes):
            raise DataValidationError(f"{name} labels outside 0..{num_classes - 1}")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true_a, pred_a), 1)
    return matrix


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, ZERO_DIVISION, dtype=np.float64)
    defined = denominator > 0
    out[defined] = numerator[defined] / denominator[defined]
    return out


def topk_accuracy(true: Sequence[int], scores: np.ndarray, k: int) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or k > scores.shape[1]:
        raise DataValidationError(f"top-{k} needs an [N, C] score matrix with C >= {k}")
    if not np.isfinite(scores).all():
        raise DataValidationError("score vectors must be finite")
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :k]  # stable: lower class ID first
    hits = [int(t) in row for t, row in zip(np.asarray(true), ranked, strict=True)]
    return float(np.mean(hits))


def classification_metrics(
    true: Sequence[int],
    pred: Sequence[int],
    num_classes: int,
    *,
    scores: np.ndarray | None = None,
    topk: Sequence[int] = (),
) -> dict[str, Any]:
    true_a = np.asarray(true, dtype=np.int64)
    pred_a = np.asarray(pred, dtype=np.int64)
    if true_a.size == 0:
        raise DataValidationError("no predictions to score")
    matrix = confusion_matrix(true_a, pred_a, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    support = matrix.sum(axis=1)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    row_normalized = [
        (row / row.sum()).tolist() if row.sum() > 0 else None for row in matrix.astype(np.float64)
    ]
    result: dict[str, Any] = {
        "n_images": int(true_a.size),
        "num_classes": int(num_classes),
        "zero_division": 0,
        "top1": float(np.mean(true_a == pred_a)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class": [
            {
                "class_id": c,
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
                "predicted": int(matrix[:, c].sum()),
            }
            for c in range(num_classes)
        ],
        "confusion": matrix.tolist(),
        "confusion_row_normalized": row_normalized,
    }
    for k in topk:
        result[f"top{k}"] = topk_accuracy(true_a, scores, k) if scores is not None else None
    return result
