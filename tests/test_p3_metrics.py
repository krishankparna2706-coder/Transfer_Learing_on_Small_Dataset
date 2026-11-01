"""P3: metric definitions checked against hand calculations and scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from fewshotlab.errors import DataValidationError
from fewshotlab.metrics.core import check_prediction_ids, classification_metrics, topk_accuracy


def test_hand_calculated_example_from_the_protocol():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert m["top1"] == 0.75
    assert m["macro_f1"] == pytest.approx(11 / 15, abs=1e-15)
    assert m["confusion"] == [[1, 1], [0, 2]]
    first, second = m["per_class"]
    assert (first["precision"], first["recall"]) == (1.0, 0.5)
    assert second["precision"] == pytest.approx(2 / 3) and second["recall"] == 1.0
    assert first["f1"] == pytest.approx(2 / 3) and second["f1"] == pytest.approx(4 / 5)
    assert m["balanced_accuracy"] == pytest.approx(0.75)


def test_never_predicted_class_uses_the_zero_convention():
    m = classification_metrics([0, 1, 2, 2], [0, 1, 1, 1], 3)
    third = m["per_class"][2]
    assert (third["precision"], third["recall"], third["f1"], third["predicted"]) == (0, 0, 0, 0)
    assert m["macro_f1"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_class_absent_from_truth_and_predictions_still_counts():
    m = classification_metrics([0, 1], [0, 1], 3)
    assert m["macro_f1"] == pytest.approx(2 / 3)  # class 2 contributes F1 = 0
    assert m["confusion_row_normalized"][2] is None  # empty row is null, not zeros or NaN


def test_every_prediction_wrong():
    m = classification_metrics([0, 1], [1, 0], 2)
    assert m["top1"] == 0.0 and m["macro_f1"] == 0.0


def test_agrees_with_scikit_learn():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 7, 500)
    pred = np.where(rng.random(500) < 0.6, true, rng.integers(0, 7, 500))
    m = classification_metrics(true, pred, 7)
    labels = list(range(7))
    assert m["macro_f1"] == pytest.approx(
        skm.f1_score(true, pred, labels=labels, average="macro", zero_division=0), abs=1e-12
    )
    assert m["top1"] == pytest.approx(skm.accuracy_score(true, pred), abs=1e-12)
    assert m["balanced_accuracy"] == pytest.approx(
        skm.balanced_accuracy_score(true, pred), abs=1e-12
    )
    p, r, f, s = skm.precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)
    for c in labels:
        row = m["per_class"][c]
        assert (row["precision"], row["recall"], row["f1"], row["support"]) == pytest.approx(
            (p[c], r[c], f[c], s[c]), abs=1e-12
        )
    assert m["confusion"] == skm.confusion_matrix(true, pred, labels=labels).tolist()


def test_row_normalization_and_denominators():
    m = classification_metrics([0, 0, 1], [0, 1, 1], 3)
    rows = m["confusion_row_normalized"]
    assert rows[0] == [0.5, 0.5, 0.0] and rows[1] == [0.0, 1.0, 0.0] and rows[2] is None
    assert sum(sum(r) for r in m["confusion"]) == m["n_images"] == 3
    assert all(0.0 <= m[k] <= 1.0 for k in ("top1", "macro_f1", "balanced_accuracy"))


def test_jointly_permuting_rows_changes_nothing():
    true, pred = np.array([0, 1, 2, 2, 1]), np.array([0, 2, 2, 1, 1])
    order = np.array([4, 2, 0, 3, 1])
    assert classification_metrics(true, pred, 3) == classification_metrics(
        true[order], pred[order], 3
    )


def test_missing_or_repeated_ids_fail():
    with pytest.raises(DataValidationError, match="duplicate"):
        check_prediction_ids(["a", "a"])
    with pytest.raises(DataValidationError, match="missing"):
        check_prediction_ids(["a"], expected=["a", "b"])
    with pytest.raises(DataValidationError, match="unexpected"):
        check_prediction_ids(["a", "c"], expected=["a"])


def test_topk_ties_go_to_the_lower_class_id():
    scores = np.array([[0.5, 0.5, 0.0]])
    assert topk_accuracy([1], scores, 1) == 0.0  # class 0 ranks first on the tie
    assert topk_accuracy([1], scores, 2) == 1.0


def test_out_of_range_labels_fail():
    with pytest.raises(DataValidationError):
        classification_metrics([0, 3], [0, 1], 3)
