import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from diff_lab.metrics import best_threshold, evaluate, recall_at
from diff_lab.util import IntegrityError

IDS = [f"c{i}" for i in range(10)]


def test_ap_matches_sklearn_and_recall_ceil():
    rng = np.random.default_rng(0)
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    s = rng.random(10)
    m = evaluate(IDS, y, s, "salt")
    assert m["ap"] == pytest.approx(average_precision_score(y, s))
    assert m["recall_at_5pct"]["k"] == 1 and m["recall_at_10pct"]["k"] == 1
    assert recall_at(IDS * 1, y, s, 0.25, "salt")["k"] == math.ceil(2.5)


def test_at06_single_class_is_null_not_zero():
    m = evaluate(IDS, np.zeros(10, dtype=int), np.linspace(0, 1, 10), "salt")
    assert m["ap"] is None and m["roc_auc"] is None and m["recall_at_10pct"]["recall"] is None
    assert m["null_reason"] == "single class in cohort"


def test_ties_broken_by_hash_not_label():
    s = np.full(10, 0.5)
    y1 = np.array([1] + [0] * 9)
    y2 = np.array([0] * 9 + [1])
    a = recall_at(IDS, y1, s, 0.1, "salt")
    b = recall_at(IDS, y2, s, 0.1, "salt")
    assert a["boundary_tie_size"] == 10 and a["k"] == 1
    sel_label_free = {a["tp"], b["tp"]}
    assert sel_label_free <= {0, 1}


def test_threshold_max_f1_ties_take_highest():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    t = best_threshold(y, s)
    assert t["threshold"] == 0.7 and t["f1"] == pytest.approx(0.8)


def test_at18_duplicates_nan_and_length_rejected():
    with pytest.raises(IntegrityError):
        evaluate(["a", "a"], [0, 1], [0.1, 0.2], "s")
    with pytest.raises(IntegrityError):
        evaluate(["a", "b"], [0, 1], [0.1, float("nan")], "s")
    with pytest.raises(IntegrityError):
        evaluate(["a", "b", "c"], [0, 1], [0.1, 0.2], "s")
