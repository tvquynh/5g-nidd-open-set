"""Column index is not a class code when the training labels are not 0..K-1.

The held-out split trains on Benign plus five of the eight attack types. Global
codes are assigned alphabetically over all nine classes, so the six training
codes are [0, 1, 2, 4, 7, 8] and any classifier that emits one column per
training class has column 3 -> class 4, column 4 -> class 7, column 5 -> class 8.
Reading the column index as the label silently renames those three, and two of
the names it invents (class 3 SYNFlood, class 5 SlowrateDoS) are held-out
classes the model cannot predict at all.

These tests pin the contract so the bug cannot return.
"""
import numpy as np
import pytest

from src.open_set import detect_open_set
from src.splits import HOLDOUT_TRAIN_ATTACKS, HOLDOUT_TEST_ATTACKS

# The nine 5G-NIDD classes, alphabetically, which is how preprocess codes them.
CLASSES = sorted(["Benign"] + HOLDOUT_TRAIN_ATTACKS + HOLDOUT_TEST_ATTACKS)
CODE = {c: i for i, c in enumerate(CLASSES)}
TRAIN_CODES = sorted(CODE[c] for c in ["Benign"] + HOLDOUT_TRAIN_ATTACKS)


def test_training_codes_are_not_contiguous():
    """The premise of the whole problem: the trained codes skip 3, 5 and 6."""
    assert TRAIN_CODES == [0, 1, 2, 4, 7, 8]
    held_out = sorted(CODE[c] for c in HOLDOUT_TEST_ATTACKS)
    assert held_out == [3, 5, 6]
    assert set(TRAIN_CODES).isdisjoint(held_out)


def test_column_index_differs_from_class_code_from_column_three():
    order = np.array(TRAIN_CODES)
    assert list(order[:3]) == [0, 1, 2]        # coincide, which hides the bug
    assert order[3] == 4 and order[4] == 7 and order[5] == 8


def test_argmax_without_translation_invents_held_out_classes():
    """Reading the column index as a label produces classes the model lacks."""
    order = np.array(TRAIN_CODES)
    proba = np.eye(6)                          # one sample per column
    naive = np.argmax(proba, axis=1)           # the bug
    correct = order[np.argmax(proba, axis=1)]  # the contract

    held_out = {3, 5, 6}
    assert held_out & set(naive.tolist()) == {3, 5}
    assert not (held_out & set(correct.tolist()))
    assert set(correct.tolist()) == set(TRAIN_CODES)


@pytest.mark.parametrize("method", ["msp", "energy", "mahalanobis", "knn"])
def test_detect_open_set_translates_predictions(method):
    """Every rule must return labels in the global code space."""
    rng = np.random.default_rng(0)
    order = np.array(TRAIN_CODES)

    logits_train = rng.normal(size=(400, 6)) * 2.0
    proba_train = np.exp(logits_train)
    proba_train /= proba_train.sum(1, keepdims=True)
    y_train = order[np.argmax(proba_train, axis=1)]

    logits_test = rng.normal(size=(120, 6)) * 2.0
    proba_test = np.exp(logits_test)
    proba_test /= proba_test.sum(1, keepdims=True)

    result = detect_open_set(method, proba_test, proba_train=proba_train,
                             y_train=y_train, threshold_quantile=0.95,
                             class_order=order)
    kept = result.pred[result.pred != -1]
    assert set(kept.tolist()) <= set(TRAIN_CODES), (
        f"{method} emitted labels outside the trained class codes: "
        f"{sorted(set(kept.tolist()) - set(TRAIN_CODES))}")


def test_rejection_mask_is_unaffected_by_the_translation():
    """The fix renames classes; it must not move the reject/keep decision."""
    rng = np.random.default_rng(1)
    order = np.array(TRAIN_CODES)
    proba_train = rng.dirichlet(np.ones(6), size=300)
    proba_test = rng.dirichlet(np.ones(6), size=100)
    y_train = order[np.argmax(proba_train, axis=1)]

    with_order = detect_open_set("msp", proba_test, proba_train=proba_train,
                                 y_train=y_train, class_order=order)
    without = detect_open_set("msp", proba_test, proba_train=proba_train,
                              y_train=y_train, class_order=None)
    np.testing.assert_array_equal(with_order.pred == -1, without.pred == -1)
    assert with_order.threshold == without.threshold


# --- Proposition 1 needs the order-statistic quantile ----------------------

def test_interpolated_quantile_breaks_the_msp_energy_equivalence():
    """The counterexample behind the proposition's quantile convention.

    MSP and -log p_max rank identically, but each calibrated by its own
    linearly interpolated empirical quantile they can disagree on a flow.
    """
    cal = np.array([[0.9, 0.1], [0.6, 0.4]])
    test = np.array([0.613, 0.387])

    msp_cal, msp_t = 1 - cal.max(1), 1 - test.max()
    log_cal, log_t = -np.log(cal.max(1)), -np.log(test.max())

    msp_flag = msp_t > np.quantile(msp_cal, 0.95, method="linear")
    log_flag = log_t > np.quantile(log_cal, 0.95, method="linear")
    assert msp_flag and not log_flag, "the counterexample no longer separates them"

    # With the order-statistic convention the proposition holds.
    msp_flag = msp_t > np.quantile(msp_cal, 0.95, method="inverted_cdf")
    log_flag = log_t > np.quantile(log_cal, 0.95, method="inverted_cdf")
    assert msp_flag == log_flag


def test_monotone_transform_preserves_ranking_regardless():
    """What the proof really rests on: order, not the quantile convention."""
    rng = np.random.default_rng(7)
    pmax = rng.uniform(0.2, 0.999, size=500)
    msp, energy = 1 - pmax, -np.log(pmax)
    np.testing.assert_array_equal(np.argsort(msp), np.argsort(energy))
