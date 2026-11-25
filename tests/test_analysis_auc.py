"""Regression tests for tie handling in the macro one-vs-rest AUC."""

import numpy as np
import pytest

from dlens.tools._analysis import _macro_auc


def _reference_macro_auc(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    """Brute-force AUC = P(pos > neg) + 0.5 * P(pos == neg), averaged over classes."""
    out = []
    for c in range(k):
        pos, neg = probs[y == c, c], probs[y != c, c]
        out.append((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return float(np.mean(out))


@pytest.mark.parametrize("shift", [0, 3, 5, 9])
def test_constant_scores_give_chance_auc_for_any_sample_order(shift):
    # A collapsed classifier outputs identical probabilities; its AUC is exactly 0.5.
    y = np.roll(np.array([0] * 2 + [1] * 6 + [2] * 4), shift)
    probs = np.full((len(y), 3), 1 / 3)
    assert _macro_auc(probs, y, 3) == pytest.approx(0.5)


def test_sorted_labels_do_not_change_auc_with_tied_scores():
    # Datasets are loaded class by class, so labels arrive sorted.
    y = np.array([0] * 2 + [1] * 6 + [2] * 4)
    probs = np.full((12, 3), 1 / 3)
    assert _macro_auc(probs, y, 3) == pytest.approx(_macro_auc(probs[::-1], y[::-1], 3))


def test_auc_is_invariant_to_shuffling_with_coarse_probabilities():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    logits = rng.normal(size=(300, 3)) + 1.5 * np.eye(3)[y]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    probs = np.round(probs, 1)  # many tied scores
    expected = _reference_macro_auc(probs, y, 3)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(y))
        assert _macro_auc(probs[perm], y[perm], 3) == pytest.approx(expected)


def test_untied_scores_are_unchanged():
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array([
        [0.7, 0.2, 0.1], [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2], [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7], [0.3, 0.1, 0.6],
    ])
    assert _macro_auc(probs, y, 3) == pytest.approx(_reference_macro_auc(probs, y, 3))
    assert _macro_auc(probs, y, 3) == pytest.approx(1.0)
