"""The leak check must catch test rows that differ from a training row only in
digits or punctuation. The shuffled-label check is a null baseline for the
real-label score: the same model refit on many permutations of the training
labels and scored on the true test labels. A model with real signal clears the
95th percentile of those scores; one trained on noise labels does so only about
one time in twenty."""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from diagnose_perfect_accuracy import near_duplicates, shuffled_label_baseline

MODEL = LogisticRegression()


def _split(seed, signal):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] > 0).astype(int) if signal else rng.integers(0, 2, size=400)
    return X[:300], y[:300], X[300:], y[300:]


def test_near_duplicates_ignore_case_digits_and_punctuation():
    train = ["Miracle cure found! Additional context 12.", "Budget approved.", "2024"]
    test = ["miracle cure found additional context 87", "Budget approved for roads.", "Budget approved", "99"]

    assert near_duplicates(train, test) == ["miracle cure found additional context 87", "Budget approved"]


def _clears_baseline(X_train, y_train, X_test, y_test):
    real = clone(MODEL).fit(X_train, y_train).score(X_test, y_test)
    return real > np.percentile(shuffled_label_baseline(MODEL, X_train, y_train, X_test, y_test), 95)


def test_scores_against_the_true_test_labels():
    # 90% of training labels are positive, and a permutation keeps that, so each
    # model predicts "positive" almost everywhere. Scored on an all-positive
    # test set that is close to 1.0; random test labels would give about 0.5.
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(300, 3))
    y_train = (rng.random(300) < 0.9).astype(int)
    X_test = rng.normal(size=(200, 3))
    y_test = np.ones(200, dtype=int)

    assert shuffled_label_baseline(MODEL, X_train, y_train, X_test, y_test).mean() > 0.85


def test_real_signal_clears_the_baseline():
    split = _split(seed=0, signal=True)

    assert abs(shuffled_label_baseline(MODEL, *split).mean() - 0.5) < 0.05
    assert _clears_baseline(*split)


def test_noise_labels_rarely_clear_the_baseline():
    # One permutation, or the mean of many, would be cleared about half the time.
    cleared = sum(_clears_baseline(*_split(seed, signal=False)) for seed in range(20))

    assert cleared <= 3


def test_each_permutation_is_new_and_the_seed_fixes_them():
    split = _split(seed=1, signal=True)
    scores = shuffled_label_baseline(MODEL, *split, seed=3)

    assert len(set(scores)) > 1
    np.testing.assert_array_equal(scores, shuffled_label_baseline(MODEL, *split, seed=3))
