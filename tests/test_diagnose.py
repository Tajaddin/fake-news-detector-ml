"""The random-label check must shuffle only the training labels and score the
model against the true test labels; otherwise it reads about 50% on any data
and can never flag anything."""

import numpy as np

from diagnose_perfect_accuracy import shuffled_label_accuracy


def test_scores_against_the_true_test_labels():
    # 90% of training labels are positive, and a permutation keeps that, so the
    # model predicts "positive" almost everywhere. Scored on an all-positive
    # test set that is close to 1.0; random test labels would give about 0.5.
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(300, 3))
    y_train = (rng.random(300) < 0.9).astype(int)
    X_test = rng.normal(size=(200, 3))
    y_test = np.ones(200, dtype=int)

    assert shuffled_label_accuracy(X_train, y_train, X_test, y_test) > 0.85


def test_same_seed_gives_the_same_result():
    rng = np.random.default_rng(1)
    X_train, X_test = rng.normal(size=(200, 4)), rng.normal(size=(200, 4))
    y_train, y_test = rng.integers(0, 2, size=200), rng.integers(0, 2, size=200)

    by_seed = [shuffled_label_accuracy(X_train, y_train, X_test, y_test, seed=s) for s in range(10)]

    assert shuffled_label_accuracy(X_train, y_train, X_test, y_test, seed=3) == by_seed[3]
    assert len(set(by_seed)) > 1
