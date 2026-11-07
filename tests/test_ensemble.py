"""Stacking must train its meta-learner on scores the base models produced for
rows they never saw, and keep every score paired with its own row's label."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from milestone3_ensemble import FakeNewsEnsemble, out_of_fold_proba, out_of_fold_text_proba


def _noise(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 5)), rng.integers(0, 2, size=n)


def _memorizer(X, y):
    # 1-nearest-neighbour reproduces its training labels exactly, so scored
    # in-sample it looks like a perfect feature even when the labels are noise.
    return KNeighborsClassifier(n_neighbors=1).fit(X, y)


def test_meta_learner_does_not_trust_a_model_that_memorized_noise():
    X_train, y_train = _noise(200, seed=0)
    X_val, y_val = _noise(100, seed=1)
    ensemble = FakeNewsEnsemble()
    ensemble.add_base_model("memorizer", _memorizer(X_train, y_train))

    ensemble.train_stacking_ensemble(X_train, y_train, X_val, y_val)

    assert abs(ensemble.meta_model.coef_[0][0]) < 1.0


def test_transformer_scores_stay_paired_with_their_labels():
    X_train, y_train = _noise(200, seed=0)
    X_val, y_val = _noise(100, seed=1)
    ensemble = FakeNewsEnsemble()
    ensemble.add_base_model("memorizer", _memorizer(X_train, y_train))

    # A transformer that knows the label helps only if score i meets label i.
    ensemble.train_stacking_ensemble(
        X_train, y_train, X_val, y_val,
        transformer_train_oof=0.1 + 0.8 * y_train,
        transformer_val=0.1 + 0.8 * y_val,
    )
    ensemble.transformer_probabilities = 0.1 + 0.8 * y_val
    preds, _ = ensemble.predict_ensemble(X_val, strategy="stacking")

    assert (preds == y_val).all()


def test_out_of_fold_proba_scores_each_row_with_a_model_that_never_saw_it():
    labels = np.array([0, 1] * 10)
    training_rows_of_scorer = {}

    def fit_predict(train_idx, held_idx):
        for i in held_idx:
            training_rows_of_scorer[i] = set(train_idx)
        return held_idx / 100.0

    scores = out_of_fold_proba(fit_predict, labels, n_splits=5)

    np.testing.assert_allclose(scores, np.arange(20) / 100.0)
    assert sorted(training_rows_of_scorer) == list(range(20))
    assert all(i not in rows for i, rows in training_rows_of_scorer.items())


def test_text_model_scores_land_on_their_own_rows():
    # milestone3_main fine-tunes DistilBERT through this; every text carries
    # its own score, so a score on the wrong row would show.
    texts = [f"statement {i}" for i in range(30)]
    labels = np.array([0, 1, 1] * 10)
    held_out_texts_seen_in_training = []

    def fit_score(train_texts, train_labels, held_texts):
        assert [labels[texts.index(t)] for t in train_texts] == list(train_labels)
        held_out_texts_seen_in_training.extend(t for t in held_texts if t in train_texts)
        return np.array([texts.index(t) / 100.0 for t in held_texts])

    scores = out_of_fold_text_proba(fit_score, texts, labels, n_splits=5)

    np.testing.assert_allclose(scores, np.arange(30) / 100.0)
    assert held_out_texts_seen_in_training == []
