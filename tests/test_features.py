"""Unit tests for FeatureExtractor: fit/transform contract, metadata stacking,
and the hash-checked pickle round trip."""

import hashlib
import logging

import pandas as pd
import pytest

import features
from features import FeatureExtractor, create_baseline_features

CORPUS = pd.DataFrame(
    {
        "text": [
            "the senator said taxes rose",
            "taxes rose again said the mayor",
            "a new study on taxes",
            "the mayor denied the study",
        ],
        "label": [0, 1, 0, 1],
    }
)


def small(**kwargs) -> FeatureExtractor:
    # min_df=1 so a four-document corpus produces a vocabulary.
    return FeatureExtractor(min_df=1, max_features=100, **kwargs)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        small().transform(CORPUS)


def test_fit_transform_shape_and_feature_names():
    fx = small()
    X = fx.fit_transform(CORPUS)
    assert X.shape[0] == len(CORPUS)
    assert X.shape[1] == fx.n_features_ == len(fx.get_feature_names())
    assert "taxes" in fx.text_vectorizer.vocabulary_


def test_fit_then_transform_matches_fit_transform():
    a = small().fit_transform(CORPUS)
    b = small().fit(CORPUS).transform(CORPUS)
    assert (a != b).nnz == 0


def test_prefers_processed_text_column_when_present():
    # Varied so no term exceeds max_df (a term in every document is dropped).
    df = CORPUS.assign(text_processed=["alpha beta", "alpha gamma", "beta gamma", "alpha delta"])
    fx = small().fit(df)
    vocab = set(fx.text_vectorizer.vocabulary_)
    assert {"alpha", "beta", "gamma", "delta"} <= vocab
    assert "senator" not in vocab and "taxes" not in vocab


def test_metadata_columns_are_appended_only_when_enabled():
    df = CORPUS.assign(text_length=[27, 31, 20, 26], exclamation_count=[0, 2, 0, 1])
    plain = small().fit_transform(df)
    with_meta = small(use_metadata=True).fit_transform(df)
    assert with_meta.shape[1] == plain.shape[1] + 2


def test_feature_names_match_matrix_width_with_metadata():
    df = CORPUS.assign(text_length=[27, 31, 20, 26], exclamation_count=[0, 2, 0, 1])
    fx = small(use_metadata=True)
    X = fx.fit_transform(df)
    names = fx.get_feature_names()
    assert len(names) == X.shape[1]
    assert names[-2:] == ["meta_text_length", "meta_exclamation_count"]
    # A frame missing a fitted metadata column must still produce the fitted width.
    with pytest.raises(KeyError):
        fx.transform(df.drop(columns=["exclamation_count"]))


def test_baseline_factory_returns_extractor():
    assert isinstance(create_baseline_features(), FeatureExtractor)


def test_save_and_load_round_trip(tmp_path):
    fx = small().fit(CORPUS)
    path = tmp_path / "fx.pkl"
    fx.save(path)

    back = FeatureExtractor.load(path)
    assert (back.transform(CORPUS) != fx.transform(CORPUS)).nnz == 0
    assert back.ngram_range == fx.ngram_range
    assert back.max_features == fx.max_features
    assert back.get_feature_names() == fx.get_feature_names()


def test_load_warns_when_file_is_not_hash_registered(tmp_path, caplog):
    fx = small().fit(CORPUS)
    path = tmp_path / "fx.pkl"
    fx.save(path)
    with caplog.at_level(logging.WARNING):
        FeatureExtractor.load(path)
    assert any("SHA256" in record.getMessage() for record in caplog.records)


def test_load_refuses_registered_file_with_wrong_hash(tmp_path, monkeypatch):
    fx = small().fit(CORPUS)
    path = tmp_path / "fx.pkl"
    fx.save(path)
    monkeypatch.setitem(features.MODEL_HASHES, path.name, "0" * 64)
    with pytest.raises(ValueError, match="Refusing to unpickle"):
        FeatureExtractor.load(path)


def test_load_accepts_registered_file_with_matching_hash(tmp_path, monkeypatch, caplog):
    fx = small().fit(CORPUS)
    path = tmp_path / "fx.pkl"
    fx.save(path)
    monkeypatch.setitem(features.MODEL_HASHES, path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    with caplog.at_level(logging.WARNING):
        back = FeatureExtractor.load(path)
    assert back.n_features_ == fx.n_features_
    assert not any("SHA256" in record.getMessage() for record in caplog.records)
