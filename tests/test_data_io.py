"""Tests for the train/val/test split logic in DataLoader."""

import pandas as pd

from data_io import DataLoader


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"text": [f"statement {i}" for i in range(n)], "label": [i % 2 for i in range(n)]})


def test_create_splits_uses_existing_split_column():
    df = _frame(30).assign(split=["train"] * 20 + ["val"] * 5 + ["test"] * 5)
    train, val, test = DataLoader().create_splits(df)
    assert (len(train), len(val), len(test)) == (20, 5, 5)
    assert set(train["split"]) == {"train"}


def test_create_splits_is_stratified_disjoint_and_seeded():
    df = _frame(200)
    train, val, test = DataLoader().create_splits(df)

    assert (len(train), len(val), len(test)) == (160, 20, 20)
    assert not (set(train.index) & set(val.index))
    assert not (set(train.index) & set(test.index))
    assert not (set(val.index) & set(test.index))
    for part in (train, val, test):
        assert abs(part["label"].mean() - 0.5) < 0.05

    train_again, _, _ = DataLoader().create_splits(df)
    assert train_again.index.equals(train.index)
