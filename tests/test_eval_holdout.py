"""Tests for the held-out LIAR evaluation, including a regression check on the
reported 62.1% accuracy / 0.676 ROC-AUC result."""

import pandas as pd
import pytest

import eval_holdout


def test_text_combines_statement_and_context_without_speaker():
    df = pd.DataFrame({"text": ["a", None], "context": ["ctx", "c2"], "speaker": ["alice", "bob"]})
    out = eval_holdout._text(df).tolist()
    assert out == ["a ctx", " c2"]
    assert all("alice" not in s and "bob" not in s for s in out)


def test_text_without_context_column():
    assert eval_holdout._text(pd.DataFrame({"text": ["only"]})).tolist() == ["only"]


@pytest.mark.slow
def test_holdout_baseline_reproduces_reported_result():
    if not (eval_holdout.PROCESSED / "test.csv").exists():
        pytest.skip("processed LIAR split not present")
    metrics = eval_holdout.run()
    assert metrics["test_samples"] == 1267
    # The README reports 62.1% accuracy and 0.676 ROC-AUC on this split.
    assert abs(metrics["accuracy"] - 0.6212) < 0.01
    assert abs(metrics["roc_auc"] - 0.6763) < 0.01
    assert 0.5 < metrics["f1"] < 0.65
