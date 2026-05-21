"""
Held-out evaluation of the TF-IDF + Logistic Regression baseline on the LIAR
test split. Self-contained (depends only on scikit-learn + pandas) so it runs
in CI without the transformer stack.

Reports accuracy, precision, recall, F1, and ROC-AUC on the canonical 1,267-row
LIAR test set. Unlike the leakage-prone ISOT corpus (which trivially hits
99-100%), LIAR is a real, hard short-statement task, so these numbers reflect
genuine signal.

    python src/eval_holdout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT = Path(__file__).resolve().parent.parent / "experiments" / "holdout_baseline.json"


def _text(df: pd.DataFrame) -> pd.Series:
    # Statement plus its subject/context (standard LIAR metadata), no speaker
    # identity, to avoid speaker-as-label leakage.
    parts = [df["text"].fillna("")]
    if "context" in df.columns:
        parts.append(df["context"].fillna(""))
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.str.cat(p, sep=" ")
    return combined


def run() -> dict:
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(_text(train), train["label"])

    proba = pipe.predict_proba(_text(test))[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = test["label"].to_numpy()

    metrics = {
        "dataset": "LIAR (held-out test)",
        "test_samples": int(len(test)),
        "model": "tfidf(1-2gram) + logistic_regression",
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }
    return metrics


def main() -> int:
    metrics = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
