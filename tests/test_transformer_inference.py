"""Transformer scores used for stacking must come back in input order. Runs only
where torch and transformers are installed (CI installs the sklearn stack only).
A tiny randomly initialised DistilBERT is built locally, so nothing is downloaded."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from milestone3_transformer_model import FakeNewsTransformer  # noqa: E402

WORDS = ["the", "senate", "passed", "a", "bill", "taxes", "rose", "jobs", "fell", "claims", "false"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-distilbert")
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])}
    # transformers 5 ignores vocab_file=, which leaves every word [UNK].
    transformers.DistilBertTokenizer(vocab=vocab).save_pretrained(path)
    torch.manual_seed(0)
    config = transformers.DistilBertConfig(
        vocab_size=5 + len(WORDS), dim=16, hidden_dim=32, n_layers=1, n_heads=2,
        initializer_range=0.5,  # large random weights, so every text scores differently
    )
    transformers.DistilBertForSequenceClassification(config).save_pretrained(path)
    return path


def test_predict_proba_returns_scores_in_input_order(tiny_model_dir):
    model = FakeNewsTransformer(model_name=str(tiny_model_dir))
    texts = ["the senate passed a bill", "taxes rose", "jobs fell", "claims false",
             "the bill", "senate jobs", "a false claims", "taxes fell"] * 3

    batched = model.predict_proba(texts, batch_size=4)
    _, one_at_a_time = model.predict(texts)

    # Every distinct text scores differently, so any reordering would show.
    assert len(np.unique(one_at_a_time.round(6))) == len(set(texts))
    np.testing.assert_allclose(batched, one_at_a_time, atol=1e-5)


def test_attention_analysis_returns_token_weights(tiny_model_dir):
    pytest.importorskip("lime")
    pytest.importorskip("shap")
    from milestone3_explainability import FakeNewsExplainer

    # Skip __init__, which loads the tokenizer from the Hugging Face Hub.
    explainer = FakeNewsExplainer.__new__(FakeNewsExplainer)
    explainer.model = transformers.DistilBertForSequenceClassification.from_pretrained(tiny_model_dir)
    explainer.tokenizer = transformers.DistilBertTokenizer.from_pretrained(tiny_model_dir)
    explainer.device = torch.device("cpu")

    text = "the senate passed a bill"
    tokens = explainer.get_attention_analysis(text)

    # Weights for the sentence's own tokens, not [CLS], [SEP] or padding.
    assert tokens and all(1 <= t["position"] <= len(text.split()) for t in tokens)
