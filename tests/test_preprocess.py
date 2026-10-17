"""Unit tests for TextPreprocessor: the cleaning steps the whole pipeline relies on."""

import pandas as pd
import pytest

from preprocess import TextPreprocessor


@pytest.fixture
def pp():
    return TextPreprocessor()


def test_non_string_input_becomes_empty(pp):
    assert pp.process_text(None) == ""
    assert pp.process_text(12.5) == ""


def test_html_tags_and_entities_removed(pp):
    assert pp.process_text("<p>Tom &amp; Jerry</p>") == "tom & jerry"


def test_urls_and_emails_removed(pp):
    out = pp.process_text("Read https://example.com/story now, mail me@site.org")
    assert "http" not in out
    assert "@" not in out
    assert "read" in out


def test_mentions_removed_but_hashtags_kept(pp):
    out = pp.process_text("@user says #breaking news")
    assert "@user" not in out
    assert "#breaking" in out


def test_contractions_expanded(pp):
    assert pp.process_text("They can't and won't") == "they cannot and will not"


def test_unicode_folded_to_ascii(pp):
    assert pp.process_text("Café naïve") == "cafe naive"


def test_whitespace_normalized_and_lowercased(pp):
    assert pp.process_text("  Many   SPACES\n\there ") == "many spaces here"


def test_options_are_respected():
    keep = TextPreprocessor(lowercase=False, remove_urls=False)
    assert keep.process_text("Visit www.a.com NOW") == "Visit www.a.com NOW"

    truncated = TextPreprocessor(max_length=5)
    assert truncated.process_text("abcdefghij") == "abcde"

    no_numbers = TextPreprocessor(remove_numbers=True)
    assert no_numbers.process_text("in 2024 we") == "in we"


def test_extract_features_counts_surface_signals(pp):
    df = pd.DataFrame(
        {
            "text": ["WOW!!! Really? http://x.co @me #tag 123", "plain words"],
            "word_count": [7, 2],
        }
    )
    out = pp.extract_features(df)
    loud, plain = out.iloc[0], out.iloc[1]

    assert loud["exclamation_count"] == 3
    assert loud["question_count"] == 1
    assert loud["url_count"] == 1
    assert loud["mention_count"] == 1
    assert loud["hashtag_count"] == 1
    assert loud["number_count"] == 1
    assert 0 < loud["capital_ratio"] < 1

    assert plain["exclamation_count"] == 0
    assert plain["url_count"] == 0
    # The input frame must not be mutated.
    assert "exclamation_count" not in df.columns
