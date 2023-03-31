import pytest

from src.datasets.twitteraae import TwitterAAE


def test_good_path():
    path = "data/moji/twitteraae_sentiment_race"

    dataset = TwitterAAE(path)

    assert dataset is not None


def test_bad_path():
    path = "data/moji/twitteraae_sentment_race"

    with pytest.raises(ValueError) as e_info:
        dataset = TwitterAAE(path)


def test_create_prompts():
    path = "data/moji/twitteraae_sentiment_race"

    dataset = TwitterAAE(path)

    train, test, demographics =  dataset.create_prompts()
