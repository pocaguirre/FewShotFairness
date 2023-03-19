import pytest

from src.datasets.biasinbios import BiasInBios


def test_good_path():
    path = "data/biasbios"

    dataset = BiasInBios(path)

    assert dataset is not None


def test_bad_path():
    path = "data/bisbios"

    with pytest.raises(ValueError) as e_info:
        dataset = BiasInBios(path)


def test_create_prompts():
    path = "data/biasbios"

    dataset = BiasInBios(path)

    train_prompts, test_prompts, test_labels, test_demographics = dataset.create_prompts()
