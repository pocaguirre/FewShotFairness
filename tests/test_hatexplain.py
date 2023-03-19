import pytest

from src.datasets.hatexplain import HatExplain


def test_good_path():
    path = "data/HateXplain"

    dataset = HatExplain(path)

    assert dataset is not None


def test_bad_path():
    path = "data/HatXplain"

    with pytest.raises(ValueError) as e_info:
        dataset = HatExplain(path)


def test_create_prompts():
    path = "data/HateXplain"

    dataset = HatExplain(path)

    train_prompts, test_prompts, test_labels, test_demographics = dataset.create_prompts()
