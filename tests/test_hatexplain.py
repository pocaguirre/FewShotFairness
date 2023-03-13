import pytest

from src.datasets.hatexplain import Hatexplain

def test_good_path():

    path = "data/HateXplain"

    dataset = Hatexplain(path, "{text} \n {label}")

    assert dataset is not None

def test_bad_path():
    path = "data/HatXplain"

    with pytest.raises(ValueError) as e_info:
        dataset = Hatexplain(path, "{text} \n {label}")

def test_create_prompts():
    path = "data/HateXplain"

    dataset = Hatexplain(path, "{text} \n {label}")

    train_prompts, test_prompts, test_labels = dataset.create_prompts()