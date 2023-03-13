import pytest

from src.datasets.biasinbios import BiasInBios

def test_good_path():

    path = "data/biasbios"

    dataset = BiasInBios(path, "{text} \n The occupation of this person is {label}")

    assert dataset is not None

def test_bad_path():
    path = "data/bisbios"

    with pytest.raises(ValueError) as e_info:
        dataset = BiasInBios(path, "{text} \n The occupation of this person is {label}")

def test_create_prompts():
    path = "data/biasbios"

    dataset = BiasInBios(path, "{text} \n The occupation of this person is {label}")

    train_prompts, test_prompts, test_labels = dataset.create_prompts()