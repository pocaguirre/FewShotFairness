import pytest

from src.datasets.sbic import SBIC

def test_good_path():

    path = "data/SBIC"

    dataset = SBIC(path, "{text} \n {label}")

    assert dataset is not None

def test_bad_path():
    path = "data/SBI"

    with pytest.raises(ValueError) as e_info:
        dataset = SBIC(path, "{text} \n {label}")

def test_create_prompts():
    path = "data/SBIC"

    dataset = SBIC(path, "{text} \n {label}")

    train_prompts, test_prompts, test_labels = dataset.create_prompts()