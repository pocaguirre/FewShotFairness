import sys

sys.path.append("../../src/")

import pytest

from src.datasets import BiasInBios


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

    train, test, demographics = dataset.create_prompts()
