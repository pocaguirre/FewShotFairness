import sys

sys.path.append("../../src/")

import pytest

from src.datasets import HateXplainGender
from src.datasets import HateXplainRace


def test_good_path_gender():
    path = "data/HateXplain"

    dataset = HateXplainGender(path)

    assert dataset is not None


def test_bad_path_gender():
    path = "data/HatXplain"

    with pytest.raises(ValueError) as e_info:
        dataset = HateXplainGender(path)


def test_create_prompts_gender():
    path = "data/HateXplain"

    dataset = HateXplainGender(path)

    train, test, demographics = dataset.create_prompts()


def test_good_path_race():
    path = "data/HateXplain"

    dataset = HateXplainRace(path)

    assert dataset is not None


def test_bad_path_race():
    path = "data/HatXplain"

    with pytest.raises(ValueError) as e_info:
        dataset = HateXplainRace(path)


def test_create_prompts_race():
    path = "data/HateXplain"

    dataset = HateXplainRace(path)

    train, test, demographics = dataset.create_prompts()
