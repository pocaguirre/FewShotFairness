import sys 

sys.path.append("../../src/")

import pytest

import pandas as pd

from src.demonstrations.random import RandomSampler


@pytest.fixture
def test_data():
    train_data = ["train" + str(x) for x in range(0, 32)]
    train_demographics = ["a"] * 32

    train_df = pd.DataFrame(
        {"prompts": train_data, "demographics": train_demographics}
    )

    test_data = ["test" + str(x) for x in range(0, 4)]
    test_demographics = ["test" + str(x) for x in range(0, 4)]
    test_labels = ["test" + str(x) for x in range(0, 4)]

    test_df = pd.DataFrame(
        {
            "prompts": test_data,
            "demographics": test_demographics,
            "labels": test_labels,
        }
    )

    return train_df, test_df

def test_zeroshot_sampler(test_data):
    
    train_df, test_df = test_data

    rs = RandomSampler(shots=0)

    output = rs.create_demonstrations(train_df, test_df, ["a"])

    assert len(output) == 4


def test_fourshot_sampler(test_data):
    train_df, test_df = test_data

    rs = RandomSampler(shots=4)

    output = rs.create_demonstrations(train_df, test_df, ["a"])

    assert len(output) == 4


def test_16shot_sampler(test_data):
    train_df, test_df = test_data

    rs = RandomSampler(shots=16)

    output = rs.create_demonstrations(train_df, test_df, ["a"])

    assert len(output) == 4
