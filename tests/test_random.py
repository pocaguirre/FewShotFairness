from src.demonstrations.random import RandomSampler


def test_zeroshot_sampler():
    train_data = ["train" + str(x) for x in range(0, 32)]

    test_data = ["test" + str(x) for x in range(0, 4)]

    rs = RandomSampler(shots=0)

    output = rs.create_demonstrations(train_data, test_data)

    assert test_data == output


def test_fourshot_sampler():
    train_data = ["train" + str(x) for x in range(0, 32)]

    test_data = ["test" + str(x) for x in range(0, 4)]

    rs = RandomSampler(shots=4)

    output = rs.create_demonstrations(train_data, test_data)

    assert len(output) == 4


def test_16shot_sampler():
    train_data = ["train" + str(x) for x in range(0, 32)]

    test_data = ["test" + str(x) for x in range(0, 4)]

    rs = RandomSampler(shots=16)

    output = rs.create_demonstrations(train_data, test_data)

    assert len(output) == 4
