import random

from typing import List, Iterable

import pandas as pd

from tqdm import tqdm

from .demonstration import Demonstration


class RandomSampler(Demonstration):
    def __init__(self, shots=16) -> None:
        super().__init__(shots)

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> List[str]:
        """Create random k-shot from train set and test set

        :param train_df: train data
        :type train_df: pd.DataFrame
        :param test_df: test data
        :type test_df: pd.DataFrame
        :param overall_demographics: demographics to focus on
        :type overall_demographics: List[str]
        :return: k-shot demonstrations for each test set item
        :rtype: List[str]
        """
        demonstrations = []

        for row in tqdm(test_df.itertuples()):
            train_dems = train_df["prompts"].sample(n=self.shots).tolist()

            if len(train_dems) != 0:
                demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)
            else:
                demonstrations.append(row.prompts)

        return demonstrations