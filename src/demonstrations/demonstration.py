from typing import List, Tuple

import pandas as pd


class Demonstration:
    def __init__(self, shots: int = 16) -> None:
        """Base class for demonstrations

        :param shots: number of shots in demonstration, defaults to 16
        :type shots: int, optional
        """

        self.shots = shots

        self.type = "N/A"

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> Tuple[List[str], pd.DataFrame]:
        """_summary_

        :param train_df: train dataset
        :type train_df: pd.DataFrame
        :param test_df: test dataset
        :type test_df: pd.DataFrame
        :param overall_demographics: demographics that we are focusing on
        :type overall_demographics: List[str]
        :return: demonstrations for test set
        :rtype: Tuple[List[str], pd.DataFrame]
        """
        pass
