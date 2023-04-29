import math

from typing import List, Tuple

import pandas as pd

from tqdm import tqdm

from .demographicdemonstration import DemographicDemonstration


class StratifiedSampler(DemographicDemonstration):
    def __init__(self, shots: int = 16) -> None:
        """Stratified demonstration initalization

        :param shots: number of shots in demonstration, defaults to 16
        :type shots: int, optional
        """
        super().__init__(shots)

    def stratified_sample_df(
        self, df: pd.DataFrame, col: str, n_samples: int, number_of_demographics: int
    ) -> pd.DataFrame:
        """Sample stratified from dataframe by demographic

        :param df: dataframe to sample from
        :type df: pd.DataFrame
        :param col: column to stratify on
        :type col: str
        :param n_samples: how many samples to return
        :type n_samples: int
        :param number_of_demographics: number of classes to focus on
        :type number_of_demographics: int
        :return: _description_
        :rtype: pd.DataFrame
        """

        # groupby the column and then sample and sample an equal amount from each demographic
        df_ = df.groupby(col).apply(
            lambda x: x.sample(math.ceil(n_samples / number_of_demographics))
        )

        # clean up output
        df_.index = df_.index.droplevel(0)
        return df_

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> Tuple[List[str], pd.DataFrame]:
        """Creates demonstrations for test set using stratified sampling of train

        :param train_df: train data
        :type train_df: pd.DataFrame
        :param test_df: test data
        :type test_df: pd.DataFrame
        :param overall_demographics: demographics to focus on
        :type overall_demographics: List[str]
        :return: demonstrations for test set with stratified demographics
        :rtype: Tuple[List[str], pd.DataFrame]
        """
        set_of_overall_demographics = set(overall_demographics)

        demonstrations = []

        # stratify sample the train data frame for each test
        for row in tqdm(test_df.itertuples()):
            train_dems = self.stratified_sample_df(
                train_df,
                "filtered_demographics",
                self.shots,
                len(set_of_overall_demographics),
            )

            train_dems = train_dems["prompts"].tolist()[: self.shots]

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)

        return demonstrations, test_df
