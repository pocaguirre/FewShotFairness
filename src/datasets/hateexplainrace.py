from typing import List, Tuple

import pandas as pd

from .hatexplain import HateXplain


class HateXplainRace(HateXplain):
    def __init__(self, path: str) -> None:
        """Wrapper for HateXplainRace dataset

        :param path: path to folder containing dataset
        :type path: str
        """
        super().__init__(path)

        self.demographics = self.race_demographics

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Creates prompts for HateXplain Race

        :return: returns the train and test prompts, train demographics the labels for the test set and the demographic groups of the test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        """

        train_df, test_df = super().create_prompts()

        set_of_overall_demographics = set(self.demographics)

        # filter out demographics not focused on in train and test
        train_df["filtered_demographics"] = train_df["demographics"].apply(
            lambda x: self.filter_demographics(x, set_of_overall_demographics)
        )
        test_df["filtered_demographics"] = test_df["demographics"].apply(
            lambda x: self.filter_demographics(x, set_of_overall_demographics)
        )

        # remove them
        filtered_train_df = (
            train_df[train_df.filtered_demographics != ""].copy().reset_index()
        )

        filtered_test_df = (
            test_df[test_df.filtered_demographics != ""].copy().reset_index()
        )

        return filtered_train_df, filtered_test_df, self.demographics
