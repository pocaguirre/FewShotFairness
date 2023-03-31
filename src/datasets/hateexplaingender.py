from typing import List, Tuple

import pandas as pd

from .hatexplain import HateXplain


class HateXplainGender(HateXplain):
    def __init__(self, path: str) -> None:
        """Wrapper for HateXplainGender dataset

        :param path: path to folder containing dataset
        :type path: str
        """      
        super().__init__(path)

        self.demographics = [
            "African",
            "Arab",
            "Asian",
            "Hispanic",
            "Caucasian",
            "Indian",
            "Indigenous",
        ]

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Creates prompts for HateXplain

        :return: returns the train and test prompts, train demographics the labels for the test set and the demographic groups of the test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]:
        """        
        train_df, test_df = super().create_prompts()
        return train_df, test_df, self.demographics
