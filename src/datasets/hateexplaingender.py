from typing import List, Tuple, Set

import pandas as pd

from .hatexplain import HateXplain


class HateXplainGender(HateXplain):
    def __init__(self, path: str, type_of_prompt: str = "no_change") -> None:
        """Wrapper for HateXplainGender dataset

        :param path: path to folder containing dataset
        :type path: str
        """
        super().__init__(path, type_of_prompt)

        self.demographics = self.gender_demographics

    def build_protected_prompt(
        self, text: str, label: str, protected_category: List[str]
    ) -> str:
        filtered_protected_category = self.filter_demographics(
            protected_category, self.demographics
        )

        return (
            "Post: "
            + text
            + "\nThe person being attacked is "
            + filtered_protected_category
            + "\nQuestion: Does this post contain offensive language?\nAnswer: "
            + label
        )

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """ "Creates prompts for HateXplain Gender

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
