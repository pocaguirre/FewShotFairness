from typing import Tuple, List, Set

import pandas as pd


class Dataset:
    """Base Class for datasets"""

    def __init__(self, path: str):
        """Base Class initalizer

        :param path: path to dataset
        :type path: str
        """
        self.path = path

    def build_prompt(self, text: str, label: str) -> str:
        """Create prompt from input text and label

        :param text: input text for dataset
        :type text: str
        :param label: classification label for input text
        :type label: str
        :return: prompt containing text and label
        :rtype: str
        """
        pass
    
    def filter_demographics(
        self, demographics: List[str], overall_demographics: Set[str]
    ) -> str:
        """filter demographics that we are focusing on

        :param demographics: demographics of that post
        :type demographics: List[str]
        :param overall_demographics: demographics to focus on
        :type overall_demographics: Set[str]
        :return: first item in intersection
        :rtype: str
        """

        set_of_demographics = set(demographics)

        intersection = set_of_demographics.intersection(overall_demographics)

        if len(intersection) == 0:
            return ""

        else:
            return list(intersection)[0]

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Creates prompt from train and test datasets

        :return: training prompts, training demographics, testing prompts, test labels, and demographics for test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        """
        pass
