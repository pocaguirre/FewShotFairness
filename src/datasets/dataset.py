from typing import Tuple, List

import pandas as pd


class Dataset:
    """Base Class for datasets"""

    def __init__(self, path: str):
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

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Creates prompt from train and test datasets

        :return: training prompts, training demographics, testing prompts, test labels, and demographics for test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        """
        pass
