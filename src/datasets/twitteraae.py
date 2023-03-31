import os

import random

from typing import Tuple, List

import numpy as np

import pandas as pd

from .dataset import Dataset


class TwitterAAE(Dataset):
    def __init__(self, path: str):
        """Wrapper for TwitterAAE dataset

        :param path: path to TwitterAAE dataset
        :type path: str
        :raises ValueError: path is invalid
        """
        super().__init__(path)

        self.datasets = dict()

        self.label_map = {
            "pos_pos": ("happy", "aa"),
            "pos_neg": ("happy", "wh"),
            "neg_pos": ("sad", "aa"),
            "neg_neg": ("sad", "wh"),
        }

        if not os.path.exists(self.path):
            raise ValueError(f"Path to Twitter AAE data: {self.path} does not exist")

        self.datasets["train"] = []
        self.datasets["test"] = []

        # create in each split of twitter aae and combine them together
        for split in ["pos_pos", "pos_neg", "neg_pos", "neg_neg"]:
            train, test = self.read_data_file(os.path.join(self.path, split + "_text"))

            self.datasets["train"].extend(train)
            self.datasets["test"].extend(test)

        random.shuffle(self.datasets["train"])
        random.shuffle(self.datasets["test"])

        self.demographics = ["aa", "wh"]

    def build_prompt(self, text: str, label: str) -> str:
        """Create prompt for twitter aae

        :param text: input text
        :type text: str
        :param label: label for input text
        :type label: str
        :return: prompt using input text and label
        :rtype: str
        """
        return text + "\n the sentiment of this post is " + label

    def read_data_file(self, input_file: str) -> Tuple[List[str], List[str]]:
        """Read TwitterAAE data file

        Using split from
        https://github.com/HanXudong/fairlib/blob/main/data/src/Moji/deepmoji_split.py
        S
        :param input_file: path to input file
        :type input_file: str
        :return: training and testing example for split
        :rtype: str
        """

        with open(input_file, "r", encoding="latin-1") as f:
            lines = f.readlines()

        np.random.shuffle(lines)

        label_name = os.path.basename(input_file)[:-5]

        labels = self.label_map[label_name]

        train = list(zip(lines[:40000], [labels[0]] * 40000, [labels[1]] * 40000))
        test = list(zip(lines[42000:44000], [labels[0]] * 2000, [labels[1]] * 2000))

        return train, test

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Create prompts for HatExplain

        :return: Tuple of training prompts, testing prompts, test labels, and demographics for test set
        :rtype: Tuple[List[str], List[List[str]], List[str], List[str], List[List[str]], List[str]]
        """
        train_prompts = []

        train_demographics = []

        # create train prompts
        for item in self.datasets["train"]:

            # item 0 is text, item 1 is label
            prompt = self.build_prompt(item[0], item[1])

            train_prompts.append(prompt)

            train_demographics.append([item[2]])

        test_prompts = []

        test_labels = []

        test_demographics = []

        # create test prompts
        for item in self.datasets["test"]:
            prompt = self.build_prompt(item[0], "")

            test_prompts.append(prompt)

            test_labels.append(item[1])

            # item 2 is demographics
            test_demographics.append([item[2]])

        train_df = pd.DataFrame(
            {"prompts": train_prompts, "demographics": train_demographics}
        )

        test_df = pd.DataFrame(
            {
                "prompts": test_prompts,
                "demographics": test_demographics,
                "labels": test_labels,
            }
        )

        return train_df, test_df, self.demographics
