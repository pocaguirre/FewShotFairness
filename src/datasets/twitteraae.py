import os

import random

from typing import Tuple, List

import numpy as np

import pandas as pd

from .dataset import Dataset


class TwitterAAE(Dataset):
    def __init__(self, path: str, type_of_prompt: str = "no_change"):
        """Wrapper for TwitterAAE dataset

        :param path: path to TwitterAAE dataset
        :type path: str
        :raises ValueError: path is invalid
        """
        super().__init__(path, type_of_prompt)

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

        self.labels = ["sad", "happy"]

    def build_prompt(self, text: str, label: str) -> str:
        """Create prompt for twitter aae

        :param text: input text
        :type text: str
        :param label: label for input text
        :type label: str
        :return: prompt using input text and label
        :rtype: str
        """
        return (
            "Post: " + text + "\nQuestion: Is this post happy or sad?\nAnswer: " + label
        )

    def build_protected_prompt(
        self, text: str, label: str, protected_category: List[str]
    ) -> str:
        filtered_protected_category = self.filter_demographics(
            protected_category, self.demographics
        )

        return (
            "Post: "
            + text
            + "\nThe person posting this is "
            + filtered_protected_category
            + "\nQuestion: Is this post happy or sad?\nAnswer: "
            + label
        )

    def build_expert_prompt(self, text: str, label: str) -> str:
        return (
            "Post: "
            + text
            + "\nQuestion: Is this post happy or sad?"
            + "\nRemember you are an expert in the field"
            + "\nAnswer: "
            + label
        )

    def build_fairness_prompt(self, text: str, label: str) -> str:
        return (
            "Post: "
            + text
            + "\nQuestion: Is this post happy or sad?"
            + "\nRemember to keep fairness in mind"
            + "\nAnswer: "
            + label
        )

    def read_data_file(self, input_file: str) -> Tuple[List[str], List[str]]:
        """Read TwitterAAE data file

        Using split from
        https://github.com/HanXudong/fairlib/blob/main/data/src/Moji/deepmoji_split.py
        S
        :param input_file: path to input file
        :type input_file: str
        :return: training and testing example for split
        :rtype: Tuple[List[str], List[str]]
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

        :return:  returns the train and test prompts, train demographics the labels for the test set and the demographic groups of the test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        """

        train_prompts = []

        train_demographics = []

        train_labels = []

        # create train prompts
        for item in self.datasets["train"]:
            # item 0 is text, item 1 is label
            if self.type_of_prompt == "protected_category":
                prompt = self.build_protected_prompt(item[0], item[1], [item[2]])
            else:
                prompt = self.build_prompt(item[0], item[1])

            train_prompts.append(prompt)

            train_demographics.append([item[2]])

            train_labels.append(item[1])

        test_prompts = []

        test_labels = []

        test_demographics = []

        # create test prompts
        for item in self.datasets["test"]:
            if self.type_of_prompt == "protected_category":
                prompt = self.build_protected_prompt(item[0], "", [item[2]])

            elif self.type_of_prompt == "expert":
                prompt = self.build_expert_prompt(item[0], "")

            elif self.type_of_prompt == "fairness":
                prompt = self.build_fairness_prompt(item[0], "")

            else:
                prompt = self.build_prompt(item[0], "")

            test_prompts.append(prompt)
            test_labels.append(item[1])

            # item 2 is demographics
            test_demographics.append([item[2]])

        # put into dataframes
        train_df = pd.DataFrame(
            {
                "prompts": train_prompts,
                "demographics": train_demographics,
                "labels": train_labels,
            }
        )

        test_df = pd.DataFrame(
            {
                "prompts": test_prompts,
                "demographics": test_demographics,
                "labels": test_labels,
            }
        )

        set_of_overall_demographics = set(self.demographics)

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
