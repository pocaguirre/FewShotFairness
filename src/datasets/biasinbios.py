import os

from typing import List, Tuple

from .dataset import Dataset

import pickle

import pandas as pd


class BiasInBios(Dataset):
    def __init__(self, path: str) -> None:
        """Wrapper for BiasInBios dataset

        :param path: path to folder containing dataset
        :type path: str
        :raises ValueError: path to dataset does not exist
        :raises ValueError: pickle file does not exist
        """
        super().__init__(path)

        self.filenames = ["train.pickle", "test.pickle"]

        self.datasets = {}

        if not os.path.exists(self.path):
            raise ValueError(f"Path to Bias in Bios data: {self.path} does not exist")

        for filename in self.filenames:
            filepath = os.path.join(self.path, filename)

            if not os.path.exists(filepath):
                raise ValueError(f"Bias in Bios data: {filepath} does not exist")

            self.datasets[filename[:-7]] = pd.DataFrame(
                pickle.load(open(filepath, "rb"))
            )

        # group test set by occuptations and gender and get the count of each
        counting_set = (
            self.datasets["test"][["g", "p", "text"]].groupby(["p", "g"]).count()
        )

        # get all professions that have at least 1000 female and 1000 male examples
        filtering_set = (counting_set["text"][:, "f"] > 1000) & (
            counting_set["text"][:, "m"] > 1000
        )

        # get the list of professions with above criteria
        self.labels = list(
            counting_set.loc[filtering_set[filtering_set == True].index]
            .reset_index()["p"]
            .unique()
        )

        # filter the test set with professions list
        self.datasets["test"] = self.datasets["test"][
            self.datasets["test"].p.isin(self.labels)
        ]

        # filter the train set with professions list
        self.datasets["train"] = self.datasets["train"][
            self.datasets["train"].p.isin(self.labels)
        ]

        # sample 500 of each profession from each gender
        self.datasets["test"] = (
            self.datasets["test"]
            .groupby(["p", "g"])
            .apply(lambda x: x.sample(500))
            .reset_index(drop=True)
        )

        self.demographics = ["m", "f"]

    def build_prompt(self, text: str, label: str) -> str:
        """Create prompt for bias in bios

        :param text: input text
        :type text: str
        :param label: label for input text
        :type label: str
        :return: prompt using input text and label
        :rtype: str
        """

        return (
            text
            + "\n "
            + "Occupations: "
            + ", ".join(self.labels)
            + "\nThe occupation of this person is "
            + label
        )

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Create prompts for Bias in Bios

        :return: Tuple of training prompts, training demographics testing prompts, test labels, and demographics for test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        """
        train_prompts = []

        train_demographics = []

        train_labels = []

        # build train dataset
        for item in self.datasets["train"].itertuples():
            prompt = self.build_prompt(item.hard_text, item.p)

            train_prompts.append(prompt)

            train_demographics.append([item.g])

            train_labels.append(item.p)

        test_prompts = []

        test_labels = []

        test_demographics = []

        # build test dataset
        for item in self.datasets["test"].itertuples():
            prompt = self.build_prompt(item.hard_text, "")

            test_prompts.append(prompt)

            test_labels.append(item.p)

            test_demographics.append([item.g])

        # format as dataframes
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

        return (
            filtered_train_df,
            filtered_test_df,
            self.demographics,
        )
