from collections import Counter

import json

import os

from typing import Tuple, List, Optional, Any, Dict, Set

import pandas as pd

from .dataset import Dataset


class HateXplain(Dataset):
    def __init__(self, path: str) -> None:
        """Wrapper for HateXplain Dataset

        :param path: path to HateXplain dataset
        :type path: str
        :raises ValueError: path does not exist
        """
        super().__init__(path)

        self.datasets: Dict[str, Any] = {}

        if not os.path.exists(self.path):
            raise ValueError(f"Path to HateXplain data: {self.path} does not exist")

        # read json
        data: Dict[str, Any] = json.load(
            open(os.path.join(self.path, "dataset.json"), "r")
        )

        # read splits
        splits: Dict[str, List[str]] = json.load(
            open(os.path.join(self.path, "post_id_divisions.json"), "r")
        )

        # split data into train and test
        self.datasets["train"] = [data[x] for x in splits["train"]]
        self.datasets["test"] = [data[x] for x in splits["test"]]

        for post in self.datasets["train"]:
            annotators = post["annotators"]

            for annotator in annotators:
                label = annotator["label"]

                if label == "normal":
                    annotator["label"] = "no"
                else:
                    annotator["label"] = "yes"

        for post in self.datasets["test"]:
            annotators = post["annotators"]

            for annotator in annotators:
                label = annotator["label"]

                if label == "normal":
                    annotator["label"] = "no"
                else:
                    annotator["label"] = "yes"

        self.race_demographics = ["African", "Arab", "Asian", "Hispanic", "Caucasian"]

        self.gender_demographics = ["Men", "Women"]

        self.labels = ['yes', 'no']

    def build_prompt(self, text: str, label: str) -> str:
        """Build prompts for HateXplain

        :param text: test example
        :type text: str
        :param label: label for text example
        :type label: str
        :return: build prompt for HateXplain
        :rtype: str
        """
        return (
            "Post: "
            + text
            + "\nQuestion: Does this post contain offensive language?\nAnswer: "
            + label
        )

    def get_majority(self, lst: List[str]) -> Optional[str]:
        """Helper to get majority of a list
        If there is no majority returns none

        Using method from
        https://github.com/INK-USC/CrossFit/blob/master/tasks/hatexplain.py

        :param lst: list of items to count
        :type lst: List[str]
        :return: returns highest counted item or none if there is a tie
        :rtype: Optional[str]
        """
        c = Counter(lst)
        rank = c.most_common()

        # if there is only one item return the item
        if len(rank) == 1:
            return rank[0][0]

        # if there is a tie return none
        elif rank[0][1] == rank[1][1]:
            return None

        # return the majority
        else:
            return rank[0][0]

    def choose_demographics(
        self, item_demographics: List[List[str]]
    ) -> Optional[List[str]]:
        """Choose the majority demographics in both race and gender columns

        :param item_demographics: list of demographics from annotators
        :type item_demographics: List[List[str]]
        :return: majority demographic in race and or gender or none if none is dominant
        :rtype: Optional[List[str]]
        """

        # flatten demographics
        item_demographics = [
            element for sublist in item_demographics for element in sublist
        ]

        # split race and gender demographics
        race_demographics = [
            x for x in item_demographics if x in self.race_demographics
        ]

        gender_demographics = [
            x for x in item_demographics if x in self.gender_demographics
        ]

        race_majority = None
        gender_majority = None

        # get majority in race and in gender
        if len(race_demographics) != 0:
            race_majority = self.get_majority(race_demographics)

        if len(gender_demographics) != 0:
            gender_majority = self.get_majority(gender_demographics)

        # if the majority is not none add it to overall demographics
        demographic = []

        if race_majority is not None:
            demographic.append(race_majority)
        if gender_majority is not None:
            demographic.append(gender_majority)

        if len(demographic) == 0:
            demographic = None

        return demographic

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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Creates prompts for HateXplain

        :return: returns the train and test prompts, train demographics the labels for the test set and the demographic groups of the test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]:
        """
        train_prompts = []

        train_demographics = []

        train_labels = []

        # create train prompts
        for item in self.datasets["train"]:
            labels = [x["label"] for x in item["annotators"]]

            item_demographics = [x["target"] for x in item["annotators"]]

            demographic = self.choose_demographics(item_demographics)

            # get majority label
            label = self.get_majority(labels)

            # if there is no majority label we remove it
            if label is not None and demographic is not None:
                sentence = " ".join(item["post_tokens"])

                prompt = self.build_prompt(sentence, label=label)

                train_prompts.append(prompt)

                train_demographics.append(demographic)

                train_labels.append(label)

        test_prompts = []

        test_labels = []

        test_demographics = []

        # create test prompts
        for item in self.datasets["test"]:
            labels = [x["label"] for x in item["annotators"]]

            item_demographics = [x["target"] for x in item["annotators"]]

            demographic = self.choose_demographics(item_demographics)

            label = self.get_majority(labels)

            if label is not None and demographic is not None:
                sentence = " ".join(item["post_tokens"])

                prompt = self.build_prompt(sentence, "")

                test_prompts.append(prompt)

                test_labels.append(label)

                test_demographics.append(demographic)

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

        return train_df, test_df
