from collections import Counter

import json

import os

from typing import Tuple, List, Optional, Any, Dict

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

    def build_prompt(self, text: str, label: str) -> str:
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

    def create_prompts(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Creates prompts for HateXplain

        :return: returns the train and test prompts, train demographics the labels for the test set and the demographic groups of the test set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]:
        """
        train_prompts = []

        train_demographics = []

        # create train prompts
        for item in self.datasets["train"]:
            labels = [x["label"] for x in item["annotators"]]

            item_demographics = [x["target"] for x in item["annotators"]]

            item_demographics = [element for sublist in item_demographics for element in sublist]

            # get majority label
            label = self.get_majority(labels)

            demographic = self.get_majority(item_demographics)

            # if there is no majority label we remove it
            if label is not None and demographic is not None:
                sentence = " ".join(item["post_tokens"])

                prompt = self.build_prompt(sentence, label=label)

                train_prompts.append(prompt)

                train_demographics.append([demographic])

        test_prompts = []

        test_labels = []

        test_demographics = []

        # create test prompts
        for item in self.datasets["test"]:
            labels = [x["label"] for x in item["annotators"]]

            item_demographics = [x["target"] for x in item["annotators"]]

            # get all the demographics associated with the item
            item_demographics = [element for sublist in item_demographics for element in sublist]

            demographic = self.get_majority(item_demographics)

            label = self.get_majority(labels)

            if label is not None and demographic is not None:
                sentence = " ".join(item["post_tokens"])

                prompt = self.build_prompt(sentence, "")

                test_prompts.append(prompt)

                test_labels.append(label)

                test_demographics.append([demographic])

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

        return train_df, test_df
