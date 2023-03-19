import os

from typing import Tuple, List

import json

from .dataset import Dataset

from collections import Counter


class HatExplain(Dataset):
    def __init__(self, path: str):
        super().__init__(path)

        self.datasets = {}

        if not os.path.exists(self.path):
            raise ValueError(f"Path to HateExplain data: {self.path} does not exist")

        data = json.load(open(os.path.join(self.path, "dataset.json"), "r"))

        splits = json.load(open(os.path.join(self.path, "post_id_divisions.json"), "r"))

        self.datasets["train"] = [data[x] for x in splits["train"]]
        self.datasets["test"] = [data[x] for x in splits["test"]]

    def build_prompt(self, text: str, label: str) -> str:
        return text + " \n " + label

    def get_majority(self, lst: List[str]) -> str:
        """
        Using method from 
        https://github.com/INK-USC/CrossFit/blob/master/tasks/hatexplain.py
        """
        c = Counter(lst)
        rank = c.most_common()
        if len(rank) == 1:
            return rank[0][0]
        elif rank[0][1] == rank[1][1]:
            return None
        else:
            return rank[0][0]

    def create_prompts(self) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
        train_prompts = []

        for item in self.datasets["train"]:
            labels = [x["label"] for x in item["annotators"]]

            label = self.get_majority(labels)

            if label is not None:
                if label == "hatespeech":
                    label = "hate speech"

                sentence = " ".join(item["post_tokens"])

                prompt = self.build_prompt(sentence, label=label)

                train_prompts.append(prompt)

        test_prompts = []

        test_labels = []

        test_demographics = []

        for item in self.datasets["test"]:
            labels = [x["label"] for x in item["annotators"]]

            item_demographics = list(set([x['label'] for x in item['annotators']]))

            label = self.get_majority(labels)

            if label is not None:
                if label == "hatespeech":
                    label = "hate speech"

                sentence = " ".join(item["post_tokens"])

                prompt = self.build_prompt(sentence, "")

                test_prompts.append(prompt)

                test_labels.append(label)

                test_demographics.append(item_demographics)

        return train_prompts, test_prompts, test_labels, test_demographics
