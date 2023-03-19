import random

import os

from typing import Tuple, List

import numpy as np

from .dataset import Dataset


class TwitterAAE(Dataset):
    def __init__(self, path: str):
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

        for split in ["pos_pos", "pos_neg", "neg_pos", "neg_neg"]:
            train, test = self.read_data_file(os.path.join(self.path, split + "_text"))

            self.datasets["train"].extend(train)
            self.datasets["test"].extend(test)

        random.shuffle(self.datasets["train"])
        random.shuffle(self.datasets["test"])

    def build_prompt(self, text: str, label: str) -> str:
        return text + " \n " + label

    def read_data_file(self, input_file: str):
        """
        Using split from 
        https://github.com/HanXudong/fairlib/blob/main/data/src/Moji/deepmoji_split.py
        """
        with open(input_file, "r", encoding="latin-1") as f:
            lines = f.readlines()

        np.random.shuffle(lines)

        label_name = os.path.basename(input_file)[:-5]

        labels = self.label_map[label_name]

        train = list(zip(lines[:40000], [labels[0]] * 40000, [labels[1]] * 40000))
        test = list(zip(lines[42000:44000], [labels[0]] * 2000, [labels[1]] * 2000))

        return train, test

    def create_prompts(self) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
        train_prompts = []

        for item in self.datasets["train"]:
            prompt = self.build_prompt(item[0], item[1])

            train_prompts.append(prompt)

        test_prompts = []

        test_labels = []

        test_demographics = []

        for item in self.datasets["test"]:
            prompt = self.build_prompt(item[0], "")

            test_prompts.append(prompt)

            test_labels.append(item[1])

            test_demographics.append([item[2]])

        return train_prompts, test_prompts, test_labels, test_demographics