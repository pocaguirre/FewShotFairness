import os

from typing import List, Tuple

from .dataset import Dataset

import pickle


class BiasInBios(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)

        self.filenames = ["train.pickle", "test.pickle"]

        self.datasets = {}

        if not os.path.exists(self.path):
            raise ValueError(f"Path to Bias in Bios data: {self.path} does not exist")

        for filename in self.filenames:
            filepath = os.path.join(self.path, filename)

            if not os.path.exists(filepath):
                raise ValueError(f"Bias in Bios data: {filepath} does not exist")

            self.datasets[filename[:-7]] = pickle.load(open(filepath, "rb"))

    def build_prompt(self, text: str, label: str) -> str:
        return text + " \n " + "The occupation of this person is " + label

    def create_prompts(self) -> Tuple[List[str], List[str], List[List[str]]]:
        train_prompts = []

        for item in self.datasets["train"]:
            prompt = self.build_prompt(
                item["text_without_gender"], item["p"]
            )

            train_prompts.append(prompt)

        test_prompts = []

        test_labels = []

        test_demographics = []

        for item in self.datasets["test"]:
            prompt = self.build_prompt(item["text_without_gender"], "")

            test_prompts.append(prompt)

            test_labels.append(item["p"])

            test_demographics.append([item["g"]])

        return train_prompts, test_prompts, test_labels, test_demographics
