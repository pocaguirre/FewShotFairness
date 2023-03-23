import os

from typing import Tuple, List, Optional, Any, Dict

import json

from .dataset import Dataset

from collections import Counter


class HatExplain(Dataset):
    def __init__(self, path: str) -> None:
        """Wrapper for HatExplain Dataset

        :param path: path to HatExplain dataset
        :type path: str
        :raises ValueError: path does not exist
        """        
        super().__init__(path)

        self.datasets: Dict[str, Any] = {}

        if not os.path.exists(self.path):
            raise ValueError(f"Path to HateExplain data: {self.path} does not exist")
        
        # read json 
        data: Dict[str, Any] = json.load(open(os.path.join(self.path, "dataset.json"), "r"))

        # read splits
        splits: Dict[str, List[str]] = json.load(open(os.path.join(self.path, "post_id_divisions.json"), "r"))

        # split data into train and test
        self.datasets["train"] = [data[x] for x in splits["train"]]
        self.datasets["test"] = [data[x] for x in splits["test"]]

    def build_prompt(self, text: str, label: str) -> str:
        return text + " \n " + label

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

    def create_prompts(self) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
        """Creates prompts for HateExplain

        :return: returns the train and test prompts, the labels for the test set and the demographic groups of the test set
        :rtype: Tuple[List[str], List[str], List[str], List[List[str]]]
        """        
        train_prompts = []

        # create train prompts
        for item in self.datasets["train"]:
            labels = [x["label"] for x in item["annotators"]]

            # get majority label
            label = self.get_majority(labels)

            # if there is no majority label we remove it
            if label is not None:
                if label == "hatespeech":
                    label = "hate speech"

                sentence = " ".join(item["post_tokens"])

                prompt = self.build_prompt(sentence, label=label)

                train_prompts.append(prompt)

        test_prompts = []

        test_labels = []

        test_demographics = []

        # create test prompts
        for item in self.datasets["test"]:
            labels = [x["label"] for x in item["annotators"]]

            item_demographics = [x['target'] for x in item['annotators']]

            # get all the demographics associated with the item 
            item_demographics = list(set([element for sublist in item_demographics for element in sublist]))

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
