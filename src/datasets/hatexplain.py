import os

from typing import Tuple, List

import json

from .dataset import Dataset

from collections import Counter

class Hatexplain(Dataset):

    def __init__(self, path: str, prompt: str):

        super().__init__(path, prompt)

        self.datasets = {}

        if not os.path.exists(self.path):

            raise ValueError(f"Path to HateExplain data: {self.path} does not exist")

        data = json.load(open(os.path.join(self.path, "dataset.json"),'r'))

        splits = json.load(open(os.path.join(self.path, "post_id_divisions.json"),'r'))

        self.datasets['train']= [data[x] for x in splits['train']]
        self.datasets['test']= [data[x] for x in splits['test']]
    
    def get_majority(self, lst: List[str]) -> str:
        c = Counter(lst)
        rank = c.most_common()
        if len(rank) == 1:
            return rank[0][0]
        elif rank[0][1] == rank[1][1]:
            return None
        else:
            return rank[0][0]

    def create_prompts(self) -> Tuple[List[str], List[str], List[str]]:
        
        train_prompts = []

        for item in self.datasets['train']:

            labels = [x['label'] for x in item["annotators"]]

            label = self.get_majority(labels)

            if label is not None:

                text = " ".join(item["post_tokens"])

                prompt = self.prompt.format(text = text, label = label)

                train_prompts.append(prompt)

        test_prompts = []

        test_labels = []

        for item in self.datasets['test']:

            labels = [x['label'] for x in item["annotators"]]

            label = self.get_majority(labels)

            if label is not None:

                text = " ".join(item["post_tokens"])

                prompt = self.prompt.format(text = text, label = "")

                test_prompts.append(prompt)

                test_labels.append(label)
            
        return train_prompts, test_prompts, test_labels