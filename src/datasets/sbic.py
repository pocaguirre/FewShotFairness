import os

from typing import Tuple, List

import pandas as pd

from .dataset import Dataset



class SBIC(Dataset):

    def __init__(self, path: str, prompt: str):
        super().__init__(path, prompt)

        self.datasets = {}

        if not os.path.exists(self.path):

            raise ValueError(f"Path to SBIC data: {self.path} does not exist")

        self.datasets['train'] = pd.read_csv(os.path.join(self.path, "SBIC.v2.trn.csv"))
        self.datasets['test'] = pd.read_csv(os.path.join(self.path, "SBIC.v2.tst.csv"))


    def create_prompts(self) -> Tuple[List[str], List[str], List[str]]:
        
        train_prompts = []

        for item in self.datasets['train'].itertuples():

            label = "Yes" if item.offensiveYN == 1.0 else "No"

            prompt = self.prompt.format(text = item.post, label = label)

            train_prompts.append(prompt)
        
        test_prompts = []

        test_labels = []

        for item in self.datasets['test'].itertuples():
            label = "Yes" if item.offensiveYN == 1.0 else "No"

            prompt = self.prompt.format(text = item.post, label = "")

            test_prompts.append(prompt)

            test_labels.append(label)
            
        return train_prompts, test_prompts, test_labels
