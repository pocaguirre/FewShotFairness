import random

from typing import List

from tqdm import tqdm

import pandas as pd

from .demonstration import Demonstration

class WithinDemographic(Demonstration):
    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> List[str]:
        demonstrations = []

        set_of_overall_demographics = set(overall_demographics)

        for row in tqdm(test_df.itertuples()):

            row_demographics = list(set(row.demographics).intersection(set_of_overall_demographics))
            
            filtered_df = train_df[train_df.demographics.str.contains('|'.join(row_demographics))]

            train_dems = filtered_df['prompts'].sample(n=self.shots).tolist()

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)
            
        return demonstrations
