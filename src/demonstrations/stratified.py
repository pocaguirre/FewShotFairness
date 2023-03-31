import math

from typing import List, Set

import pandas as pd

from tqdm import tqdm

from .demonstration import Demonstration



class StratifiedSampler(Demonstration):
    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)
    
    def stratified_sample_df(self, df: pd.DataFrame, col: str, n_samples: int, number_of_demographics: int):
        df_ = df.groupby(col).apply(lambda x: x.sample(math.ceil(n_samples/number_of_demographics)))
        df_.index = df_.index.droplevel(0)
        return df_
    
    def filter_demographics(self, demographics: List[str], overall_demographics: Set[str]) -> str:

        set_of_demographics = set(demographics)

        intersection = set_of_demographics.intersection(overall_demographics)

        if len(intersection) == 0:
            return ""

        else: 
            return list(intersection)[0]

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> List[str]:
        
        set_of_overall_demographics = set(overall_demographics)

        train_df['filtered_demographics'] = train_df['demographics'].apply(lambda x: self.filter_demographics(x, set_of_overall_demographics))
        test_df['filtered_demographics'] = train_df['demographics'].apply(lambda x: self.filter_demographics(x, set_of_overall_demographics))

        train_df = train_df[train_df.filtered_demographics != ""]

        test_df = test_df[test_df.filtered_demographics != ""]

        demonstrations = []

        for row in tqdm(test_df.itertuples()):
            train_dems = self.stratified_sample_df(train_df, "demographics", self.shots, len(set_of_overall_demographics))

            train_dems = train_dems['prompts'].tolist()[:self.shots]

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)
            
        return demonstrations




