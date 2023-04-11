from typing import List

from tqdm import tqdm

import pandas as pd

from .demographicdemonstration import DemographicDemonstration


class ExcludingDemographic(DemographicDemonstration):
    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> List[str]:
        
        set_of_overall_demographics = set(overall_demographics)

        train_df["filtered_demographics"] = train_df["demographics"].apply(
            lambda x: self.filter_demographics(x, set_of_overall_demographics)
        )
        test_df["filtered_demographics"] = test_df["demographics"].apply(
            lambda x: self.filter_demographics(x, set_of_overall_demographics)
        )

        train_df = train_df[train_df.filtered_demographics != ""]

        test_df = test_df[test_df.filtered_demographics != ""]

        demonstrations = []

        pre_computed_exclusions = dict()

        for demographic in set_of_overall_demographics:
            pre_computed_exclusions[demographic] = train_df[~(train_df.filtered_demographics == demographic)]
        

        for row in tqdm(test_df.itertuples()):
            filtered_df = pre_computed_exclusions[row.filtered_demographics]

            train_dems = filtered_df["prompts"].sample(n=self.shots).tolist()

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)

        return demonstrations
