from typing import List

from tqdm import tqdm

import pandas as pd

from .demographicdemonstration import DemographicDemonstration


class ExcludingDemographic(DemographicDemonstration):
    def __init__(self, shots: int = 16) -> None:
        """Excluding demographic inititalization

        :param shots: shots in demonstration, defaults to 16
        :type shots: int, optional
        """
        super().__init__(shots)

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> List[str]:
        """Create demonstrations for test set using excluding demographic sampling

        :param train_df: train dataset
        :type train_df: pd.DataFrame
        :param test_df: test dataset
        :type test_df: pd.DataFrame
        :param overall_demographics: demographics that we are focusing on
        :type overall_demographics: List[str]
        :return: demonstrations created with excluding sampling
        :rtype: Tuple[List[str], pd.DataFrame]
        """

        set_of_overall_demographics = set(overall_demographics)

        demonstrations = []

        # compute the dataframes for the excluding each demographic
        pre_computed_exclusions = dict()

        for demographic in set_of_overall_demographics:
            pre_computed_exclusions[demographic] = train_df[
                ~(train_df.filtered_demographics == demographic)
            ]

        # create prompts
        for row in tqdm(test_df.itertuples()):
            filtered_df = pre_computed_exclusions[row.filtered_demographics]

            train_dems = filtered_df["prompts"].sample(n=self.shots).tolist()

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)

        return demonstrations, test_df
