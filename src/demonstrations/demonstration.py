from typing import List

import pandas as pd


class Demonstration:
    def __init__(self, shots: int = 16) -> None:

        self.shots = shots

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> List[str]:
        pass
