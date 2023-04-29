from typing import List

from .demonstration import Demonstration

from sentence_transformers import SentenceTransformer

import pandas as pd


class SemanticDemonstration(Demonstration):
    def __init__(self, shots: int = 16) -> None:
        """Base demonstration for semantic focused demonstrations

        :param shots: number of shots in demonstration, defaults to 16
        :type shots: int, optional
        """
        super().__init__(shots)

        # select embedding type
        self.embedding = SentenceTransformer("all-mpnet-base-v2")

        self.train_vectors = None

        self.test_vectors = None

        self.type = "semantic"

    def embed(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """embeds train and test dataframes using sentence transformers

        :param train_df: train set of prompts
        :type train_df: pd.DataFrame
        :param test_df: test set of prompts
        :type test_df: pd.DataFrame
        """
        self.train_vectors = self.embedding.encode(
            train_df["prompts"].tolist(), batch_size=32, show_progress_bar=True
        )

        self.test_vectors = self.embedding.encode(
            test_df["prompts"].tolist(), batch_size=32, show_progress_bar=True
        )
