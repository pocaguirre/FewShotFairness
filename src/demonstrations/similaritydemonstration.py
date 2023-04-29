from typing import List, Tuple

import pandas as pd

import faiss

from .semanticdemonstration import SemanticDemonstration


class SimilarityDemonstration(SemanticDemonstration):
    def __init__(self, shots: int = 16) -> None:
        """Similarity demonstration initalization

        :param shots: number of shots in demonstration, defaults to 16
        :type shots: int, optional
        """
        super().__init__(shots)

    def create_demonstrations(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        overall_demographics: List[str],
    ) -> Tuple[List[str], pd.DataFrame]:
        """Create demonstrations for test set by finding closest train set vectors

        :param train_df: train data
        :type train_df: pd.DataFrame
        :param test_df: test data
        :type test_df: pd.DataFrame
        :param overall_demographics: demographics to focus on
        :type overall_demographics: List[str]
        :return: demonstrations for test set with closest train set vectors as shots
        :rtype: Tuple[List[str], pd.DataFrame]
        """

        # embed our train and test df to vectors
        self.embed(train_df, test_df)

        vector_dim = self.train_vectors.shape[1]

        # create vector lookup for train
        index = faiss.IndexFlatIP(vector_dim)

        # normalize vector so that cosine similarity is an inner product
        faiss.normalize_L2(self.train_vectors)

        faiss.normalize_L2(self.test_vectors)

        # look for closest train vectors to test vectors
        index.add(self.train_vectors)

        distances, neighbors = index.search(self.test_vectors, self.shots)

        demonstrations = []

        # create prompts
        for neighbor, row in zip(neighbors, test_df.itertuples()):
            train_dems = train_df["prompts"].iloc[neighbor].tolist()

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)

        return demonstrations, test_df
