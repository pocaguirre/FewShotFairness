from typing import List, Tuple

import pandas as pd

import faiss

from .semanticdemonstration import SemanticDemonstration


class DiversityDemonstration(SemanticDemonstration):
    def __init__(self, shots: int = 16) -> None:
        """Diversity demonstration class

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
        """Create demonstrations for test set using diversity sampling of train set

        :param train_df: train dataset
        :type train_df: pd.DataFrame
        :param test_df: test dataset
        :type test_df: pd.DataFrame
        :param overall_demographics: demographics that we are focusing on
        :type overall_demographics: List[str]
        :return: demonstrations created with diversity sampling
        :rtype: Tuple[List[str], pd.DataFrame]
        """

        # create vectors from test and train df
        self.embed(train_df, test_df)

        # create kmeans clusterer
        verbose = True
        vector_dim = self.train_vectors.shape[1]
        kmeans = faiss.Kmeans(vector_dim, self.shots, verbose=verbose)

        # normalize the vectors in order to make cosine similarity a dot product
        faiss.normalize_L2(self.train_vectors)

        faiss.normalize_L2(self.test_vectors)

        # train kmeans
        kmeans.train(self.train_vectors)

        # create vector lookup for train set
        index = faiss.IndexFlatIP(vector_dim)

        index.add(self.train_vectors)

        # get train items nearest to each centroid
        distances, neighbors = index.search(kmeans.centroids, 1)

        # create prompts
        neighbors = [element for sublist in neighbors for element in sublist]

        train_dems = train_df["prompts"].iloc[neighbors].tolist()

        demonstrations = []

        for row in test_df.itertuples():
            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)

        return demonstrations, test_df
