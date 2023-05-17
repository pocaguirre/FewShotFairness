from typing import List, Tuple

import pandas as pd

import faiss

from .semanticdemonstration import SemanticDemonstration

class ExcludingDiversityDemonstration(SemanticDemonstration):
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

        set_of_overall_demographics = set(overall_demographics)

        train_pre_computed_inclusions = dict()

        for demographic in set_of_overall_demographics:
            train_pre_computed_inclusions[demographic] = train_df[
                ~(train_df.filtered_demographics == demographic)
            ].index.tolist()

        # embed our train and test df to vectors
        self.embed(train_df, test_df)

        vector_dim = self.train_vectors.shape[1]

        # normalize vector so that cosine similarity is an inner product
        faiss.normalize_L2(self.train_vectors)

        faiss.normalize_L2(self.test_vectors)

        diverse_prompts_demographic_map = dict()

        for demographic in train_pre_computed_inclusions:
            demographic_index = train_pre_computed_inclusions[demographic]

            filtered_train_vectors = self.train_vectors[demographic_index]

            kmeans = faiss.Kmeans(vector_dim, self.shots, verbose=True)

            index = faiss.IndexFlatIP(vector_dim)
            kmeans.train(filtered_train_vectors)

            index.add(filtered_train_vectors)

            distances, neighbors = index.search(kmeans.centroids, 1)

            neighbors = [element for sublist in neighbors for element in sublist]

            train_dems = train_df["prompts"].iloc[neighbors].tolist()

            diverse_prompts_demographic_map[demographic] = train_dems

        demonstrations = []

        for row in test_df.itertuples():
            train_dems = diverse_prompts_demographic_map[row.filtered_demographics]

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)

        return demonstrations, test_df
