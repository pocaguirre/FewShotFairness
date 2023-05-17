from typing import List, Tuple

import pandas as pd

import faiss

from .semanticdemonstration import SemanticDemonstration

class ExcludingSimilarityDemonstration(SemanticDemonstration):
    def __init__(self, shots: int = 16) -> None:
        """Similarity demonstration initalization

        :param shots: number of shots in demonstration, defaults to 16
        :type shots: int, optional
        """
        SemanticDemonstration.__init__(shots)

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

        train_pre_computed_exclusions = dict()

        for demographic in set_of_overall_demographics:
            train_pre_computed_exclusions[demographic] = train_df[
                ~(train_df.filtered_demographics == demographic)
            ].index.tolist()

        test_pre_computed_exclusions = dict()

        for demographic in set_of_overall_demographics:
            test_pre_computed_exclusions[demographic] = test_df[
                ~(test_df.filtered_demographics == demographic)
            ].index.tolist()

        # embed our train and test df to vectors
        self.embed(train_df, test_df)

        vector_dim = self.train_vectors.shape[1]

        # normalize vector so that cosine similarity is an inner product
        faiss.normalize_L2(self.train_vectors)

        faiss.normalize_L2(self.test_vectors)

        train_demographic_vector_index_map = dict()

        for demographic in train_pre_computed_exclusions:
            demographic_index = train_pre_computed_exclusions[demographic]

            index = faiss.IndexFlatIP(vector_dim)

            index.add(self.train_vectors[demographic_index])

            train_demographic_vector_index_map[demographic] = index

        test_demographic_vector_map = dict()

        for demographic in test_pre_computed_exclusions:
            demographic_index = test_pre_computed_exclusions[demographic]

            test_demographic_vector_map[demographic] = self.test_vectors[
                demographic_index
            ]

        demonstrations = []

        for demographic in test_demographic_vector_map:
            test_vector_filtered = test_demographic_vector_map[demographic]

            train_vector_index = train_demographic_vector_index_map[demographic]

            distances, neighbors = train_vector_index.search(
                test_vector_filtered, self.shots
            )

            for neighbor, row in zip(neighbors, test_df.itertuples()):
                train_dems = train_df["prompts"].iloc[neighbor].tolist()

                demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)

        return demonstrations, test_df
