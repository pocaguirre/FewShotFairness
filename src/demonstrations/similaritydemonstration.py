from typing import List, Tuple

import pandas as pd

import faiss

from .semanticdemonstration import SemanticDemonstration


class SimilarityDemonstration(SemanticDemonstration):

    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)

    def create_demonstrations(self, train_df: pd.DataFrame, test_df: pd.DataFrame, overall_demographics: List[str]) -> Tuple[List[str], pd.DataFrame]:
        self.embed(train_df, test_df)

        vector_dim = self.train_vectors.shape[1]

        index = faiss.IndexFlatIP(vector_dim)

        faiss.normalize_L2(self.train_vectors)

        faiss.normalize_L2(self.test_vectors)

        index.add(self.train_vectors)

        distances, neighbors = index.search(self.test_vectors, self.shots)

        demonstrations = []

        for neighbor, row in zip(neighbors, test_df.itertuples()): 

            train_dems = train_df["prompts"].iloc[neighbor].tolist()

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)
        
        return demonstrations, test_df




