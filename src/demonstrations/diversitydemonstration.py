from typing import List 

import pandas as pd

import faiss

from .semanticdemonstration import SemanticDemonstration

class DiversityDemonstration(SemanticDemonstration):

    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)

    def create_demonstrations(self, train_df: pd.DataFrame, test_df: pd.DataFrame, overall_demographics: List[str]) -> List[str]:
        self.embed(train_df, test_df)

        verbose = True
        vector_dim = self.train_vectors.shape[1]
        kmeans = faiss.Kmeans(vector_dim, self.shots, verbose=verbose)

        faiss.normalize_L2(self.train_vectors)

        faiss.normalize_L2(self.test_vectors)

        kmeans.train(self.train_vectors)

        index = faiss.IndexFlatIP(vector_dim)

        index.add(self.train_vectors)
        
        distances, neighbors = index.search(kmeans.centroids, 1)

        neighbors = [element for sublist in neighbors for element in sublist]

        train_dems = train_df["prompts"].iloc[neighbors].tolist()

        demonstrations = []

        for row in test_df.itertuples(): 

            demonstrations.append("\n\n".join(train_dems) + "\n\n" + row.prompts)
        
        return demonstrations




